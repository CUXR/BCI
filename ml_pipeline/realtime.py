#!/usr/bin/env python3
"""Real-time BCI system with Muse 2 via BrainFlow.

Connects to Muse 2 headband, processes EEG in real-time, and classifies
brain states: blink detection + motor imagery (left/right grab).

Prerequisites:
    python train.py   # trains and saves models first

Usage:
    python realtime.py              # live Muse 2
    python realtime.py --simulate   # synthetic board for testing
    python realtime.py --headless   # terminal-only, no GUI
"""

import sys
import time
import pickle
import argparse
import warnings
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

from brainflow.board_shim import (
    BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets,
)
from brainflow.data_filter import DataFilter, DetrendOperations

from config import (
    TARGET_SFREQ, CH_NAMES, N_CHANNELS, MODELS_DIR,
    RT_BLINK_WINDOW_S, RT_MI_WINDOW_S, RT_UPDATE_INTERVAL_MS,
    RT_BLINK_COOLDOWN_S, RT_MI_COOLDOWN_S,
    RT_BLINK_THRESHOLD, RT_MI_THRESHOLD,
    MI_BANDPASS, BLINK_BANDPASS, NOTCH_FREQ,
)
from preprocessing import (
    preprocess_mi_epoch, preprocess_blink_epoch,
    interpolate_nans,
)
from features import extract_mi_features, extract_blink_features

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


# ═══════════════════════════════════════════════════════════════════
# Real-time classifier
# ═══════════════════════════════════════════════════════════════════

class RealtimeClassifier:
    """Manages model loading, preprocessing, and real-time inference."""

    def __init__(self, models_path=None):
        self.models_path = models_path or (MODELS_DIR / "realtime_models.pkl")
        self.bundle = None
        self.last_blink_time = 0
        self.last_mi_time = 0
        self.blink_threshold = RT_BLINK_THRESHOLD
        self.mi_threshold = RT_MI_THRESHOLD

    def load_models(self):
        if not self.models_path.exists():
            print(f"ERROR: Model file not found: {self.models_path}")
            print("Run 'python train.py' first.")
            return False

        with open(self.models_path, "rb") as f:
            self.bundle = pickle.load(f)

        print(f"Models loaded (trained {self.bundle['training_date'][:10]} "
              f"on {self.bundle['n_subjects']} subjects: {self.bundle['subjects']})")
        return True

    def _extract_window(self, board_data, eeg_channels, ch_indices, n_samples):
        """Extract EEG window from BrainFlow board data.

        Returns (n_channels, n_samples).
        """
        window = np.zeros((len(ch_indices), n_samples))
        for i, ch_idx in enumerate(ch_indices):
            if ch_idx < len(eeg_channels):
                raw = board_data[eeg_channels[ch_idx], -n_samples:]
                window[i] = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return window

    def classify_blink(self, board_data, eeg_channels):
        """Check for intentional blink. Returns (is_blink, confidence)."""
        now = time.time()
        if now - self.last_blink_time < RT_BLINK_COOLDOWN_S:
            return False, 0.0

        if self.bundle is None or self.bundle.get("blink_pipeline") is None:
            return False, 0.0

        ch_indices = self.bundle["blink_ch_indices"]
        n_samples = int(RT_BLINK_WINDOW_S * TARGET_SFREQ)

        if board_data.shape[1] < n_samples:
            return False, 0.0

        eeg_window = self._extract_window(board_data, eeg_channels, ch_indices, n_samples)
        cleaned = preprocess_blink_epoch(eeg_window, TARGET_SFREQ)

        features = extract_blink_features(cleaned, TARGET_SFREQ, ch_indices)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        X = features.reshape(1, -1)

        try:
            proba = self.bundle["blink_pipeline"].predict_proba(X)[0]
            blink_prob = float(proba[1]) if len(proba) > 1 else 0.0
            is_blink = blink_prob > self.blink_threshold

            if is_blink:
                self.last_blink_time = now
            return is_blink, blink_prob
        except Exception:
            return False, 0.0

    def classify_mi(self, board_data, eeg_channels):
        """Classify motor imagery. Returns (prediction, confidence).

        prediction: "left", "right", or "idle"
        """
        now = time.time()
        if now - self.last_mi_time < RT_MI_COOLDOWN_S:
            return "idle", 0.5

        if self.bundle is None or self.bundle.get("mi_pipeline") is None:
            return "idle", 0.5

        ch_indices = self.bundle["mi_ch_indices"]
        n_samples = int(RT_MI_WINDOW_S * TARGET_SFREQ)

        if board_data.shape[1] < n_samples:
            return "idle", 0.5

        eeg_window = self._extract_window(board_data, eeg_channels, ch_indices, n_samples)
        cleaned = preprocess_mi_epoch(eeg_window, TARGET_SFREQ)

        features = extract_mi_features(cleaned, TARGET_SFREQ, ch_indices)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        W = self.bundle.get("mi_csp_W")
        if W is not None:
            projected = W.T @ cleaned
            csp_feat = np.log(np.var(projected, axis=1) + 1e-12)
            features = np.concatenate([features, csp_feat])

        X = features.reshape(1, -1)

        try:
            proba = self.bundle["mi_pipeline"].predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])

            self.last_mi_time = now

            if confidence < self.mi_threshold:
                return "idle", confidence

            return ("left" if pred_idx == 0 else "right"), confidence
        except Exception:
            return "idle", 0.5


# ═══════════════════════════════════════════════════════════════════
# Headless mode (terminal output)
# ═══════════════════════════════════════════════════════════════════

def run_headless(board, eeg_channels, classifier, sr):
    """Simple terminal-based real-time loop."""
    print("\nRunning in headless mode. Press Ctrl+C to stop.\n")
    print(f"{'Time':>10s}  {'State':>8s}  {'Confidence':>10s}  {'Blink':>6s}")
    print("-" * 42)

    try:
        while True:
            n_needed = int(max(RT_MI_WINDOW_S, 3.0) * sr)
            data = board.get_current_board_data(n_needed, BrainFlowPresets.DEFAULT_PRESET)

            if data.shape[1] < int(RT_BLINK_WINDOW_S * sr):
                time.sleep(0.05)
                continue

            is_blink, blink_conf = classifier.classify_blink(data, eeg_channels)

            if is_blink:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"{ts:>10s}  {'BLINK':>8s}  {blink_conf:>10.0%}  {'***':>6s}")
            elif data.shape[1] >= int(RT_MI_WINDOW_S * sr):
                mi_pred, mi_conf = classifier.classify_mi(data, eeg_channels)
                if mi_pred != "idle":
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"{ts:>10s}  {mi_pred.upper():>8s}  {mi_conf:>10.0%}")

            time.sleep(RT_UPDATE_INTERVAL_MS / 1000.0)

    except KeyboardInterrupt:
        print("\nStopped.")


# ═══════════════════════════════════════════════════════════════════
# GUI mode (PyQt6)
# ═══════════════════════════════════════════════════════════════════

def run_gui(board, eeg_channels, classifier, sr):
    """Launch the PyQt6 GUI for visual feedback."""
    from PyQt6 import QtWidgets, QtCore, QtGui
    import pyqtgraph as pg

    class BCIWindow(QtWidgets.QMainWindow):
        COLORS = {
            "idle": "#666666", "left": "#2196F3",
            "right": "#4CAF50", "blink": "#FF9800",
        }

        def __init__(self):
            super().__init__()
            self.current_state = "idle"
            self.blink_flash = 0
            self.is_running = False

            self.setWindowTitle("BCI Real-Time — ML Pipeline")
            self.setMinimumSize(800, 600)
            self.setStyleSheet("background-color: #1a1a2e; color: #e0e0e0;")
            self._build_ui()

            self.timer = QtCore.QTimer()
            self.timer.setInterval(RT_UPDATE_INTERVAL_MS)
            self.timer.timeout.connect(self._update)

        def _build_ui(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QVBoxLayout(central)
            layout.setSpacing(10)
            layout.setContentsMargins(16, 16, 16, 16)

            # Controls row
            ctrl = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton("Start")
            self.start_btn.setFixedWidth(100)
            self.start_btn.setStyleSheet(
                "QPushButton { background: #4CAF50; color: white; padding: 8px; "
                "border-radius: 4px; font-weight: bold; font-size: 14px; }"
                "QPushButton:hover { background: #45a049; }")
            self.start_btn.clicked.connect(self._toggle)
            ctrl.addWidget(self.start_btn)

            self.status_lbl = QtWidgets.QLabel("Ready — press Start")
            self.status_lbl.setStyleSheet("font-size: 13px; color: #aaa; padding: 0 12px;")
            ctrl.addWidget(self.status_lbl)
            ctrl.addStretch()
            layout.addLayout(ctrl)

            # EEG plot
            self.eeg_plot = pg.PlotWidget()
            self.eeg_plot.setTitle("Raw EEG", color="w", size="11pt")
            self.eeg_plot.setBackground("#0d0d1a")
            self.eeg_plot.setMaximumHeight(200)
            self.eeg_plot.getAxis("left").setTicks([])
            colors = ["#00BCD4", "#FFFFFF", "#FFEB3B", "#8BC34A"]
            self.eeg_curves = {}
            for i, ch in enumerate(eeg_channels[:4]):
                pen = pg.mkPen(color=colors[i % len(colors)], width=1)
                label = CH_NAMES[i] if i < len(CH_NAMES) else f"Ch{ch}"
                self.eeg_curves[ch] = self.eeg_plot.plot([], [], pen=pen, name=label)
            self.eeg_plot.addLegend(offset=(10, 10))
            layout.addWidget(self.eeg_plot)

            # State display
            frame = QtWidgets.QFrame()
            frame.setStyleSheet(
                "QFrame { background: #0d0d1a; border: 2px solid #333; "
                "border-radius: 10px; }")
            v = QtWidgets.QVBoxLayout(frame)
            v.setContentsMargins(20, 20, 20, 20)

            self.state_label = QtWidgets.QLabel("IDLE")
            self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.state_label.setStyleSheet(
                "font-size: 72px; font-weight: bold; color: #666; "
                "background: transparent;")
            v.addWidget(self.state_label)

            self.arrow_label = QtWidgets.QLabel("")
            self.arrow_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.arrow_label.setStyleSheet(
                "font-size: 48px; color: #444; background: transparent;")
            v.addWidget(self.arrow_label)

            # Confidence bars
            bars = QtWidgets.QHBoxLayout()
            self.blink_bar = self._make_bar("Blink", "#FF9800")
            self.mi_bar = self._make_bar("MI Confidence", "#2196F3")
            bars.addLayout(self.blink_bar["layout"])
            bars.addLayout(self.mi_bar["layout"])
            v.addLayout(bars)

            layout.addWidget(frame, stretch=1)

            # Log
            self.log = QtWidgets.QListWidget()
            self.log.setMaximumHeight(120)
            self.log.setStyleSheet(
                "QListWidget { background: #0d0d1a; border: 1px solid #333; "
                "border-radius: 6px; font-family: monospace; font-size: 12px; }")
            layout.addWidget(self.log)

        def _make_bar(self, title, color):
            layout = QtWidgets.QVBoxLayout()
            lbl = QtWidgets.QLabel(title)
            lbl.setStyleSheet("font-size: 11px; color: #aaa; background: transparent;")
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%v%")
            bar.setFixedHeight(22)
            bar.setStyleSheet(
                f"QProgressBar {{ background: #1a1a2e; border: 1px solid #444; "
                f"border-radius: 4px; text-align: center; color: white; }}"
                f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}")
            layout.addWidget(bar)
            return {"layout": layout, "bar": bar}

        def _toggle(self):
            if self.is_running:
                self.is_running = False
                self.timer.stop()
                self.start_btn.setText("Start")
                self.start_btn.setStyleSheet(
                    "QPushButton { background: #4CAF50; color: white; padding: 8px; "
                    "border-radius: 4px; font-weight: bold; font-size: 14px; }")
                self.status_lbl.setText("Stopped")
            else:
                self.is_running = True
                self.timer.start()
                self.start_btn.setText("Stop")
                self.start_btn.setStyleSheet(
                    "QPushButton { background: #f44336; color: white; padding: 8px; "
                    "border-radius: 4px; font-weight: bold; font-size: 14px; }")
                self.status_lbl.setText(f"Running @ {sr} Hz | {len(eeg_channels)} ch")

        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key.Key_Space:
                self._toggle()
            elif event.key() == QtCore.Qt.Key.Key_Q:
                self.close()

        def _update(self):
            n = int(max(RT_MI_WINDOW_S, 5.0) * sr)
            data = board.get_current_board_data(n, BrainFlowPresets.DEFAULT_PRESET)
            if data.shape[1] < int(RT_BLINK_WINDOW_S * sr):
                return

            self._update_eeg(data)

            is_blink, blink_conf = classifier.classify_blink(data, eeg_channels)
            self.blink_bar["bar"].setValue(int(blink_conf * 100))

            if is_blink:
                self._set_state("blink", blink_conf)
                self.blink_flash = 5
                return

            if self.blink_flash > 0:
                self.blink_flash -= 1
                if self.blink_flash > 0:
                    return

            if data.shape[1] >= int(RT_MI_WINDOW_S * sr):
                mi_pred, mi_conf = classifier.classify_mi(data, eeg_channels)
                self.mi_bar["bar"].setValue(int(mi_conf * 100))
                self._set_state(mi_pred, mi_conf)

        def _set_state(self, state, conf):
            color = self.COLORS.get(state, self.COLORS["idle"])
            text_map = {"idle": "IDLE", "left": "LEFT", "right": "RIGHT", "blink": "BLINK"}
            arrow_map = {"idle": "", "left": "<<<", "right": ">>>", "blink": "O"}

            self.state_label.setText(text_map.get(state, "IDLE"))
            self.state_label.setStyleSheet(
                f"font-size: 72px; font-weight: bold; color: {color}; "
                "background: transparent;")
            self.arrow_label.setText(arrow_map.get(state, ""))
            self.arrow_label.setStyleSheet(
                f"font-size: 48px; color: {color}; background: transparent;")

            if state != "idle":
                ts = datetime.now().strftime("%H:%M:%S")
                entry = f"{ts}  {text_map.get(state, 'IDLE'):<6} ({conf:.0%})"
                self.log.insertItem(0, entry)
                item = self.log.item(0)
                if item:
                    item.setForeground(QtGui.QColor(color))
                while self.log.count() > 100:
                    self.log.takeItem(self.log.count() - 1)

            self.current_state = state

        def _update_eeg(self, data):
            n_show = min(data.shape[1], int(5 * sr))
            if n_show < 10:
                return
            x = np.linspace(-n_show / sr, 0, n_show)
            for i, ch in enumerate(eeg_channels[:4]):
                sig = data[ch, -n_show:].copy()
                try:
                    DataFilter.detrend(sig, DetrendOperations.LINEAR.value)
                except Exception:
                    pass
                self.eeg_curves[ch].setData(x, sig + i * 120)

        def closeEvent(self, event):
            self.timer.stop()
            self.is_running = False
            event.accept()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = BCIWindow()
    window.show()
    print("GUI launched. Press Space to start/stop, Q to quit.")
    app.exec()


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BCI Real-Time System")
    parser.add_argument("--simulate", action="store_true",
                        help="Use synthetic board (no Muse needed)")
    parser.add_argument("--headless", action="store_true",
                        help="Terminal output only, no GUI")
    parser.add_argument("--models", type=str, default=None,
                        help="Path to model bundle (.pkl)")
    parser.add_argument("--serial", type=str, default=None,
                        help="Muse serial number for BLE connection")
    args = parser.parse_args()

    # Load models
    models_path = Path(args.models) if args.models else None
    classifier = RealtimeClassifier(models_path)
    if not classifier.load_models():
        print("\nTrain models first:  python train.py")
        sys.exit(1)

    # Connect board
    params = BrainFlowInputParams()
    if args.serial:
        params.serial_number = args.serial

    if args.simulate:
        board_id = BoardIds.SYNTHETIC_BOARD.value
        print("Using SYNTHETIC board (predictions will be random)")
    else:
        board_id = BoardIds.MUSE_2_BOARD.value
        print("Connecting to Muse 2...")

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    sr = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    print(f"Board ready: {len(eeg_channels)} EEG channels @ {sr} Hz")

    try:
        if args.headless:
            run_headless(board, eeg_channels, classifier, sr)
        else:
            run_gui(board, eeg_channels, classifier, sr)
    finally:
        print("Shutting down...")
        board.stop_stream()
        board.release_session()
        print("Done.")


if __name__ == "__main__":
    main()
