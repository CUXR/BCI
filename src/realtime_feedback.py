"""Real-Time BCI Feedback System.

Connects to Muse 2 via BrainFlow, processes EEG in real-time, and displays
brain state predictions (blink detection + motor imagery L/R) in a PyQt6 GUI.

Prerequisites:
    python src/train_and_save_models.py   # Train and save models first

Usage:
    python src/realtime_feedback.py              # Live Muse 2
    python src/realtime_feedback.py --simulate   # Synthetic data for testing
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

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from brainflow.board_shim import (
    BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets)
from brainflow.data_filter import DataFilter, DetrendOperations

# Handle Ctrl+C gracefully
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_filters import EEGFilter
from training.bandpower import train_motor_imagery as mi_mod
import train_blink_detector as blink_mod

MODELS_FILE = PROJECT_ROOT / "models" / "realtime_models.pkl"

CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
SAMPLING_RATE = 256

# Timing
BLINK_WINDOW_S = 0.5
MI_WINDOW_S = 3.0
UPDATE_INTERVAL_MS = 200
BLINK_COOLDOWN_S = 1.0
MI_COOLDOWN_S = 0.5

# Default confidence thresholds
DEFAULT_BLINK_THRESHOLD = 0.65
DEFAULT_MI_THRESHOLD = 0.55


# ============================================================
# Real-time classifier
# ============================================================

class RealtimeClassifier:
    """Manages model loading, preprocessing, and real-time inference."""

    def __init__(self, models_file=MODELS_FILE):
        self.models_file = models_file
        self.bundle = None
        self.eeg_filter = EEGFilter(sampling_rate=SAMPLING_RATE, notch_freq=60.0)
        self.last_blink_time = 0
        self.last_mi_time = 0
        self.blink_threshold = DEFAULT_BLINK_THRESHOLD
        self.mi_threshold = DEFAULT_MI_THRESHOLD

    def load_models(self):
        """Load trained model bundle from disk."""
        if not self.models_file.exists():
            print(f"ERROR: Model file not found: {self.models_file}")
            print("Run 'python src/train_and_save_models.py' first!")
            return False

        with open(self.models_file, "rb") as f:
            self.bundle = pickle.load(f)

        print(f"Models loaded (trained {self.bundle['training_date'][:10]} "
              f"on {self.bundle['n_subjects']} subjects: "
              f"{self.bundle['subjects']})")
        return True

    def _preprocess(self, raw_window):
        """Preprocess a raw EEG window: (n_channels, n_samples) → (n_samples, n_channels).

        Applies NaN sanitization, artifact removal, and comprehensive filtering
        per channel, matching the offline training pipeline.
        """
        n_ch, n_samples = raw_window.shape
        processed = np.zeros((n_samples, n_ch), dtype=np.float64)

        for i in range(n_ch):
            sig = raw_window[i].astype(np.float64)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            sig = self.eeg_filter.apply_moving_artifact_removal(sig, threshold=3.0)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            sig = self.eeg_filter.apply_comprehensive_filter(sig)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            processed[:, i] = sig

        return processed

    def _extract_eeg_window(self, board_data, eeg_channels, ch_indices, n_samples):
        """Extract and rearrange EEG channels from BrainFlow board data.

        Returns (len(ch_indices), n_samples) array.
        """
        window = np.zeros((len(ch_indices), n_samples))
        for i, ch_idx in enumerate(ch_indices):
            if ch_idx < len(eeg_channels):
                window[i] = board_data[eeg_channels[ch_idx], -n_samples:]
        return window

    def classify_blink(self, board_data, eeg_channels):
        """Check if the current window contains an intentional blink.

        Returns (is_blink: bool, confidence: float).
        """
        now = time.time()
        if now - self.last_blink_time < BLINK_COOLDOWN_S:
            return False, 0.0

        if self.bundle is None or self.bundle["blink_pipeline"] is None:
            return False, 0.0

        ch_indices = self.bundle["blink_ch_indices"]
        n_samples = int(BLINK_WINDOW_S * SAMPLING_RATE)

        if board_data.shape[1] < n_samples:
            return False, 0.0

        eeg_window = self._extract_eeg_window(
            board_data, eeg_channels, ch_indices, n_samples)
        epoch = self._preprocess(eeg_window)

        features = blink_mod.extract_blink_features(
            epoch, SAMPLING_RATE, ch_indices)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        X = features.reshape(1, -1)

        try:
            proba = self.bundle["blink_pipeline"].predict_proba(X)[0]
            blink_prob = float(proba[1])
            is_blink = blink_prob > self.blink_threshold

            if is_blink:
                self.last_blink_time = now

            return is_blink, blink_prob
        except Exception:
            return False, 0.0

    def classify_mi(self, board_data, eeg_channels):
        """Classify motor imagery (left vs right vs idle).

        Returns (prediction: str, confidence: float).
        prediction is "left", "right", or "idle".
        """
        now = time.time()
        if now - self.last_mi_time < MI_COOLDOWN_S:
            return "idle", 0.5

        if self.bundle is None or self.bundle["mi_pipeline"] is None:
            return "idle", 0.5

        ch_indices = self.bundle["mi_ch_indices"]
        n_samples = int(MI_WINDOW_S * SAMPLING_RATE)

        if board_data.shape[1] < n_samples:
            return "idle", 0.5

        eeg_window = self._extract_eeg_window(
            board_data, eeg_channels, ch_indices, n_samples)
        epoch = self._preprocess(eeg_window)

        # Spectral + time-domain features
        features = mi_mod.extract_features_single_epoch(
            epoch, SAMPLING_RATE, ch_indices)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # CSP features
        W = self.bundle.get("mi_csp_W")
        if W is not None:
            projected = epoch @ W
            csp_feat = np.log(np.var(projected, axis=0) + 1e-12)
            features = np.concatenate([features, csp_feat])

        X = features.reshape(1, -1)

        try:
            proba = self.bundle["mi_pipeline"].predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])

            self.last_mi_time = now

            if confidence < self.mi_threshold:
                return "idle", confidence

            label = "left" if pred_idx == 0 else "right"
            return label, confidence
        except Exception:
            return "idle", 0.5


# ============================================================
# PyQt6 GUI
# ============================================================

class FeedbackWindow(QtWidgets.QMainWindow):
    """Main GUI window for real-time BCI feedback."""

    COLORS = {
        "idle":  "#666666",
        "left":  "#2196F3",
        "right": "#4CAF50",
        "blink": "#FF9800",
    }

    def __init__(self, board, eeg_channels, classifier, sampling_rate):
        super().__init__()
        self.board = board
        self.eeg_channels = list(eeg_channels)
        self.classifier = classifier
        self.sr = sampling_rate

        self.current_state = "idle"
        self.current_confidence = 0.0
        self.blink_flash_counter = 0
        self.prediction_log = deque(maxlen=100)
        self.is_running = False

        self.setWindowTitle("BCI Real-Time Feedback")
        self.setMinimumSize(920, 780)
        self.setStyleSheet("background-color: #1a1a2e; color: #e0e0e0;")

        self._build_ui()
        self._setup_timer()

    # ---- UI Construction ----

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # Top controls
        layout.addLayout(self._build_controls())

        # Raw EEG plot
        layout.addWidget(self._build_eeg_plot())

        # Prediction display (central)
        layout.addWidget(self._build_prediction_panel(), stretch=1)

        # Band powers
        layout.addWidget(self._build_band_panel())

        # Prediction log
        layout.addWidget(self._build_log())

    def _build_controls(self):
        row = QtWidgets.QHBoxLayout()

        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.setFixedWidth(100)
        self.start_btn.setStyleSheet(
            "QPushButton { background: #4CAF50; color: white; padding: 8px; "
            "border-radius: 4px; font-weight: bold; font-size: 14px; }"
            "QPushButton:hover { background: #45a049; }")
        self.start_btn.clicked.connect(self._toggle_running)
        row.addWidget(self.start_btn)

        self.status_label = QtWidgets.QLabel("Ready — press Start")
        self.status_label.setStyleSheet(
            "font-size: 13px; color: #aaa; padding: 0 12px;")
        row.addWidget(self.status_label)
        row.addStretch()

        # Threshold spinboxes
        for label_text, default, attr_name in [
            ("Blink:", DEFAULT_BLINK_THRESHOLD, "blink_thresh_spin"),
            ("MI:", DEFAULT_MI_THRESHOLD, "mi_thresh_spin"),
        ]:
            lbl = QtWidgets.QLabel(label_text)
            lbl.setStyleSheet("font-size: 12px;")
            row.addWidget(lbl)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.30, 0.95)
            spin.setValue(default)
            spin.setSingleStep(0.05)
            spin.setFixedWidth(70)
            spin.setStyleSheet(
                "background: #2a2a4a; padding: 3px; border: 1px solid #444; "
                "border-radius: 3px;")
            row.addWidget(spin)
            setattr(self, attr_name, spin)

        return row

    def _build_eeg_plot(self):
        self.raw_plot = pg.PlotWidget()
        self.raw_plot.setTitle("Raw EEG", color="w", size="11pt")
        self.raw_plot.setLabel("bottom", "Time", units="s")
        self.raw_plot.setLabel("left", "Channels")
        self.raw_plot.setBackground("#0d0d1a")
        self.raw_plot.addLegend(offset=(10, 10))
        self.raw_plot.setMaximumHeight(190)
        self.raw_plot.getAxis("left").setTicks([])

        colors = ["#00BCD4", "#FFFFFF", "#FFEB3B", "#8BC34A"]
        self.raw_curves = {}
        for i, ch in enumerate(self.eeg_channels[:4]):
            label = CH_NAMES[i] if i < len(CH_NAMES) else f"Ch{ch}"
            pen = pg.mkPen(color=colors[i % len(colors)], width=1)
            self.raw_curves[ch] = self.raw_plot.plot(
                [], [], pen=pen, name=label)

        return self.raw_plot

    def _build_prediction_panel(self):
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            "QFrame { background: #0d0d1a; border: 2px solid #333; "
            "border-radius: 10px; }")
        v = QtWidgets.QVBoxLayout(frame)
        v.setSpacing(2)
        v.setContentsMargins(20, 12, 20, 12)

        # State label
        self.state_label = QtWidgets.QLabel("IDLE")
        self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._style_state_label("idle")
        v.addWidget(self.state_label)

        # Arrow / symbol
        self.arrow_label = QtWidgets.QLabel("")
        self.arrow_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.arrow_label.setStyleSheet(
            "font-size: 44px; color: #444; background: transparent;")
        v.addWidget(self.arrow_label)

        # Confidence bars
        bars_row = QtWidgets.QHBoxLayout()
        bars_row.setSpacing(30)

        self.blink_bar = self._make_conf_bar("Blink Score", "#FF9800")
        self.mi_bar = self._make_conf_bar("MI Confidence", "#2196F3")
        bars_row.addLayout(self.blink_bar["layout"])
        bars_row.addLayout(self.mi_bar["layout"])
        v.addLayout(bars_row)

        return frame

    def _make_conf_bar(self, title, color):
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

    def _build_band_panel(self):
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            "QFrame { background: #0d0d1a; border: 1px solid #333; "
            "border-radius: 6px; }")
        frame.setMaximumHeight(110)
        h = QtWidgets.QHBoxLayout(frame)
        h.setContentsMargins(10, 6, 10, 6)

        self.band_bars = {}
        band_info = [
            ("Delta", "#F44336"), ("Theta", "#FF9800"),
            ("Mu", "#FFEB3B"), ("Alpha", "#2196F3"),
            ("Beta", "#4CAF50"), ("Gamma", "#9C27B0"),
        ]
        for name, color in band_info:
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(2)

            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setOrientation(QtCore.Qt.Orientation.Vertical)
            bar.setFixedWidth(28)
            bar.setStyleSheet(
                f"QProgressBar {{ background: #1a1a2e; border: 1px solid #333; }}"
                f"QProgressBar::chunk {{ background: {color}; }}")
            col.addWidget(bar, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

            lbl = QtWidgets.QLabel(name)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                "font-size: 10px; background: transparent; color: #bbb;")
            col.addWidget(lbl)

            h.addLayout(col)
            self.band_bars[name.lower()] = bar

        return frame

    def _build_log(self):
        self.log_list = QtWidgets.QListWidget()
        self.log_list.setMaximumHeight(115)
        self.log_list.setStyleSheet(
            "QListWidget { background: #0d0d1a; border: 1px solid #333; "
            "border-radius: 6px; font-family: monospace; font-size: 12px; }"
            "QListWidget::item { padding: 1px 4px; }")
        return self.log_list

    # ---- Timer & Controls ----

    def _setup_timer(self):
        self.update_timer = QtCore.QTimer()
        self.update_timer.setInterval(UPDATE_INTERVAL_MS)
        self.update_timer.timeout.connect(self._update)

    def _toggle_running(self):
        if self.is_running:
            self.is_running = False
            self.update_timer.stop()
            self.start_btn.setText("Start")
            self.start_btn.setStyleSheet(
                "QPushButton { background: #4CAF50; color: white; padding: 8px; "
                "border-radius: 4px; font-weight: bold; font-size: 14px; }"
                "QPushButton:hover { background: #45a049; }")
            self.status_label.setText("Stopped")
        else:
            self.is_running = True
            self.update_timer.start()
            self.start_btn.setText("Stop")
            self.start_btn.setStyleSheet(
                "QPushButton { background: #f44336; color: white; padding: 8px; "
                "border-radius: 4px; font-weight: bold; font-size: 14px; }"
                "QPushButton:hover { background: #d32f2f; }")
            self.status_label.setText(
                f"Running @ {self.sr} Hz | {len(self.eeg_channels)} channels")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self._toggle_running()
        elif event.key() == QtCore.Qt.Key.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)

    # ---- Main Update Loop ----

    def _update(self):
        """Called at UPDATE_INTERVAL_MS. Runs blink + MI classification."""
        n_needed = int(max(MI_WINDOW_S, 5.0) * self.sr)
        data = self.board.get_current_board_data(
            n_needed, BrainFlowPresets.DEFAULT_PRESET)

        if data.shape[1] < int(BLINK_WINDOW_S * self.sr):
            return

        # Sync thresholds from spinboxes
        self.classifier.blink_threshold = self.blink_thresh_spin.value()
        self.classifier.mi_threshold = self.mi_thresh_spin.value()

        # Update visualizations
        self._update_raw_plot(data)
        self._update_band_powers(data)

        # --- Blink detection (priority) ---
        is_blink, blink_conf = self.classifier.classify_blink(
            data, self.eeg_channels)
        self.blink_bar["bar"].setValue(int(blink_conf * 100))

        if is_blink:
            self._set_state("blink", blink_conf)
            self.blink_flash_counter = 5  # ~1s flash (5 × 200ms)
            return

        # Blink flash cooldown
        if self.blink_flash_counter > 0:
            self.blink_flash_counter -= 1
            if self.blink_flash_counter > 0:
                return

        # --- Motor imagery ---
        if data.shape[1] >= int(MI_WINDOW_S * self.sr):
            mi_pred, mi_conf = self.classifier.classify_mi(
                data, self.eeg_channels)
            self.mi_bar["bar"].setValue(int(mi_conf * 100))
            self._set_state(mi_pred, mi_conf)
        else:
            self._set_state("idle", 0.5)

    # ---- State Display ----

    def _style_state_label(self, state):
        color = self.COLORS.get(state, self.COLORS["idle"])
        self.state_label.setStyleSheet(
            f"font-size: 60px; font-weight: bold; color: {color}; "
            "padding: 8px; background: transparent;")

    def _set_state(self, state, confidence):
        prev = self.current_state
        # Skip redundant updates
        if state == prev and abs(confidence - self.current_confidence) < 0.02:
            return

        self.current_state = state
        self.current_confidence = confidence
        color = self.COLORS.get(state, self.COLORS["idle"])

        text_map = {
            "idle": "IDLE", "left": "LEFT",
            "right": "RIGHT", "blink": "BLINK",
        }
        display = text_map.get(state, "IDLE")
        self.state_label.setText(display)
        self._style_state_label(state)

        arrow_map = {"idle": "", "left": "<<<", "right": ">>>", "blink": "O"}
        self.arrow_label.setText(arrow_map.get(state, ""))
        self.arrow_label.setStyleSheet(
            f"font-size: 44px; color: {color}; background: transparent;")

        # Log non-trivial transitions
        if state != "idle" or prev != "idle":
            ts = datetime.now().strftime("%H:%M:%S")
            conf_str = f"({confidence:.0%})" if confidence > 0 else ""
            entry = f"{ts}  {display:<6} {conf_str}"
            self.log_list.insertItem(0, entry)
            item = self.log_list.item(0)
            if item:
                item.setForeground(QtGui.QColor(color))
            while self.log_list.count() > 100:
                self.log_list.takeItem(self.log_list.count() - 1)

    # ---- Raw EEG Plot ----

    def _update_raw_plot(self, data):
        n_display = min(data.shape[1], int(5 * self.sr))
        if n_display < 10:
            return

        x = np.linspace(-n_display / self.sr, 0, n_display)
        for i, ch in enumerate(self.eeg_channels[:4]):
            sig = data[ch, -n_display:].copy()
            try:
                DataFilter.detrend(sig, DetrendOperations.LINEAR.value)
            except Exception:
                pass
            # Vertical offset per channel for readability
            self.raw_curves[ch].setData(x, sig + i * 120)

    # ---- Band Power Display ----

    def _update_band_powers(self, data):
        ch = self.eeg_channels[0]
        n = min(data.shape[1], int(2 * self.sr))
        if n < 64:
            return

        sig = data[ch, -n:].copy().astype(np.float64)
        sig = np.nan_to_num(sig, nan=0.0)
        try:
            DataFilter.detrend(sig, DetrendOperations.LINEAR.value)
        except Exception:
            pass

        from scipy.signal import welch as sp_welch
        nperseg = min(int(self.sr), len(sig))
        freqs, psd = sp_welch(sig, fs=self.sr, nperseg=nperseg)

        bands = {
            "delta": (1, 4), "theta": (4, 8), "mu": (8, 12),
            "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45),
        }
        total_mask = (freqs >= 1) & (freqs <= 45)
        total = np.trapz(psd[total_mask], freqs[total_mask])

        for name, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            power = np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0
            relative = power / (total + 1e-12)
            bar_val = min(100, int(relative * 250))
            if name in self.band_bars:
                self.band_bars[name].setValue(bar_val)

    # ---- Cleanup ----

    def closeEvent(self, event):
        self.update_timer.stop()
        self.is_running = False
        event.accept()


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="BCI Real-Time Feedback System")
    parser.add_argument(
        "--simulate", action="store_true",
        help="Use synthetic board (no Muse headband needed)")
    parser.add_argument(
        "--models", type=str, default=str(MODELS_FILE),
        help="Path to trained model bundle (.pkl)")
    args = parser.parse_args()

    # Load models
    classifier = RealtimeClassifier(models_file=Path(args.models))
    if not classifier.load_models():
        print("\nTo train models first, run:")
        print("  python src/train_and_save_models.py")
        sys.exit(1)

    # Connect board
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

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

    # Launch GUI
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    window = FeedbackWindow(board, eeg_channels, classifier, sr)
    window.show()

    print("GUI launched. Press Space to start/stop, Q to quit.")

    try:
        app.exec()
    finally:
        print("Shutting down...")
        board.stop_stream()
        board.release_session()
        print("Done.")


if __name__ == "__main__":
    main()
