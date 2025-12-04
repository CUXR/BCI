from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations
from eeg_filters import EEGFilter

# Handle Ctrl+C
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

class BandPlotWindow(QtWidgets.QMainWindow):
    def __init__(self, board, eeg_channels, sampling_rate):
        super().__init__()
        self.board = board
        self.eeg_channels = list(eeg_channels)
        # Use first EEG channel for band power calculations by default
        self.eeg_channel_for_bands = self.eeg_channels[0] if len(self.eeg_channels) else None
        self.sr = sampling_rate
        
        # Human-friendly electrode labels for Muse 2 order
        default_labels = ['TP9', 'AF7', 'AF8', 'TP10']
        self.eeg_channel_to_label = {}
        for idx, ch in enumerate(self.eeg_channels):
            label = default_labels[idx] if idx < len(default_labels) else f"Ch{ch}"
            self.eeg_channel_to_label[ch] = label
        
        # Initialize EEG filter
        self.eeg_filter = EEGFilter(sampling_rate, notch_freq=50.0)  # 50 Hz for Europe, change to 60 for US
        
        # ==== UI LAYOUT ====
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vlayout = QtWidgets.QVBoxLayout(central)
        
        # Controls row
        controls = QtWidgets.QHBoxLayout()
        vlayout.addLayout(controls)
        
        # Band toggles
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        self.band_checkboxes = {}
        for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            cb = QtWidgets.QCheckBox(band_name)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_band_toggle)
            controls.addWidget(cb)
            self.band_checkboxes[band_name] = cb
        
        # Raw channel toggles
        self.raw_channel_checkboxes = {}
        for idx, ch in enumerate(self.eeg_channels):
            cb = QtWidgets.QCheckBox(self.eeg_channel_to_label[ch])
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_raw_toggle)
            controls.addWidget(cb)
            self.raw_channel_checkboxes[ch] = cb
        
        # Refresh button
        self.refresh_btn = QtWidgets.QPushButton('Refresh')
        self.refresh_btn.clicked.connect(self._on_refresh)
        controls.addWidget(self.refresh_btn)
        controls.addStretch(1)
        
        # Raw data plot
        self.raw_plot = pg.PlotWidget()
        self.raw_plot.setTitle("Raw EEG")
        self.raw_plot.setLabel('bottom', 'Time', units='s')
        self.raw_plot.setLabel('left', 'Amplitude', units='uV')
        self.raw_plot.addLegend()
        vlayout.addWidget(self.raw_plot)
        # One curve per raw EEG channel
        self.raw_curves = {}
        raw_colors = ['c', 'w', 'y', 'g', 'r', 'm']
        for idx, ch in enumerate(self.eeg_channels):
            color = raw_colors[idx % len(raw_colors)]
            label = self.eeg_channel_to_label[ch]
            self.raw_curves[ch] = self.raw_plot.plot([], [], pen=pg.mkPen(color=color, width=1), name=label)
        self.raw_window_s = 5  # show last 5 seconds of raw data
        
        # Band power plot
        self.band_plot = pg.PlotWidget()
        self.band_plot.setTitle("Live Band Powers")
        self.band_plot.addLegend()
        self.band_plot.setLabel('bottom', 'Time', units='s')
        self.band_plot.setLabel('left', 'Power')
        vlayout.addWidget(self.band_plot)
        
        # State for band power history
        self.time_axis = []
        self.curves = {}
        colors = {'delta': 'r', 'theta': 'g', 'alpha': 'b', 'beta': 'y', 'gamma': 'm'}
        for name in self.bands:
            pen = pg.mkPen(color=colors[name], width=2)
            self.curves[name] = self.band_plot.plot([], [], pen=pen, name=name)
        
        # Timer to update periodically
        self.timer = QtCore.QTimer()
        self.timer.setInterval(200)  # ms, i.e. 5 Hz update
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        
        # Buffer size (in samples) for PSD window
        self.psd_nfft = DataFilter.get_nearest_power_of_two(self.sr * 2)  # e.g. 2 seconds
        self.psd_overlap = self.psd_nfft // 2

    def _on_band_toggle(self, _state):
        # Show/hide curves based on checkboxes
        for name, cb in self.band_checkboxes.items():
            self.curves[name].setVisible(cb.isChecked())

    def _on_raw_toggle(self, _state):
        for ch, cb in self.raw_channel_checkboxes.items():
            self.raw_curves[ch].setVisible(cb.isChecked())

    def _on_refresh(self):
        # Clear accumulated band power history and reset curves
        self.time_axis = []
        for name in self.curves:
            self.curves[name].setData([], [])
        # Also clear raw (next update will repopulate from current window)
        for ch in self.raw_curves:
            self.raw_curves[ch].setData([], [])

    def update_plot(self):
        # Pull current data window
        n_samples_psd = self.sr * 2
        data = self.board.get_current_board_data(
            max(n_samples_psd, self.sr * self.raw_window_s),
            BrainFlowPresets.DEFAULT_PRESET
        )
        
        # ---- Raw EEG plot (last raw_window_s seconds) for all channels ----
        # Build x axis based on longest available channel buffer
        # Use any channel present to compute duration; fall back if empty
        if len(self.eeg_channels) > 0 and data.shape[1] > 0:
            any_ch = self.eeg_channels[0]
            duration_s = data.shape[1] / float(self.sr)
            x_raw = np.linspace(-duration_s, 0.0, data.shape[1])
            for ch in self.eeg_channels:
                sig_ch = data[ch]
                # Detrend for visualization to correct DC offsets
                try:
                    DataFilter.detrend(sig_ch, DetrendOperations.LINEAR.value)
                except Exception:
                    pass
                self.raw_curves[ch].setData(x_raw, sig_ch)
        
        # ---- Filtering for band power ----
        if self.eeg_channel_for_bands is None:
            return
        signal_raw_for_bands = data[self.eeg_channel_for_bands]
        signal = signal_raw_for_bands.copy()
        signal = self.eeg_filter.apply_comprehensive_filter(
            signal,
            bandpass=True,
            notch=True,
            detrend=True
        )
        signal = self.eeg_filter.apply_moving_artifact_removal(signal, threshold=3.0)
        
        # Compute PSD
        if len(signal) >= self.psd_nfft:
            psd = DataFilter.get_psd_welch(
                signal[-self.psd_nfft:],
                self.psd_nfft,
                self.psd_overlap,
                self.sr,
                WindowOperations.HANNING.value
            )
        else:
            psd = None
        
        # Compute band powers
        band_powers = {}
        for name, (f_low, f_high) in self.bands.items():
            if psd is not None:
                bp = DataFilter.get_band_power(psd, f_low, f_high)
            else:
                bp = 0
            band_powers[name] = bp
        
        # Update history for band power plot
        t = QtCore.QTime.currentTime().msecsSinceStartOfDay() / 1000.0
        self.time_axis.append(t)
        for name in self.bands:
            y_existing = self.curves[name].yData if self.curves[name].yData is not None else []
            y_new = list(y_existing) + [band_powers[name]]
            self.curves[name].setData(self.time_axis, y_new)
        
        # Trim history to avoid unbounded growth
        max_len = 200
        if len(self.time_axis) > max_len:
            self.time_axis = self.time_axis[-max_len:]
            for name in self.bands:
                y = self.curves[name].yData
                self.curves[name].setData(self.time_axis, y[-max_len:])

def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board_id = BoardIds.MUSE_2_BOARD.value
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    sr = BoardShim.get_sampling_rate(board_id)
    eeg_chs = BoardShim.get_eeg_channels(board_id)
    
    app = QtWidgets.QApplication([])
    win = BandPlotWindow(board, eeg_chs, sr)
    win.show()
    app.exec()
    
    board.stop_stream()
    board.release_session()

if __name__ == '__main__':
    main()