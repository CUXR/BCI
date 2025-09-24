from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations

# Handle Ctrl+C
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

class BandPlotWindow(QtWidgets.QMainWindow):
    def __init__(self, board, eeg_channel, sampling_rate):
        super().__init__()
        self.board = board
        self.eeg_channel = eeg_channel
        self.sr = sampling_rate
        
        # Set up GUI plot
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.graphWidget.setTitle("Live Band Powers")
        self.graphWidget.addLegend()
        self.graphWidget.setLabel('bottom', 'Time', units='s')
        self.graphWidget.setLabel('left', 'Power')
        
        self.time_axis = []  # time stamps
        # Store curves for each band
        self.curves = {}
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        colors = {'delta':'r', 'theta':'g', 'alpha':'b', 'beta':'y', 'gamma':'m'}
        for name in self.bands:
            pen = pg.mkPen(color=colors[name], width=2)
            self.curves[name] = self.graphWidget.plot([], [], pen=pen, name=name)
        
        # Timer to update periodically
        self.timer = QtCore.QTimer()
        self.timer.setInterval(200)  # ms, i.e. 5 Hz update
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        
        # Buffer size (in samples) for PSD window
        self.psd_nfft = DataFilter.get_nearest_power_of_two(self.sr * 2)  # e.g. 2 seconds
        self.psd_overlap = self.psd_nfft // 2

    def update_plot(self):
        # get latest data (say last 2 seconds)
        n_samples = self.sr * 2
        data = self.board.get_current_board_data(n_samples)
        # data is shape [channel][samples]
        signal = data[self.eeg_channel]
        # maybe detrend
        DataFilter.detrend(signal, DetrendOperations.CONSTANT.value)
        
        # compute PSD
        # if signal is too short, return None
        if len(signal) >= self.psd_nfft:
            psd = DataFilter.get_psd_welch(signal, self.psd_nfft,
                                        self.psd_overlap, self.sr,
                                        WindowOperations.HANNING.value)
        else:
            psd = None
        # compute band powers
        band_powers = {}
        for name, (f_low, f_high) in self.bands.items():
            if psd is not None:
                bp = DataFilter.get_band_power(psd, f_low, f_high)
            else:
                bp = 0
            band_powers[name] = bp

        # update time
        t = QtCore.QTime.currentTime().msecsSinceStartOfDay() / 1000.0
        self.time_axis.append(t)
        for name in self.bands:
            y = self.curves[name].yData if self.curves[name].yData is not None else []
            y = list(y) + [band_powers[name]]
            self.curves[name].setData(self.time_axis, y)
        
        # Optionally trim old data so plot doesn’t endlessly grow
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
    # pick first EEG channel
    eeg_chan = eeg_chs[0]
    
    app = QtWidgets.QApplication([])
    win = BandPlotWindow(board, eeg_chan, sr)
    win.show()
    app.exec_()
    
    board.stop_stream()
    board.release_session()

if __name__ == '__main__':
    main()