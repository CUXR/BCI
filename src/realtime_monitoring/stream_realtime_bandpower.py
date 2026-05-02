import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, DetrendOperations, WindowOperations

params = BrainFlowInputParams()
board_id = BoardIds.MUSE_2_BOARD.value
board = BoardShim(board_id, params)

board.prepare_session()
board.start_stream()

sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)
nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

try:
    while True:
        # grab the last ~1 second of EEG per channel
        window_len = sampling_rate
        data = board.get_current_board_data(window_len, BrainFlowPresets.DEFAULT_PRESET)
        if data.shape[1] < window_len:
            time.sleep(0.05)
            continue

        # compute bandpowers for each EEG channel
        bands = []
        for ch in eeg_channels:
            sig = data[ch, :]
            DataFilter.detrend(sig, DetrendOperations.LINEAR.value)
            psd = DataFilter.get_psd_welch(
                sig, nfft, nfft // 2, sampling_rate, WindowOperations.HANNING.value
            )
            # integrate PSD over canonical bands
            delta = DataFilter.get_band_power(psd, 1.0, 4.0)
            theta = DataFilter.get_band_power(psd, 4.0, 8.0)
            alpha = DataFilter.get_band_power(psd, 8.0, 13.0)
            beta  = DataFilter.get_band_power(psd, 13.0, 30.0)
            gamma = DataFilter.get_band_power(psd, 30.0, 45.0)
            bands.append((delta, theta, alpha, beta, gamma))

        # example: alpha/beta ratio from channel 0
        a_over_b = bands[0][2] / max(bands[0][3], 1e-12)
        print(f"Alpha/Beta ch0: {a_over_b:.2f}")

        # small sleep to control loop timing
        time.sleep(0.1)
finally:
    board.stop_stream()
    board.release_session()