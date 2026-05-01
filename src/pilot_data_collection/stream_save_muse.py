import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

# 1) choose your Muse connection
params = BrainFlowInputParams()

# Option A: Muse 2 via BLE (macOS/Win/Linux). Optionally set mac address like "xx:xx:xx:xx:xx:xx"
board_id = BoardIds.MUSE_2_BOARD.value
# params.mac_address = "XX:XX:XX:XX:XX:XX"  # optional

# Option B: Muse 2 via BLED112 dongle
# board_id = BoardIds.MUSE_2_BLED_BOARD.value
# params.serial_port = "COM3"  # or "/dev/ttyACM0" on Linux/macOS

board = BoardShim(board_id, params)

# 2) prepare and start streaming
board.prepare_session()

# Save EEG preset to CSV in real time (w = write, a = append).
board.add_streamer("file://temp/muse_default.csv:w", BrainFlowPresets.DEFAULT_PRESET)

# Save IMU preset too (gyro/accel), optional.
board.add_streamer("file://temp/muse_aux.csv:w", BrainFlowPresets.AUXILIARY_PRESET)

board.start_stream()  # internal ring buffer starts filling

print("Streaming for 10 seconds...")
time.sleep(10)

# pull everything currently buffered (for all presets)
data = board.get_board_data()  # shape: n_channels x n_samples
print("Buffered shape:", data.shape)

board.stop_stream()
board.release_session()

# If you also want an immediate one-shot write outside the streamers:
# DataFilter.write_file(data, "snapshot.csv", "w")  # see BrainFlow examples page