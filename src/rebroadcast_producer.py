import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

# 1) choose your Muse connection
params = BrainFlowInputParams()
board_id = BoardIds.MUSE_2_BOARD.value
board = BoardShim(board_id, params)


board.prepare_session()

# 2) add streamer
board.add_streamer("streaming_board://225.1.1.1:6677", BrainFlowPresets.DEFAULT_PRESET)

# 3) start streaming
board.start_stream()

# 4) sleep for 10 seconds
time.sleep(10)

# 5) stop streaming
board.stop_stream()

# 6) release session
board.release_session()