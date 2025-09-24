import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# configure parameters to match the rebroadcaster
p = BrainFlowInputParams()
p.ip_address = "225.1.1.1"   # must match rebroadcast producer
p.ip_port = 6677             # must match rebroadcast producer
p.master_board = BoardIds.MUSE_2_BOARD.value  # original board type

# create a STREAMING_BOARD consumer
stream_board = BoardShim(BoardIds.STREAMING_BOARD.value, p)
stream_board.prepare_session()
stream_board.start_stream()

try:
    while True:
        # grab newest data since last call
        data = stream_board.get_board_data()  
        if data.shape[1] > 0:
            print(f"Received {data.shape[1]} samples")
            # for example, print first channel
            print(data[0, -10:])  # last 10 values of first channel
        else:
            print("No new data yet")
        time.sleep(0.1)  # small pause to avoid busy loop
except KeyboardInterrupt:
    print("Stopping...")
finally:
    stream_board.stop_stream()
    stream_board.release_session()