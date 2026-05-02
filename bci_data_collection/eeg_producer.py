import argparse
import signal
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets


def main():
    parser = argparse.ArgumentParser(description="Stream MUSE 2 EEG via BrainFlow multicast")
    parser.add_argument("--ip", default="225.1.1.1", help="Multicast IP (default: 225.1.1.1)")
    parser.add_argument("--port", type=int, default=6677, help="UDP port (default: 6677)")
    parser.add_argument("--serial-number", default=None, help="Muse BLE name (e.g. Muse-15C3 — last 4 chars of the printed serial)")
    parser.add_argument("--mac-address", default=None, help="Muse BLE MAC address")
    args = parser.parse_args()

    params = BrainFlowInputParams()
    if args.serial_number:
        params.serial_number = args.serial_number
    if args.mac_address:
        params.mac_address = args.mac_address
    board_id = BoardIds.MUSE_2_BOARD.value
    board = BoardShim(board_id, params)

    board.prepare_session()
    streamer = f"streaming_board://{args.ip}:{args.port}"
    board.add_streamer(streamer, BrainFlowPresets.DEFAULT_PRESET)
    board.start_stream()

    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    print(f"MUSE 2 streaming to {streamer}")
    if not args.ip.startswith("127.") and args.ip != "localhost":
        print(
            "  SECURITY: Raw EEG is sent on this network destination; anyone who can "
            "receive this traffic may capture it. Use isolated/trusted networks only."
        )
    print(f"  Sampling rate: {sampling_rate} Hz | EEG channels: {eeg_channels}")
    print("Press Ctrl+C to stop.")

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)

    try:
        while running:
            time.sleep(0.5)
    finally:
        board.stop_stream()
        board.release_session()
        print("Session released.")


if __name__ == "__main__":
    main()
