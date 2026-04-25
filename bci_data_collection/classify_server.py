import argparse
import json
import socket
import threading
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from classifier import classify


def main():
    parser = argparse.ArgumentParser(
        description="Consume EEG multicast, classify, serve results over TCP"
    )
    parser.add_argument("--eeg-ip", default="225.1.1.1", help="Multicast IP to consume (default: 225.1.1.1)")
    parser.add_argument("--eeg-port", type=int, default=6677, help="Multicast port (default: 6677)")
    parser.add_argument("--tcp-port", type=int, default=5000, help="TCP server port (default: 5000)")
    parser.add_argument("--window-sec", type=float, default=1.0, help="Classification window in seconds (default: 1.0)")
    args = parser.parse_args()

    master_board_id = BoardIds.MUSE_2_BOARD.value
    p = BrainFlowInputParams()
    p.ip_address = args.eeg_ip
    p.ip_port = args.eeg_port
    p.master_board = master_board_id

    stream_board = BoardShim(BoardIds.STREAMING_BOARD.value, p)
    stream_board.prepare_session()
    stream_board.start_stream()

    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    eeg_channels = BoardShim.get_eeg_channels(master_board_id)
    window_samples = int(sampling_rate * args.window_sec)

    clients = []
    clients_lock = threading.Lock()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", args.tcp_port))
    server_sock.listen(5)

    def accept_clients():
        while True:
            conn, addr = server_sock.accept()
            print(f"TCP client connected: {addr}")
            with clients_lock:
                clients.append(conn)

    accept_thread = threading.Thread(target=accept_clients, daemon=True)
    accept_thread.start()

    print(f"Consuming EEG from {args.eeg_ip}:{args.eeg_port}")
    print(f"TCP server on port {args.tcp_port}")
    print(f"Window: {args.window_sec}s ({window_samples} samples @ {sampling_rate} Hz)")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            data = stream_board.get_current_board_data(window_samples)
            if data.shape[1] < window_samples:
                time.sleep(0.05)
                continue

            eeg_data = data[eeg_channels, :]
            label, confidence = classify(eeg_data, eeg_channels)

            msg = json.dumps({
                "label": label,
                "confidence": confidence,
                "ts": time.time(),
            }) + "\n"
            msg_bytes = msg.encode()

            with clients_lock:
                dead = []
                for conn in clients:
                    try:
                        conn.sendall(msg_bytes)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        dead.append(conn)
                for conn in dead:
                    clients.remove(conn)
                    conn.close()
                    print("TCP client disconnected")

            print(f"  {label:>8} ({confidence:.0%})")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stream_board.stop_stream()
        stream_board.release_session()
        with clients_lock:
            for conn in clients:
                conn.close()
        server_sock.close()
        print("All resources released.")


if __name__ == "__main__":
    main()
