import os
import sys
import time

from PyQt6 import QtWidgets, QtGui, QtCore
from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
    BrainFlowPresets,
)
from brainflow.board_shim import BrainFlowError  # for nicer error handling


class MuseEEGGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # ------------ paths & basic state ------------
        # Directory where we save CSV files
        self.output_dir = os.path.join(os.path.expanduser("~"), "MuseEEG")
        os.makedirs(self.output_dir, exist_ok=True)

        self.eeg_file = os.path.join(self.output_dir, "muse_default.csv")
        self.aux_file = os.path.join(self.output_dir, "muse_aux.csv")
        self.marker_file = os.path.join(self.output_dir, "markers.csv")

        # BrainFlow objects
        self.board = None
        self.is_streaming = False

        # Create marker file with header if not exists
        if not os.path.exists(self.marker_file):
            with open(self.marker_file, "w") as f:
                f.write("timestamp,code,label\n")

        self._build_ui()

    # ------------ UI SETUP ------------
    def _build_ui(self):
        self.setWindowTitle("Muse2 Motor Imagery Recorder")
        self.setMinimumSize(720, 420)

        # Slight mac-style feel
        self.setWindowIcon(QtGui.QIcon())  # you can set your own .icns here
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            QLabel#statusLabel {
                color: black;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton {
                color: black;
                padding: 8px 18px;
                border-radius: 8px;
                border: 1px solid #d0d0d4;
                background-color: #ffffff;
                font-size: 13px;
            }
            QPushButton:hover {
                color: black;
                background-color: #e9ecf5;
            }
            QPushButton:pressed {
                color: black;
                background-color: #d0d3dd;
            }
            QPushButton#primaryButton {
                background-color: #007aff;
                color: white;
                border: none;
            }
            QPushButton#primaryButton:hover {
                background-color: #2f8dff;
            }
            QPushButton#primaryButton:pressed {
                background-color: #0060d0;
            }
            QTextEdit {
                color: black;
                background-color: #ffffff;
                border-radius: 8px;
                padding: 8px;
                border: 1px solid #d0d0d4;
                font-family: Menlo, Monaco, monospace;
                font-size: 12px;
            }
        """)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Status row
        self.status_label = QtWidgets.QLabel("Status: Disconnected")
        self.status_label.setObjectName("statusLabel")
        self._set_status_color(connected=False)

        small_hint = QtWidgets.QLabel(
            "Muse2 + BrainFlow • Data saved in ~/MuseEEG (muse_default.csv, muse_aux.csv, markers.csv)"
        )
        small_font = small_hint.font()
        small_font.setPointSize(10)
        small_hint.setFont(small_font)
        small_hint.setStyleSheet("color: #6e6e73;")

        status_layout = QtWidgets.QVBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(small_hint)

        main_layout.addLayout(status_layout)

        # Connection controls
        connect_btn = QtWidgets.QPushButton("Connect")
        connect_btn.clicked.connect(self.connect_board)

        disconnect_btn = QtWidgets.QPushButton("Disconnect")
        disconnect_btn.clicked.connect(self.disconnect_board)

        conn_layout = QtWidgets.QHBoxLayout()
        conn_layout.addWidget(connect_btn)
        conn_layout.addWidget(disconnect_btn)
        conn_layout.addStretch(1)

        main_layout.addLayout(conn_layout)

        # Start/Stop controls
        self.start_btn = QtWidgets.QPushButton("Start Recording")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self.start_streaming)

        self.stop_btn = QtWidgets.QPushButton("Stop Recording")
        self.stop_btn.clicked.connect(self.stop_streaming)

        rec_layout = QtWidgets.QHBoxLayout()
        rec_layout.addWidget(self.start_btn)
        rec_layout.addWidget(self.stop_btn)
        rec_layout.addStretch(1)

        main_layout.addLayout(rec_layout)

        # Marker buttons row
        blink_btn = QtWidgets.QPushButton("Blink")
        blink_btn.clicked.connect(lambda: self.add_marker(10, "blink"))

        m1_btn = QtWidgets.QPushButton("Marker 1 – Left MI")
        m1_btn.clicked.connect(lambda: self.add_marker(1, "left_motor_imagery"))

        m2_btn = QtWidgets.QPushButton("Marker 2 – Right MI")
        m2_btn.clicked.connect(lambda: self.add_marker(2, "right_motor_imagery"))

        m3_btn = QtWidgets.QPushButton("Marker 3 – Relax")
        m3_btn.clicked.connect(lambda: self.add_marker(3, "relax"))

        marker_layout = QtWidgets.QHBoxLayout()
        marker_layout.addWidget(blink_btn)
        marker_layout.addWidget(m1_btn)
        marker_layout.addWidget(m2_btn)
        marker_layout.addWidget(m3_btn)
        marker_layout.addStretch(1)

        main_layout.addLayout(marker_layout)

        # Log area
        self.log_edit = QtWidgets.QTextEdit()
        self.log_edit.setReadOnly(True)
        main_layout.addWidget(self.log_edit, stretch=1)

        self._log("Ready. Click ‘Connect’ to connect to Muse2.")

    # ------------ HELPERS ------------
    def _set_status_color(self, connected: bool):
        if connected:
            self.status_label.setStyleSheet(
                "color: #0a7d33; font-weight: 600;"
            )
        else:
            self.status_label.setStyleSheet(
                "color: #d0021b; font-weight: 600;"
            )

    def _log(self, msg: str):
        t = time.strftime("%H:%M:%S")
        self.log_edit.append(f"[{t}] {msg}")

    # ------------ BRAINFLOW: CONNECT/DISCONNECT ------------
    def connect_board(self):
        if self.board is not None:
            self._log("Already connected.")
            return

        self._log("Connecting to Muse2…")
        try:
            BoardShim.enable_board_logger()

            params = BrainFlowInputParams()

            # Default: Muse 2 via BLE.
            # If you want to force a specific MAC, uncomment:
            # params.mac_address = "XX:XX:XX:XX:XX:XX"

            board_id = BoardIds.MUSE_2_BOARD.value

            # For BLED112 dongle, instead:
            # board_id = BoardIds.MUSE_2_BLED_BOARD.value
            # params.serial_port = "COM3" or "/dev/ttyACM0"

            self.board = BoardShim(board_id, params)
            self.board.prepare_session()

            self._set_status_color(True)
            self.status_label.setText("Status: Connected")
            self._log("Muse2 session prepared successfully.")

        except BrainFlowError as e:
            self._log(f"BrainFlow error: {e}")
            self.board = None
            self._set_status_color(False)
            self.status_label.setText("Status: Disconnected")
        except Exception as e:
            self._log(f"Unexpected error while connecting: {e}")
            self.board = None
            self._set_status_color(False)
            self.status_label.setText("Status: Disconnected")

    def disconnect_board(self):
        if self.board is None:
            self._log("Not connected.")
            return

        # If still streaming, stop first
        if self.is_streaming:
            self.stop_streaming()

        try:
            self.board.release_session()
            self._log("BrainFlow session released.")
        except Exception as e:
            self._log(f"Error while releasing session: {e}")

        self.board = None
        self.is_streaming = False
        self._set_status_color(False)
        self.status_label.setText("Status: Disconnected")

    # ------------ START / STOP STREAMING ------------
    def start_streaming(self):
        if self.board is None:
            self._log("Cannot start: Muse2 is not connected.")
            return
        if self.is_streaming:
            self._log("Streaming already in progress.")
            return

        # Overwrite previous files each time; change to ':a' to append instead.
        eeg_streamer = f"file://{self.eeg_file}:w"
        aux_streamer = f"file://{self.aux_file}:w"

        try:
            self.board.add_streamer(eeg_streamer, BrainFlowPresets.DEFAULT_PRESET)
            self.board.add_streamer(aux_streamer, BrainFlowPresets.AUXILIARY_PRESET)

            # Start BrainFlow ring buffer; streaming continues until we call stop_stream()
            self.board.start_stream()
            self.is_streaming = True
            self._log(f"Streaming started. EEG -> {self.eeg_file}, AUX -> {self.aux_file}")
            self.status_label.setText("Status: Connected • Recording")

        except BrainFlowError as e:
            self._log(f"BrainFlow error while starting stream: {e}")
        except Exception as e:
            self._log(f"Unexpected error while starting stream: {e}")

    def stop_streaming(self):
        if self.board is None:
            self._log("Cannot stop: Muse2 is not connected.")
            return
        if not self.is_streaming:
            self._log("Streaming is not running.")
            return

        try:
            self.board.stop_stream()
            self.is_streaming = False
            self._log("Streaming stopped. Data written to CSV.")
            self.status_label.setText("Status: Connected • Idle")
        except BrainFlowError as e:
            self._log(f"BrainFlow error while stopping stream: {e}")
        except Exception as e:
            self._log(f"Unexpected error while stopping stream: {e}")

    # ------------ MARKERS ------------
    def add_marker(self, code: int, label: str):
        if self.board is None or not self.is_streaming:
            self._log(f"Cannot add marker ({label}): not recording.")
            return

        ts = time.time()

        # Try to send a BrainFlow marker if supported
        try:
            # Some boards support a marker channel; if Muse2 doesn’t, this will just fail silently.
            self.board.insert_marker(float(code))
        except Exception:
            pass

        # Always log to markers.csv so you can align with EEG later
        try:
            with open(self.marker_file, "a") as f:
                f.write(f"{ts},{code},{label}\n")
        except Exception as e:
            self._log(f"Failed to write marker file: {e}")

        pretty_name = label.replace("_", " ").title()
        self._log(f"Marker sent: {pretty_name} (code={code})")


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Let Qt pick the native macOS style automatically
    # (no need to set a style explicitly on Mac)
    window = MuseEEGGui()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
