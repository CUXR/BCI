"""
Psychopy experiment for streaming MUSE EEG data with markers.
Combines EEG data and markers into a single output file.
"""

import os
import sys
import time
import threading
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from psychopy import visual, core, event
from brainflow.board_shim import (
    BoardIds,
    BoardShim,
    BrainFlowError,
    BrainFlowInputParams,
    BrainFlowPresets,
)


class MusePsychopyRecorder:
    """Records MUSE EEG data with markers in Psychopy experiment."""
    
    def __init__(self, output_dir=None):
        """Initialize the recorder.
        
        Args:
            output_dir: Directory to save output file. Defaults to ~/MuseEEG/psychopy
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.expanduser("~"), "MuseEEG", "psychopy")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(output_dir, f"muse_psychopy_{timestamp}.csv")
        
        self.board = None
        self.is_streaming = False
        self.is_recording = False
        
        # Data collection
        self.data_buffer = []
        self.marker_buffer = []
        self.recording_thread = None
        self.stop_recording_flag = threading.Event()
        self.buffer_lock = threading.Lock()  # Lock for thread-safe buffer access
        
        # Board info (will be set after connection)
        self.board_id = BoardIds.MUSE_2_BOARD.value
        self.sampling_rate = None
        self.eeg_channels = None
        self.marker_channel = None
        self.timestamp_channel = None
        
        # Marker definitions (matching muse_eeg_gui.py)
        self.markers = {
            'blink': (10, 'blink'),
            'left_mi': (1, 'left_motor_imagery'),
            'right_mi': (2, 'right_motor_imagery'),
            'relax': (3, 'relax'),
        }
        
    def connect(self):
        """Connect to MUSE board."""
        if self.board is not None:
            print("Already connected.")
            return True
            
        print("Connecting to Muse2...")
        try:
            BoardShim.enable_board_logger()
            
            params = BrainFlowInputParams()
            self.board = BoardShim(self.board_id, params)
            self.board.prepare_session()
            
            # Get board channel info after connection
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            self.marker_channel = BoardShim.get_marker_channel(self.board_id)
            self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
            
            print("Muse2 connected successfully.")
            return True
            
        except BrainFlowError as e:
            print(f"BrainFlow error: {e}")
            self.board = None
            return False
        except Exception as e:
            print(f"Unexpected error while connecting: {e}")
            self.board = None
            return False
    
    def disconnect(self):
        """Disconnect from MUSE board."""
        if self.board is None:
            return
            
        if self.is_recording:
            self.stop_recording()
            
        try:
            self.board.release_session()
            print("BrainFlow session released.")
        except Exception as e:
            print(f"Error while releasing session: {e}")
            
        self.board = None
        self.is_streaming = False
    
    def start_streaming(self):
        """Start streaming data from MUSE."""
        if self.board is None:
            print("Cannot start: Muse2 is not connected.")
            return False
        if self.is_streaming:
            print("Streaming already in progress.")
            return True
            
        try:
            self.board.start_stream()
            self.is_streaming = True
            print("Streaming started.")
            return True
        except BrainFlowError as e:
            print(f"BrainFlow error while starting stream: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error while starting stream: {e}")
            return False
    
    def stop_streaming(self):
        """Stop streaming data from MUSE."""
        if self.board is None or not self.is_streaming:
            return
            
        try:
            self.board.stop_stream()
            self.is_streaming = False
            print("Streaming stopped.")
        except BrainFlowError as e:
            print(f"BrainFlow error while stopping stream: {e}")
        except Exception as e:
            print(f"Unexpected error while stopping stream: {e}")
    
    def _collect_data(self):
        """Background thread to collect data and markers."""
        start_time = None
        last_sample_count = 0
        
        while not self.stop_recording_flag.is_set():
            if not self.is_streaming:
                time.sleep(0.1)
                continue
                
            try:
                # Get all accumulated board data since last call (no arguments)
                # This gets data from the DEFAULT_PRESET automatically
                data = self.board.get_board_data()
                
                # Initialize start time on first data collection
                if start_time is None:
                    start_time = time.time()
                
                if data.shape[1] > 0:
                    # Get timestamp channel
                    if self.timestamp_channel is not None and self.timestamp_channel < data.shape[0]:
                        timestamps = data[self.timestamp_channel, :]
                    else:
                        timestamps = None
                    
                    # Process each sample
                    for i in range(data.shape[1]):
                        # Use BrainFlow timestamp if available, otherwise calculate from start
                        if timestamps is not None and len(timestamps) > i and timestamps[i] > 0:
                            sample_time = timestamps[i]
                        else:
                            # Fallback: calculate from start time and sample index
                            if self.sampling_rate:
                                sample_time = start_time + (i + last_sample_count) / self.sampling_rate
                            else:
                                sample_time = time.time()
                        
                        # Check for markers in this sample
                        # If marker found, add it as a separate row with empty EEG
                        if self.marker_channel is not None and self.marker_channel < data.shape[0]:
                            marker_val = data[self.marker_channel, i]
                            if marker_val > 0:
                                # Find matching marker
                                for label, (code, _) in self.markers.items():
                                    if abs(marker_val - code) < 0.1:  # Allow small float differences
                                        # Add marker as separate row with empty EEG
                                        with self.buffer_lock:
                                            marker_sample = {
                                                'timestamp': sample_time,
                                                'eeg_data': [],  # Empty EEG for markers
                                                'marker_code': code,
                                                'marker_label': self.markers[label][1],
                                            }
                                            self.data_buffer.append(marker_sample)
                                        break
                        
                        # Extract EEG channels (only for non-marker samples)
                        eeg_data = []
                        if len(self.eeg_channels) > 0:
                            for ch_idx in self.eeg_channels:
                                if ch_idx < data.shape[0]:
                                    eeg_data.append(float(data[ch_idx, i]))
                        
                        # Store EEG sample (always without markers - markers are separate rows)
                        with self.buffer_lock:
                            sample = {
                                'timestamp': sample_time,
                                'eeg_data': eeg_data,
                                'marker_code': None,  # No markers in EEG rows
                                'marker_label': None,
                            }
                            self.data_buffer.append(sample)
                    
                    last_sample_count += data.shape[1]
                    if len(self.data_buffer) % 100 == 0:  # Print every 100 samples
                        print(f"Collected {len(self.data_buffer)} samples...")
                
                # Sleep to control collection rate (collect ~10 times per second)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error collecting data: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def start_recording(self):
        """Start recording data to file."""
        if not self.is_streaming:
            print("Cannot start recording: streaming not started.")
            return False
        if self.is_recording:
            print("Recording already in progress.")
            return True
            
        self.is_recording = True
        self.data_buffer = []
        self.marker_buffer = []
        self.stop_recording_flag.clear()
        
        # Start data collection thread
        self.recording_thread = threading.Thread(target=self._collect_data, daemon=True)
        self.recording_thread.start()
        
        print("Recording started.")
        return True
    
    def stop_recording(self):
        """Stop recording and save data to file."""
        if not self.is_recording:
            return
            
        print("Stopping recording...")
        self.is_recording = False
        self.stop_recording_flag.set()
        
        # Wait for thread to finish (give it time to finish current collection)
        if self.recording_thread is not None:
            self.recording_thread.join(timeout=3.0)
        
        # Get any remaining data from the board buffer
        try:
            if self.board is not None and self.is_streaming:
                remaining_data = self.board.get_board_data()
                if remaining_data.shape[1] > 0:
                    print(f"Collecting {remaining_data.shape[1]} remaining samples...")
                    # Process remaining data
                    if self.timestamp_channel is not None and self.timestamp_channel < remaining_data.shape[0]:
                        timestamps = remaining_data[self.timestamp_channel, :]
                    else:
                        timestamps = None
                    
                    current_time = time.time()
                    for i in range(remaining_data.shape[1]):
                        if timestamps is not None and len(timestamps) > i and timestamps[i] > 0:
                            sample_time = timestamps[i]
                        else:
                            sample_time = current_time
                        
                        # Check for markers in remaining data
                        if self.marker_channel is not None and self.marker_channel < remaining_data.shape[0]:
                            marker_val = remaining_data[self.marker_channel, i]
                            if marker_val > 0:
                                # Add marker as separate row
                                for label, (code, _) in self.markers.items():
                                    if abs(marker_val - code) < 0.1:
                                        marker_sample = {
                                            'timestamp': sample_time,
                                            'eeg_data': [],  # Empty EEG for markers
                                            'marker_code': code,
                                            'marker_label': self.markers[label][1],
                                        }
                                        self.data_buffer.append(marker_sample)
                                        break
                        
                        # Extract EEG channels (only for non-marker samples)
                        eeg_data = []
                        if len(self.eeg_channels) > 0:
                            for ch_idx in self.eeg_channels:
                                if ch_idx < remaining_data.shape[0]:
                                    eeg_data.append(float(remaining_data[ch_idx, i]))
                        
                        # Store EEG sample (without markers)
                        sample = {
                            'timestamp': sample_time,
                            'eeg_data': eeg_data,
                            'marker_code': None,  # No markers in EEG rows
                            'marker_label': None,
                        }
                        self.data_buffer.append(sample)
        except Exception as e:
            print(f"Error collecting remaining data: {e}")
        
        # Save data to file
        self._save_data()
        print(f"Recording stopped. Data saved to {self.output_file}")
    
    def _save_data(self):
        """Save collected data to CSV file."""
        if len(self.data_buffer) == 0:
            print("No data to save.")
            return
        
        print(f"Saving {len(self.data_buffer)} samples to {self.output_file}")
        
        # Determine number of EEG channels from first non-marker sample or channel info
        num_channels = 4  # Default for MUSE
        if self.eeg_channels is not None:
            num_channels = len(self.eeg_channels)
        else:
            # Find first non-marker sample to determine channel count
            for sample in self.data_buffer:
                if len(sample['eeg_data']) > 0:
                    num_channels = len(sample['eeg_data'])
                    break
        
        # Get all EEG channel names
        eeg_channel_names = [f'EEG_{i}' for i in range(num_channels)]
        
        # Write CSV
        try:
            with self.buffer_lock:  # Lock while saving to ensure consistency
                with open(self.output_file, 'w', newline='') as f:
                    fieldnames = ['timestamp', 'marker_code', 'marker_label'] + eeg_channel_names
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for sample in self.data_buffer:
                        row = {
                            'timestamp': sample['timestamp'],
                            'marker_code': sample['marker_code'] if sample['marker_code'] is not None else '',
                            'marker_label': sample['marker_label'] if sample['marker_label'] else '',
                        }
                        # Add EEG channels - empty for marker rows, filled for EEG rows
                        for i, ch_name in enumerate(eeg_channel_names):
                            if i < len(sample['eeg_data']) and len(sample['eeg_data']) > 0:
                                row[ch_name] = sample['eeg_data'][i]
                            else:
                                row[ch_name] = ''  # Empty for marker rows
                        writer.writerow(row)
            
            print(f"Successfully saved {len(self.data_buffer)} samples to {self.output_file}")
        except Exception as e:
            print(f"Error saving data: {e}")
            import traceback
            traceback.print_exc()
    
    def add_marker(self, marker_key):
        """Add a marker to the stream.
        
        Args:
            marker_key: Key from self.markers dict ('blink', 'left_mi', 'right_mi', 'relax')
        """
        if self.board is None or not self.is_streaming:
            print(f"Cannot add marker ({marker_key}): not streaming.")
            return
            
        if marker_key not in self.markers:
            print(f"Unknown marker key: {marker_key}")
            return
        
        if not self.is_recording:
            print(f"Cannot add marker ({marker_key}): not recording.")
            return
        
        code, label = self.markers[marker_key]
        marker_time = time.time()
        
        try:
            # Insert marker into BrainFlow stream
            self.board.insert_marker(float(code))
            
            # Add marker as separate row in buffer (with empty EEG)
            with self.buffer_lock:
                marker_sample = {
                    'timestamp': marker_time,
                    'eeg_data': [],  # Empty EEG for markers
                    'marker_code': code,
                    'marker_label': label,
                }
                self.data_buffer.append(marker_sample)
            
            print(f"Marker sent: {label} (code={code})")
        except Exception as e:
            print(f"Failed to insert marker: {e}")


def run_experiment():
    """Run the Psychopy experiment."""
    # Create recorder
    recorder = MusePsychopyRecorder()
    
    # Create window with modern dark theme
    win = visual.Window(
        size=(1280, 800),
        fullscr=False,
        color=[0.15, 0.15, 0.2],  # Dark blue-gray background (lighter for visibility)
        units='pix',
        screen=0,
        colorSpace='rgb',
        allowGUI=True
    )
    
    # Color scheme (RGB values 0-1)
    bg_color = [0.15, 0.15, 0.2]
    primary_color = [0.2, 0.5, 0.9]  # Blue
    success_color = [0.2, 0.8, 0.4]  # Green
    warning_color = [1.0, 0.7, 0.2]  # Orange
    error_color = [0.9, 0.3, 0.3]    # Red
    text_color = [0.95, 0.95, 0.95]  # Off-white
    accent_color = [0.4, 0.6, 1.0]  # Light blue
    
    # Title
    title = visual.TextStim(
        win,
        text="MUSE EEG Recording",
        pos=(0, 320),
        height=48,
        color=accent_color,
        font='Arial',
        bold=True,
        colorSpace='rgb'
    )
    
    # Status indicator box
    status_box = visual.Rect(
        win,
        width=600,
        height=80,
        pos=(0, 240),
        fillColor=[0.2, 0.2, 0.25],
        lineColor=[0.4, 0.4, 0.5],
        lineWidth=2,
        colorSpace='rgb'
    )
    
    status_text = visual.TextStim(
        win,
        text="● Disconnected",
        pos=(0, 240),
        height=28,
        color=error_color,
        font='Arial',
        bold=True,
        colorSpace='rgb'
    )
    
    # Instructions box
    instructions_box = visual.Rect(
        win,
        width=1000,
        height=280,
        pos=(0, 50),
        fillColor=[0.18, 0.18, 0.23],
        lineColor=[0.35, 0.35, 0.45],
        lineWidth=2,
        colorSpace='rgb'
    )
    
    instructions_title = visual.TextStim(
        win,
        text="Controls",
        pos=(0, 180),
        height=24,
        color=text_color,
        font='Arial',
        bold=True,
        colorSpace='rgb'
    )
    
    instructions = visual.TextStim(
        win,
        text="SPACE  →  Connect to MUSE\n"
             "ENTER  →  Start Recording\n"
             "ESC    →  Stop & Exit",
        pos=(-200, 100),
        height=20,
        color=text_color,
        font='Arial',
        alignText='left',
        colorSpace='rgb'
    )
    
    markers_title = visual.TextStim(
        win,
        text="Markers (while recording)",
        pos=(0, 50),
        height=24,
        color=text_color,
        font='Arial',
        bold=True,
        colorSpace='rgb'
    )
    
    # Marker buttons visual representation
    marker_y_pos = -20
    marker_spacing = 80
    
    marker_boxes = []
    marker_labels = []
    marker_keys = [
        ('B', 'Blink', [0.9, 0.6, 0.2]),
        ('L', 'Left MI', [0.3, 0.7, 0.9]),
        ('R', 'Right MI', [0.5, 0.8, 0.4]),
        ('E', 'Relax', [0.8, 0.5, 0.7]),
    ]
    
    for i, (key, label, color) in enumerate(marker_keys):
        x_pos = -150 + i * marker_spacing
        box = visual.Rect(
            win,
            width=70,
            height=50,
            pos=(x_pos, marker_y_pos),
            fillColor=[c * 0.3 for c in color],
            lineColor=color,
            lineWidth=2,
            colorSpace='rgb'
        )
        key_text = visual.TextStim(
            win,
            text=key,
            pos=(x_pos, marker_y_pos + 8),
            height=24,
            color=color,
            font='Arial',
            bold=True,
            colorSpace='rgb'
        )
        label_text = visual.TextStim(
            win,
            text=label,
            pos=(x_pos, marker_y_pos - 20),
            height=14,
            color=text_color,
            font='Arial',
            colorSpace='rgb'
        )
        marker_boxes.append((box, key_text, label_text))
    
    # Footer info
    footer = visual.TextStim(
        win,
        text="Data saved to ~/MuseEEG/psychopy/",
        pos=(0, -280),
        height=16,
        color=[0.6, 0.6, 0.6],
        font='Arial',
        colorSpace='rgb'
    )
    
    # Recording indicator (pulsing dot)
    recording_dot = visual.Circle(
        win,
        radius=8,
        pos=(-280, 240),
        fillColor=success_color,
        lineColor=None,
        colorSpace='rgb'
    )
    recording_dot.setAutoDraw(False)
    
    # Marker feedback (brief flash)
    marker_feedback = visual.TextStim(
        win,
        text="",
        pos=(0, 160),
        height=32,
        color=accent_color,
        font='Arial',
        bold=True,
        colorSpace='rgb'
    )
    marker_feedback.setAutoDraw(False)
    marker_feedback_time = 0
    
    # Main loop
    connected = False
    recording = False
    clock = core.Clock()
    pulse_phase = 0
    
    # Initial draw to show window immediately
    status_box.draw()
    status_text.draw()
    instructions_box.draw()
    instructions_title.draw()
    instructions.draw()
    markers_title.draw()
    for box, key_text, label_text in marker_boxes:
        box.draw()
        key_text.draw()
        label_text.draw()
    title.draw()
    footer.draw()
    win.flip()
    
    while True:
        # Check for key presses
        keys = event.getKeys()
        current_time = clock.getTime()
        
        if 'escape' in keys:
            break
        elif 'space' in keys and not connected:
            if recorder.connect():
                connected = True
                if recorder.start_streaming():
                    status_text.text = "● Connected • Streaming"
                    status_text.color = success_color
        elif 'return' in keys and connected and not recording:
            if recorder.start_recording():
                recording = True
                status_text.text = "● Connected • Recording"
                status_text.color = success_color
                recording_dot.setAutoDraw(True)
        elif 'b' in keys and recording:
            recorder.add_marker('blink')
            marker_feedback.text = "Blink Marker Added"
            marker_feedback.color = [0.9, 0.6, 0.2]
            marker_feedback.setAutoDraw(True)
            marker_feedback_time = current_time
        elif 'l' in keys and recording:
            recorder.add_marker('left_mi')
            marker_feedback.text = "Left MI Marker Added"
            marker_feedback.color = [0.3, 0.7, 0.9]
            marker_feedback.setAutoDraw(True)
            marker_feedback_time = current_time
        elif 'r' in keys and recording:
            recorder.add_marker('right_mi')
            marker_feedback.text = "Right MI Marker Added"
            marker_feedback.color = [0.5, 0.8, 0.4]
            marker_feedback.setAutoDraw(True)
            marker_feedback_time = current_time
        elif 'e' in keys and recording:
            recorder.add_marker('relax')
            marker_feedback.text = "Relax Marker Added"
            marker_feedback.color = [0.8, 0.5, 0.7]
            marker_feedback.setAutoDraw(True)
            marker_feedback_time = current_time
        
        # Hide marker feedback after 0.5 seconds
        if marker_feedback.getAutoDraw() and (current_time - marker_feedback_time) > 0.5:
            marker_feedback.setAutoDraw(False)
        
        # Pulsing recording indicator
        if recording:
            pulse_phase += 0.1
            alpha = 0.5 + 0.5 * np.sin(pulse_phase)
            recording_dot.fillColor = [c * alpha for c in success_color]
        
        # Draw all elements
        status_box.draw()
        status_text.draw()
        instructions_box.draw()
        instructions_title.draw()
        instructions.draw()
        markers_title.draw()
        
        # Draw marker boxes
        for box, key_text, label_text in marker_boxes:
            if recording:
                # Highlight when recording
                box.lineWidth = 3
            else:
                box.lineWidth = 2
            box.draw()
            key_text.draw()
            label_text.draw()
        
        title.draw()
        footer.draw()
        
        # Draw dynamic elements (auto-drawn)
        win.flip()
    
    # Cleanup
    if recording:
        recorder.stop_recording()
    if connected:
        recorder.stop_streaming()
        recorder.disconnect()
    
    win.close()
    core.quit()


if __name__ == "__main__":
    run_experiment()

