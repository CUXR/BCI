# MUSE EEG Psychopy Recording

This module provides a Psychopy-based experiment interface for streaming and recording MUSE EEG data with markers.

## Features

- Streams EEG data from MUSE2 headband using BrainFlow
- Records data and markers together in a single CSV file
- Supports the same marker actions as `muse_eeg_gui.py`:
  - Blink (code: 10)
  - Left Motor Imagery (code: 1)
  - Right Motor Imagery (code: 2)
  - Relax (code: 3)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the experiment:

```bash
python muse_psychopy_recording.py
```

### Controls

- **SPACE**: Connect to MUSE headband
- **ENTER**: Start recording
- **ESC**: Stop recording and exit
- **B**: Add Blink marker (while recording)
- **L**: Add Left Motor Imagery marker (while recording)
- **R**: Add Right Motor Imagery marker (while recording)
- **E**: Add Relax marker (while recording)

## Output

Data is saved to `~/MuseEEG/psychopy/muse_psychopy_YYYYMMDD_HHMMSS.csv`

The CSV file contains:
- `timestamp`: Sample timestamp
- `marker_code`: Marker code (empty if no marker)
- `marker_label`: Marker label (empty if no marker)
- `EEG_0`, `EEG_1`, ...: EEG channel values

## Integration

The `MusePsychopyRecorder` class can be imported and used in custom Psychopy experiments:

```python
from muse_psychopy_recording import MusePsychopyRecorder

recorder = MusePsychopyRecorder()
recorder.connect()
recorder.start_streaming()
recorder.start_recording()

# ... your experiment code ...

recorder.add_marker('left_mi')  # Add marker
recorder.stop_recording()
recorder.disconnect()
```

