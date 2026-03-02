# MUSE EEG Psychopy Recording

This module provides a Psychopy-based experiment interface for streaming and recording MUSE EEG data with markers, implementing the unified trial log structure per `PROGRAMMING_REQUIREMENTS.md` and experiment procedure per `EXPERIMENT_PROCEDURE.md`.

## Features

- Streams EEG data from MUSE2 headband using BrainFlow
- **Multi-device support**: Select between two Muse 2 headbands via in-app selection or `--serial` flag
- **Auto-retry BLE connection**: Automatically retries up to 3 times on Bluetooth failure
- Records raw EEG data and trial log in separate CSV files
- Implements unified trial log structure with consistent columns
- Supports motor imagery trials (left/right) with TTS audio instructions and visual cues
- Supports intentional blink trials with event-centered extraction (±500ms window)
- Supports baseline trials (quiet rest and active engagement)
- Visual interface with distinct colors/symbols for left vs. right motor imagery cues

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the experiment:

```bash
python muse_psychopy_recording_structured.py --name <name> --number <subject_number>
```

### CLI Arguments

| Argument     | Required | Description                                      |
|-------------|----------|--------------------------------------------------|
| `--name`    | Yes      | Subject's name                                   |
| `--number`  | Yes      | Subject number (e.g. 6 → saves to `data/sub06/`) |
| `--serial`  | No       | Muse Bluetooth name (e.g. `Muse-15C3`) to skip device selection screen |

### Examples

```bash
# Interactive device selection
python muse_psychopy_recording_structured.py --name alan --number 6

# Direct connection to a specific Muse
python muse_psychopy_recording_structured.py --name alan --number 6 --serial Muse-15C3
```

### Known Devices

Two Muse 2 headbands are configured in the source code (`MUSE_DEVICES` dict):

| Key | Bluetooth Name | Label          |
|-----|---------------|----------------|
| 1   | `Muse-15C3`  | Muse #1 (15C3) |
| 2   | `Muse-12A6`  | Muse #2 (12A6) |

### Controls

**Device Selection (Phase 0):**
- **1**: Select Muse #1 (15C3)
- **2**: Select Muse #2 (12A6)
- **0**: Auto-connect to any available Muse
- **ESC**: Exit

**Setup:**
- **SPACE**: Connect to selected Muse headband (auto-retries up to 3 times)
- **ENTER**: Start the structured experiment
- **ESC**: Abort and save data

**During Experiment:**
- **SPACE**: End current motor imagery trial (press when done imagining)
- **ESC**: Abort experiment early (data is saved)

**Blink Block:**
- **B**: Record intentional blink
- **ENTER**: Finish blink block early

## Experiment Flow

1. **Device Selection**: Choose which Muse headband to connect to (press 1, 2, or 0)
2. **Connection**: Press SPACE to connect (auto-retries on BLE failure)
3. **Start**: Press ENTER to begin the structured experiment
4. **Motor Imagery Blocks** (10 rounds of LEFT + RIGHT):
   - TTS audio instruction plays (full instruction on first trial, short "Grab left/right" after)
   - Recording starts immediately after audio
   - Participant imagines the movement
   - Press SPACE when done imagining
   - 5-second rest period
5. **Blink Block**: Record 10 intentional blinks (press B for each)
6. **Completion**: Data saved automatically

## Output Files

Two CSV files are created per session in `data/sub{NN}/`:

1. **`eeg_data_YYYYMMDD_HHMMSS.csv`**: Raw EEG data with markers
   - `timestamp`: Sample timestamp
   - `marker_code`: Marker code (empty if no marker)
   - `marker_label`: Marker label (empty if no marker)
   - `EEG_0` (TP9), `EEG_1` (AF7), `EEG_2` (AF8), `EEG_3` (TP10)

2. **`trial_log_YYYYMMDD_HHMMSS.csv`**: Unified trial log
   - `trial_id`: Unique identifier
   - `trial_type`: `motor_imagery_left`, `motor_imagery_right`, `blink_intentional`, `baseline_quiet`, `baseline_active`
   - `label`: Class label for ML training (`left`, `right`, `blink`, `baseline`)
   - `start_time`, `end_time`: Timestamps
   - `event_marker`: Event timestamp (for blink trials)
   - `duration`: Trial duration in seconds
   - `session_id`, `participant_id`, `block_number`, `trial_number`, `notes`

3. **`metadata.yaml`**: Session metadata (date, participant name)

## Integration

The `MusePsychopyRecorder` class can be imported and used in custom experiments:

```python
from muse_psychopy_recording_structured import MusePsychopyRecorder, TrialType

recorder = MusePsychopyRecorder(
    subject_name="alan",
    subject_number=6,
    participant_id="alan",
    serial_number="Muse-15C3",
)
recorder.connect()        # Auto-retries up to 3 times
recorder.start_streaming()
recorder.start_recording()

# Motor imagery trial
recorder.start_trial(TrialType.MOTOR_IMAGERY_LEFT, block_number=1)
# ... wait for participant ...
recorder.end_trial()

# Record a blink
recorder.add_blink_marker()

# Save and cleanup
recorder.stop_recording()
recorder.disconnect()
```
