# MUSE EEG Psychopy Recording

This module provides a Psychopy-based experiment interface for streaming and recording MUSE EEG data with markers, implementing the unified trial log structure per `PROGRAMMING_REQUIREMENTS.md` and experiment procedure per `EXPERIMENT_PROCEDURE.md`.

## Features

- Streams EEG data from MUSE2 headband using BrainFlow
- Records raw EEG data and trial log in separate CSV files
- Implements unified trial log structure with consistent columns
- Supports motor imagery trials (left/right) with visual cues and participant-controlled end markers
- Supports intentional blink trials with event-centered extraction (±200ms window)
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
python muse_psychopy_recording.py
```

### Controls

**Setup:**
- **SPACE**: Connect to MUSE headband
- **ENTER**: Start recording
- **ESC**: Stop recording and exit

**Motor Imagery Trials:**
- **L**: Start LEFT motor imagery trial (visual cue appears)
- **R**: Start RIGHT motor imagery trial (visual cue appears)
- **SPACEBAR**: End current trial (when cue is displayed)

**Baseline Trials:**
- **N**: Enter baseline mode
- **Q**: Start quiet rest baseline
- **A**: Start active engagement baseline
- **SPACEBAR**: End baseline trial

**Blink Block:**
- **K**: Enter blink block (typically at end of experiment)
- **B**: Record intentional blink (press button at same moment as blink)
- **K**: Finish blink block (after recording blinks)

## Output Files

Two CSV files are created with timestamp:

1. **`eeg_data_YYYYMMDD_HHMMSS.csv`**: Raw EEG data with markers
   - `timestamp`: Sample timestamp
   - `marker_code`: Marker code (empty if no marker)
   - `marker_label`: Marker label (empty if no marker)
   - `EEG_0`, `EEG_1`, ...: EEG channel values

2. **`trial_log_YYYYMMDD_HHMMSS.csv`**: Unified trial log per PROGRAMMING_REQUIREMENTS.md
   - `trial_id`: Unique identifier
   - `trial_type`: Type of trial (`motor_imagery_left`, `motor_imagery_right`, `blink_intentional`, `baseline_quiet`, `baseline_active`)
   - `label`: Class label for ML training
   - `start_time`: Start timestamp
   - `end_time`: End timestamp
   - `event_marker`: Event timestamp (for blink trials)
   - `duration`: Trial duration in seconds
   - `session_id`: Session identifier
   - `participant_id`: Participant identifier (if provided)
   - `block_number`: Block number
   - `trial_number`: Trial number within session
   - `notes`: Optional notes

## Experiment Flow

The experiment follows `EXPERIMENT_PROCEDURE.md`:

1. **Pre-experiment**: Meditation training with Muse app (done separately)
2. **Setup**: Connect MUSE, start recording
3. **Motor Imagery Blocks**: 
   - Left imagery trials (press L, then SPACEBAR when done)
   - Right imagery trials (press R, then SPACEBAR when done)
   - Optional rest periods between blocks
4. **Baseline Collection**:
   - Quiet rest periods (press N, then Q, then SPACEBAR)
   - Active engagement periods (press N, then A, then SPACEBAR)
5. **Intentional Blinking**: 
   - 10–20 intentional blink trials at the end (press K, then B for each blink, then K to finish)

## Trial Types

### Motor Imagery Trials
- **Type**: `motor_imagery_left` or `motor_imagery_right`
- **Extraction**: Interval-based from visual cue onset to SPACEBAR press
- **Visual Cues**: Distinct colors (cyan-blue for left, green for right) with arrows

### Blink Trials
- **Type**: `blink_intentional`
- **Extraction**: Event-centered with ±200ms window around button press
- **Procedure**: Press B at the same moment as intentional blink

### Baseline Trials
- **Types**: `baseline_quiet` or `baseline_active`
- **Extraction**: Interval-based from start to SPACEBAR press
- **Usage**: For false positive control and threshold calibration

## Integration

The `MusePsychopyRecorder` class can be imported and used in custom Psychopy experiments:

```python
from muse_psychopy_recording import MusePsychopyRecorder, TrialType

recorder = MusePsychopyRecorder(participant_id="P001", blink_window_ms=200)
recorder.connect()
recorder.start_streaming()
recorder.start_recording()

# Start a motor imagery trial
recorder.start_trial(TrialType.MOTOR_IMAGERY_LEFT, block_number=1)
# ... wait for participant ...
recorder.end_trial()

# Record a blink
recorder.add_blink_marker()

# Save and cleanup
recorder.stop_recording()
recorder.disconnect()
```

## Data Processing

The trial log CSV can be filtered by `trial_type` and `label` when building ML datasets:

- Motor Imagery Classifier: Filter `trial_type IN ('motor_imagery_left', 'motor_imagery_right')`
- Blink Classifier: Filter `trial_type == 'blink_intentional'`
- Baseline: Filter `trial_type IN ('baseline_quiet', 'baseline_active')` for threshold calibration

EEG segments are extracted from the raw EEG data file using the timestamps in the trial log.
