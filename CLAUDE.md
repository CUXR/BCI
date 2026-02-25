# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BCI-VR Navigation Control: Decodes left/right motor imagery and intentional blinks from Muse 2 headband EEG signals to enable hands-free VR navigation in Unity. Hardware: 4-channel EEG (TP9, AF7, AF8, TP10) at 256 Hz via BrainFlow SDK.

**Control scheme**: Left motor imagery → navigate left, Right motor imagery → navigate right, Intentional blink → confirm/select, Baseline/idle → no action.

## Setup & Commands

```bash
# Python 3.10, uses uv package manager
pip install -r requirements.txt

# Run the main experiment (requires Muse headband connected)
python src/psychopy_recording/muse_psychopy_recording_structured.py --name <name> --number <subject_number>
# e.g. python src/psychopy_recording/muse_psychopy_recording_structured.py --name alan --number 6
# Creates data/sub06/metadata.yaml automatically

# Run cross-subject analysis
python data/analyze_all.py

# Basic Muse streaming
python src/stream_save_muse.py
```

No test suite or linter is configured.

## Architecture

Three-layer design:

1. **Hardware/Streaming** (`src/stream_save_muse.py`, `src/stream_realtime_bandpower.py`, `src/rebroadcast_*.py`): BrainFlow interface for Muse 2. Streams EEG to CSV or rebroadcasts for consumers.

2. **Experiment/Recording** (`src/psychopy_recording/muse_psychopy_recording_structured.py`): The active experiment implementation. PsychoPy-based visual cue paradigm with threaded background EEG collection. Core class `MusePsychopyRecorder` orchestrates trials via state machine (`ExperimentState` enum). Produces two CSVs per session: raw EEG data + unified trial log.

3. **Analysis** (`data/analyze_all.py`, `src/eeg_filters.py`): Signal processing (MNE-based bandpass/notch filtering, artifact removal) and cross-subject visualization (PSD, band power, epoch averaging, L/R comparison).

`src/muse_eeg_gui.py` is a legacy PyQt6 GUI, superseded by PsychoPy.

## Critical Data Contracts

The **unified session log** is the central design principle (see `PROGRAMMING_REQUIREMENTS.md`). Every trial is one row with consistent columns:

- `trial_id`, `trial_type`, `label`, `start_time`, `end_time`, `event_marker`, `duration`, `session_id`
- Trial types: `motor_imagery_left`, `motor_imagery_right`, `blink_intentional`, `baseline_quiet`, `baseline_active`
- Motor imagery trials are interval-based (cue onset → spacebar press)
- Blink trials are event-centered (±200ms around button press)

**Never modify the trial log schema without updating both the recorder and `data/analyze_all.py`.**

## Data Layout

```
data/sub{NN}/
  eeg_data_YYYYMMDD_HHMMSS.csv    # Timestamp + EEG_0..EEG_3 columns
  trial_log_YYYYMMDD_HHMMSS.csv   # Unified trial log
  metadata.yaml                    # date, participant name
```

EEG column mapping: EEG_0=TP9, EEG_1=AF7, EEG_2=AF8, EEG_3=TP10.

## Key Constraints

- Timestamp synchronization between EEG data and trial log is critical — both use the same BrainFlow session clock
- Data collection runs in a background thread; shared state modifications require care
- TTS uses platform-specific commands (`say` on macOS, `espeak`/`pyttsx3` elsewhere)
- Experiment protocol details are in `EXPERIMENT_PROCEDURE.md`
- Data structure spec is in `PROGRAMMING_REQUIREMENTS.md` — treat as authoritative

## ML Pipeline (In Development)

Target classifiers: motor imagery (L/R), intentional blink, baseline/idle. Planned features: band power, CSP (Common Spatial Patterns), time-frequency. Planned models: LDA, SVM, Random Forest, with EEGNet/LSTM exploration. Use `conda` skill when running Python scripts that need environment activation.
