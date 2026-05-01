# Pilot EEG Data Collection Module

This module contains scripts used to collect pilot EEG recordings from Muse 2 headbands, including the structured PsychoPy experiment, low-level stream capture, and multicast rebroadcast utilities.

## Purpose

- Run reproducible pilot data collection sessions for motor imagery and intentional blink tasks.
- Save synchronized EEG samples and trial metadata for downstream analysis and model training.
- Provide fallback collection paths (basic CSV streaming and legacy GUI) when full PsychoPy sessions are not needed.

## Setup Requirements

- Python dependencies installed from repository root:
  - `pip install -r requirements.txt`
- Hardware/software:
  - Muse 2 headband (Bluetooth connected)
  - BrainFlow-compatible environment
  - PsychoPy installed for structured experiment runs
- Optional:
  - macOS `say` command (or Linux `espeak`) for text-to-speech instructions in the structured experiment

## Module Layout

- `psychopy_recording/muse_psychopy_recording_structured.py`: Primary structured pilot protocol.
- `stream_save_muse.py`: Quick raw streaming to CSV.
- `rebroadcast_producer.py`: Streams Muse data to multicast.
- `rebroadcast_consumer.py`: Subscribes to multicast stream.
- `muse_eeg_gui.py`: Legacy PyQt GUI recorder (kept for compatibility/debugging).

## How To Run

From repository root:

```bash
# Main pilot experiment (recommended)
python src/pilot_data_collection/psychopy_recording/muse_psychopy_recording_structured.py --name <name> --number <subject_number>

# Optional: target a specific Muse by Bluetooth name
python src/pilot_data_collection/psychopy_recording/muse_psychopy_recording_structured.py --name <name> --number <subject_number> --serial Muse-15C3

# Quick raw CSV capture (10s example)
python src/pilot_data_collection/stream_save_muse.py

# Multicast rebroadcast utilities
python src/pilot_data_collection/rebroadcast_producer.py
python src/pilot_data_collection/rebroadcast_consumer.py

# Legacy GUI recorder
python src/pilot_data_collection/muse_eeg_gui.py
```

## Expected Inputs / Outputs

Inputs:

- Muse 2 EEG stream (TP9, AF7, AF8, TP10) via BrainFlow.
- Participant metadata via CLI (`--name`, `--number`, optional `--serial`) for structured runs.

Outputs (structured experiment):

- `data/sub{NN}/metadata.yaml`
- `data/sub{NN}/eeg_data_YYYYMMDD_HHMMSS.csv`
- `data/sub{NN}/trial_log_YYYYMMDD_HHMMSS.csv`

Output schema contracts follow `PROGRAMMING_REQUIREMENTS.md`:

- Trial log columns include `trial_id`, `trial_type`, `label`, `start_time`, `end_time`, `event_marker`, `duration`, `session_id`.
- Blink trials are event-centered; motor imagery trials are interval-based.

## Known Limitations

- Real device connection quality depends on Bluetooth stability and local RF noise.
- Script defaults (for example notch frequency and fixed loop timing in utility scripts) may require local adjustment.
- `muse_eeg_gui.py` is legacy and superseded by the structured PsychoPy protocol.
- No formal automated test suite exists for acquisition scripts; validation is currently manual/hardware-in-the-loop.
