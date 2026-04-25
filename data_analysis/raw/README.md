# Raw

Per-subject views of the **unprocessed** EEG signal — directly from the
CSVs in `data/Muse_Data/Phase{1,2}/sub*/`. No filtering, no feature
extraction; just the recorded waveform with trial markers overlaid.

## Per-subject data structure

Each `data/Muse_Data/Phase*/sub*/` directory contains:

| File | Contents |
|---|---|
| `eeg_data_YYYYMMDD_HHMMSS.csv` | Continuous 4-channel EEG @ 256 Hz: `timestamp, marker_code, marker_label, EEG_0(TP9), EEG_1(AF7), EEG_2(AF8), EEG_3(TP10)` |
| `trial_log_YYYYMMDD_HHMMSS.csv` | One row per trial: `trial_id, trial_type, label, start_time, end_time, event_marker, duration, session_id` |
| `metadata.yaml` | `participant`, `date`, etc. |

Trial types: `motor_imagery_left`, `motor_imagery_right`,
`blink_intentional`, `baseline_quiet`, `baseline_active`.

## Plots produced

`generate.py` writes three plots per subject into
`outputs/<subject_key>/`:

| File | What it shows |
|---|---|
| `plot_01_continuous.png` | Full session, 4 channels, with trial spans color-coded by type. Use this to spot dropouts, large artifacts, or session-wide drift. |
| `plot_02_averaged_epochs.png` | Mean ± std epochs (5 s window) per trial type per channel, with individual trials underlaid. Use this to see whether evoked responses are visible above the trial-to-trial variability. |
| `plot_04_blinks.png` | Zoomed view of the blink block with trial-window shading and event-marker lines. Use this to confirm blink markers actually align with the EOG deflections. |

## Run

```bash
python data_analysis/raw/generate.py
```

Subject keys come from `data_analysis/_common.py::find_subject_data` —
recursive over `data/`, with parent-folder disambiguation when the same
subject appears in multiple phases (e.g. `Phase2_sub05`).

`outputs/` is gitignored — regenerated on demand.
