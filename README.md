# BCI-VR Navigation Control

EEG-based control system for VR navigation using the Muse 2 headband. Decodes left/right motor imagery and intentional blinks from 4-channel forehead EEG to enable hands-free VR navigation in Unity — no physical controllers required.

## Control Scheme

| Brain Signal | Action | Accuracy |
|---|---|---|
| Left motor imagery | Navigate left | ~61% |
| Right motor imagery | Navigate right | ~61% |
| Intentional blink | Confirm / select | **88.7%** |
| Idle / baseline | No action | — |

## Hardware

- **Muse 2 headband** — 4 dry EEG electrodes at 256 Hz via Bluetooth
- **Channels:** TP9 (left temporal), AF7 (left frontal), AF8 (right frontal), TP10 (right temporal)
- **Interface:** BrainFlow SDK

## Quick Start

```bash
# 1. Setup (Python 3.11, conda recommended)
conda create -n muse python=3.11 -y
conda activate muse
pip install -r requirements.txt

# 2. Collect data (requires Muse headband)
python src/psychopy_recording/muse_psychopy_recording_structured.py

# 3. Train classifiers
python src/realtime_feedback/train_and_save_models.py

# 4. Run real-time feedback
python src/realtime_feedback/realtime_feedback.py              # with Muse
python src/realtime_feedback/realtime_feedback.py --simulate   # without hardware
```

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Muse 2 EEG     │────>│  BrainFlow SDK   │────>│  Processing     │
│  4ch @ 256 Hz   │ BT  │  Ring Buffer     │     │  Pipeline       │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌─────────────────────────────────┤
                        │                                 │
                 ┌──────▼───────┐               ┌────────▼────────┐
                 │ Blink Detect │               │ Motor Imagery   │
                 │ LDA, 500ms   │               │ RF, 3s window   │
                 │ 88.7% acc    │               │ ~61% acc        │
                 └──────┬───────┘               └────────┬────────┘
                        │                                 │
                        └─────────────┬───────────────────┘
                                      │
                              ┌───────▼───────┐
                              │  PyQt6 GUI    │
                              │  Real-time    │
                              │  Feedback     │
                              └───────────────┘
```

### Three Layers

1. **Hardware / Streaming** — BrainFlow interface for Muse 2. Streams EEG to CSV or rebroadcasts over multicast for multiple consumers.

2. **Experiment / Recording** — PsychoPy-based visual cue paradigm with threaded background EEG collection. Produces paired CSVs per session (raw EEG + trial log). See [Experiment Procedure](EXPERIMENT_PROCEDURE.md).

3. **Analysis / ML** — Signal processing (MNE bandpass/notch filtering, artifact removal), feature extraction (band power, Hjorth, CSP, time-frequency), and classification (LDA, SVM, RF). Three pipeline versions tested across 4 subjects.

## Project Structure

```
src/
  training/                             # Offline ML training pipelines
    bandpower/
      train_motor_imagery.py            #   v1: band power + Hjorth + CSP
    timefreq/
      train_motor_imagery_v2_timefreq.py  # v2: + time-frequency features
    eegnet/
      train_motor_imagery_v3_eegnet.py  #   v3: + EEGNet-inspired features
  realtime_feedback/                    # Real-time feedback system
    train_and_save_models.py            #   Train & save models for real-time use
    realtime_feedback.py                #   PyQt6 GUI with live classification
  train_blink_detector.py              # Blink detection pipeline
  eeg_filters.py                       # MNE-based bandpass, notch, artifact removal
  psychopy_recording/
    muse_psychopy_recording_structured.py  # Data collection experiment
  stream_save_muse.py                  # Basic Muse streaming to CSV
  stream_realtime_bandpower.py         # Live band power computation
  rebroadcast_producer.py              # Multicast EEG streaming (producer)
  rebroadcast_consumer.py              # Multicast EEG streaming (consumer)
  visualize_data_live.py               # Real-time EEG + band power plots
  muse_eeg_gui.py                      # Legacy PyQt6 GUI (superseded)

models/
  realtime_models.pkl                  # Trained model bundle for real-time use

data/
  sub{NN}/                             # Per-subject directories
    eeg_data_YYYYMMDD_HHMMSS.csv       #   Raw EEG (timestamp + 4 channels)
    trial_log_YYYYMMDD_HHMMSS.csv      #   Trial metadata
    metadata.yaml
  classification_summary.md            # Full results breakdown
  classification_results*.json         # Raw pipeline outputs
  blink_detection_results.json
```

## Data Collection

Each session records two CSVs:

**EEG data** — continuous 4-channel EEG with timestamps aligned to BrainFlow's session clock:
```
timestamp, marker_code, marker_label, EEG_0(TP9), EEG_1(AF7), EEG_2(AF8), EEG_3(TP10)
```

**Trial log** — one row per trial with consistent schema:
```
trial_id, trial_type, label, start_time, end_time, event_marker, duration, session_id
```

Trial types: `motor_imagery_left`, `motor_imagery_right`, `blink_intentional`, `baseline_quiet`, `baseline_active`

Motor imagery epochs are interval-based (cue onset to spacebar press, 2–5s). Blink epochs are event-centered (500ms window around button press).

## Classification Results

4 subjects (sub02–sub05), 77 MI epochs, 190 blink epochs. Full breakdown in [`data/classification_summary.md`](data/classification_summary.md).

### Blink Detection

| Evaluation | Config | Classifier | Accuracy |
|---|---|---|---|
| Within-subject | all 4 channels | LDA | **88.7%** |
| Cross-subject (LOSO) | all 4 channels | LDA | **86.0%** |

### Motor Imagery (Left vs Right)

| Evaluation | Pipeline | Config | Classifier | Accuracy |
|---|---|---|---|---|
| Within-subject | v1 Band Power | all 4 channels | SVM-rbf | **61.0%** |
| Cross-subject (LOSO) | v3 EEGNet | temporal only | RF | **62.2%** |

Three feature pipelines were compared:
- **v1** — Band power + Hjorth + lateralization asymmetry + CSP
- **v2** — v1 + sliding-window time-frequency (ERD/ERS dynamics)
- **v3** — v2 + EEGNet-inspired fixed spatial/temporal features

Key findings: Blink detection is deployment-ready. Motor imagery is near chance with 4 subjects — needs more data or paradigm refinement. Random Forest generalizes best across subjects.

## Real-Time Feedback System

The real-time GUI (`src/realtime_feedback/realtime_feedback.py`) provides live brain state feedback:

- **Blink detection** — 500ms sliding window, LDA classifier, ~1s cooldown
- **Motor imagery** — 3s sliding window, Random Forest with CSP features
- **Band powers** — live delta/theta/mu/alpha/beta/gamma visualization
- **Raw EEG** — 4-channel waveform display (last 5 seconds)

Adjustable confidence thresholds via the GUI. Blink detection has priority over motor imagery each update cycle (5 Hz).

### Keyboard Shortcuts

| Key | Action |
|---|---|
| Space | Start / stop classification |
| Q | Quit |

## Preprocessing Pipeline

Applied per-channel, matching both offline training and real-time inference:

1. NaN/Inf sanitization
2. Moving artifact removal (rolling z-score, 3-sigma threshold, 0.5s window)
3. Linear detrend
4. Notch filter (60 Hz powerline removal)
5. Bandpass filter (1–40 Hz, 4th-order Butterworth)

## Dependencies

- `brainflow` — EEG acquisition
- `mne` — Signal filtering
- `scikit-learn` — ML classifiers
- `PyQt6`, `pyqtgraph` — Real-time GUI
- `psychopy` — Experiment paradigm
- `numpy`, `scipy`, `pandas` — Numerics
- `torch` — EEGNet features (v3 pipeline only)

## Team

| Name | Role |
|---|---|
| Alan Wu | Algorithm design, full-stack development |
| Michelle | Hardware / electronics integration |
| Spencer | Software development |
| Ashee | Research, HCI, user studies |

Cornell CUXR Lab
