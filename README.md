# BCI-VR Navigation Control

EEG-based control system for VR navigation using the Muse headband. Translates brain signals (motor imagery and intentional blinks) into actions for Unity environments without physical controllers.

## Goal

Decode left/right motor imagery and intentional blinks from forehead EEG to enable hands-free VR navigation.

## Current Progress

Data collection pipeline complete. Muse-to-Unity connection working via keyboard simulation. Currently collecting participant data and developing ML classification models.

## Components

| Component | Description |
|-----------|-------------|
| `src/stream_save_muse.py` | EEG data recording from Muse |
| `src/stream_realtime_bandpower.py` | Live band power computation |
| `src/psychopy_recording/` | Structured experiment paradigm |
| `src/muse_eeg_gui.py` | Real-time visualization interface |
| `src/eeg_filters.py` | Signal processing utilities |

## Setup

Python 3.11

```
pip install -r requirements.txt
```

## Current Status

### Completed
- [x] Full data collection pipeline from Muse headband
- [x] Muse to Unity connection (via keyboard input simulation)
- [x] Basic signal streaming and visualization

### In Progress
- [ ] Collecting data samples from CUXR team participants
- [ ] Building ML model for motor imagery classification
- [ ] Left/right motor imagery detection
- [ ] Intentional blink detection (as confirm action)
- [ ] Baseline/idle state classification (thinking nothing vs. active intention)

## Control Scheme Design
- **Left Motor Imagery:** Navigate/move left
- **Right Motor Imagery:** Navigate/move right
- **Intentional Blink:** Confirm/select action
- **Baseline/Idle:** No action (natural state)

