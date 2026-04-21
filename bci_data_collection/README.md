# bci-data-collection

Two-program pipeline that streams raw EEG from a MUSE 2 headband to a local
multicast port, classifies windows of that data, and serves the resulting
labels (`LEFT` / `RIGHT` / `FORWARD` / `BACKWARD`) over TCP for downstream
consumers (Unity, robots, dashboards, etc.).

## Data flow

```
MUSE 2 headband
    │ Bluetooth
    ▼
┌──────────────────┐
│  eeg_producer.py │  Program 1 — BrainFlow multicast (UDP)
└────────┬─────────┘
         │ streaming_board://225.1.1.1:6677
         ▼
┌──────────────────────┐
│  classify_server.py  │  Program 2 — windowed classification
│  └─ classifier.py    │  (stub today; pluggable seam for real model)
└────────┬─────────────┘
         │ TCP 0.0.0.0:5000 (newline-delimited JSON)
         ▼
   External device(s)
```

Each TCP message is a single JSON line, e.g.:

```json
{"label": "LEFT", "confidence": 0.73, "ts": 1713200042.1}
```

## Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for environment + dependency management
- A MUSE 2 headband paired over Bluetooth (for live runs)

## Setup

```bash
uv sync
```

That installs `brainflow`, `numpy`, and `pytest` (dev) into a local `.venv`.

## Run

Open three terminals.

**Terminal 1 — start the EEG producer:**

```bash
uv run python src/eeg_producer.py
```

**Terminal 2 — start the classification server:**

```bash
uv run python src/classify_server.py
```

**Terminal 3 — connect a TCP client to read classifications:**

```bash
nc localhost 5000
```

You should see one JSON line per classification (~10 Hz with the default
0.1 s loop sleep).

### CLI flags

Both programs accept overrides:

| Program | Flag | Default | Purpose |
|---|---|---|---|
| `eeg_producer.py` | `--ip` | `225.1.1.1` | Multicast group / unicast target |
| `eeg_producer.py` | `--port` | `6677` | UDP port |
| `classify_server.py` | `--eeg-ip` | `225.1.1.1` | Must match producer `--ip` |
| `classify_server.py` | `--eeg-port` | `6677` | Must match producer `--port` |
| `classify_server.py` | `--tcp-port` | `5000` | TCP port for downstream clients |
| `classify_server.py` | `--window-sec` | `1.0` | Classification window length |

## Tests

```bash
uv run pytest tests/ -v
```

The unit tests cover the classifier contract and TCP message format. They
do **not** require a MUSE — hardware-dependent paths are exercised by the
end-to-end run above.

## Layout

```
bci_data_collection/
├── pyproject.toml
├── src/
│   ├── eeg_producer.py     # Program 1: MUSE → BrainFlow multicast
│   ├── classify_server.py  # Program 2: multicast → classify → TCP
│   └── classifier.py       # Pluggable classifier (stub today)
└── tests/
    ├── test_classifier.py
    └── test_message.py
```

## Swapping in a real model

`src/classifier.py` exposes one function:

```python
def classify(window, channels) -> tuple[str, float]:
    # window:   numpy array, shape (n_channels, n_samples)
    # channels: list of EEG channel indices (from BrainFlow)
    # returns:  (label, confidence) where label ∈ LABELS
```

Drop in feature extraction + a trained model (e.g. unpickle a sklearn
pipeline) inside that function and the rest of the pipeline keeps working.
