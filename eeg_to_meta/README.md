# eeg_to_meta — Muse EEG → Meta Quest Real-Time Inference Pipeline

Real-time backend that loads a pre-trained `.pkl` model, receives live
Muse 2 EEG data, runs classification, and sends predictions to a
Unity/Meta Quest app over WebSocket.

## Classes

| Label                  | Description                     |
|------------------------|---------------------------------|
| `idle`                 | No action detected              |
| `intentional_blink`    | Deliberate double/hard blink    |
| `left_motor_imagery`   | Imagining left hand grab        |
| `right_motor_imagery`  | Imagining right hand grab       |

## Architecture

```
Muse 2 (BLE)
  │  BrainFlow
  ▼
muse_stream.py ──push──► buffer.py (ring buffer)
                              │
                         get_window()
                              │
                              ▼
                        inference.py
                        ├─ preprocess.py  (bandpass, notch, detrend)
                        ├─ feature extraction (band powers, Hjorth, CSP)
                        └─ model_loader.py (.pkl model)
                              │
                         prediction
                              │
                              ▼
                        smoothing.py (majority vote + debounce)
                              │
                              ▼
                        websocket_server.py ──JSON──► Unity / Meta Quest
```

## Folder Structure

```
eeg_to_meta/
├── config.py              # all constants, thresholds, paths
├── model_loader.py        # load & validate .pkl model
├── buffer.py              # thread-safe rolling EEG ring buffer
├── preprocess.py          # bandpass, notch, detrend, artifact removal
├── inference.py           # feature extraction + model prediction
├── smoothing.py           # majority vote, confidence averaging, debounce
├── websocket_server.py    # async WS server → Unity/Quest
├── muse_stream.py         # Muse 2 via BrainFlow + mock stream
├── main.py                # entry point — wires everything together
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test with synthetic EEG (no Muse needed)
python main.py --mock

# 3. Test with BrainFlow synthetic board
python main.py --simulate

# 4. Live Muse 2 + WebSocket to Unity
python main.py

# 5. Headless mode (no WebSocket)
python main.py --mock --no-ws

# 6. Custom model path
python main.py --model /path/to/your/model.pkl

# 7. Quest / Unity on another machine (listen on all interfaces — trusted LAN only)
python main.py --ws-host 0.0.0.0
```

## Privacy and network exposure

- **Raw EEG** never leaves the machine in `eeg_to_meta` except via your Muse BLE link; the WebSocket carries **class predictions only**, not voltage traces.
- The WebSocket server defaults to **127.0.0.1**. Use `--ws-host 0.0.0.0` only on a **trusted network** so a headset on Wi‑Fi can connect; there is **no encryption or client authentication** (`ws://` only).
- **`bci_data_collection/eeg_producer.py`** sends **raw EEG** over BrainFlow’s streamer (default: multicast). Treat that path like sensitive health data: isolated VLAN, lab Wi‑Fi, or VPN — not public hotspots.
- **`bci_data_collection/classify_server.py`** defaults TCP bind to **127.0.0.1**; pass `--tcp-bind 0.0.0.0` only if you need remote clients and accept LAN exposure.

## WebSocket JSON Format

Every ~100 ms the server sends a JSON frame to all connected clients:

```json
{
  "timestamp": 1711234567.89,
  "predicted_class": "left_motor_imagery",
  "confidence": 0.73,
  "raw_probs": {
    "left": 0.73,
    "right": 0.27
  },
  "stable": true
}
```

### Unity Client Expectations

- Connect to `ws://<backend-ip>:8765`
- Messages are UTF-8 JSON, one per WebSocket frame
- Frequency ≈ 10 Hz
- `predicted_class` values: `"idle"`, `"intentional_blink"`, `"left_motor_imagery"`, `"right_motor_imagery"`
- Use `"stable": true` to filter out noisy single-frame predictions
- `"confidence"` is 0.0–1.0 (averaged over the smoothing window)
- `"raw_probs"` contains per-class probabilities from the model for the current frame

### Example Unity C# Listener

```csharp
using NativeWebSocket;
using UnityEngine;
using Newtonsoft.Json.Linq;

public class BCIListener : MonoBehaviour
{
    WebSocket ws;

    async void Start()
    {
        ws = new WebSocket("ws://192.168.1.100:8765");
        ws.OnMessage += (bytes) =>
        {
            var msg = System.Text.Encoding.UTF8.GetString(bytes);
            var json = JObject.Parse(msg);
            string action = json["predicted_class"].ToString();
            bool stable = json["stable"].ToObject<bool>();
            float conf = json["confidence"].ToObject<float>();

            if (stable && conf > 0.5f)
            {
                Debug.Log($"BCI action: {action} ({conf:P0})");
                // Dispatch to your game logic here
            }
        };
        await ws.Connect();
    }

    void Update() { ws?.DispatchMessageQueue(); }
    async void OnDestroy() { await ws?.Close(); }
}
```

## Model Details

The pipeline loads `ml_pipeline/models/realtime_models.pkl` which contains:

| Key              | Type              | Shape / Value                |
|------------------|-------------------|------------------------------|
| `mi_pipeline`    | VotingClassifier  | expects 82 features          |
| `mi_csp_W`       | ndarray           | (4, 4) CSP spatial filter    |
| `blink_pipeline` | VotingClassifier  | expects 75 features          |
| `sampling_rate`  | int               | 256                          |
| `ch_names`       | list              | ["TP9","AF7","AF8","TP10"]   |

### Feature Breakdown

**Motor Imagery (82 features)**:
- 5 bands × 4 channels × 2 (absolute + relative power) = 40
- 4 channels × 3 Hjorth params = 12
- 4 channels × 3 stats (ZCR, kurtosis, skew) = 12
- 4 channels × 1 alpha/beta ratio = 4
- 2 pairs × 5 bands asymmetry = 10
- **Subtotal: 78 spectral/time features**
- 4 CSP log-variance features
- **Total: 82**

**Blink Detection (75 features)**:
- 4 channels × 3 amplitude (ptp, rms, max_abs) = 12
- 4 channels × 3 Hjorth = 12
- 4 channels × 4 stats (skew, kurtosis, zcr, peak_pos) = 16
- 4 channels × 8 band powers = 32
- 1 frontal correlation + 1 temporal correlation + 1 amplitude ratio = 3
- **Total: 75**

## What I Still Need to Verify About the .pkl File

1. **ASR consistency**: The training pipeline attempts to use ASR (Artifact
   Subspace Reconstruction) if `asrpy` or `meegkit` is installed.  Check
   whether ASR was actually active during training (`pip list | grep asr`
   in the training environment).  If it was, install the same package here
   and enable the commented-out ASR call in `preprocess.py`.

2. **scikit-learn version**: The model was trained with sklearn 1.8.0.
   Loading it with a different major version may silently produce wrong
   results.  Run `python -c "import sklearn; print(sklearn.__version__)"`.

3. **Feature order**: The feature vector is order-dependent.  This pipeline
   reproduces the exact loop order from `ml_pipeline/features.py`.  If that
   file was modified after training, the features here will be wrong.

4. **Channel mapping**: BrainFlow's `get_eeg_channels()` returns hardware
   channel indices.  Verify that channel 0 = TP9, 1 = AF7, 2 = AF8,
   3 = TP10 for your Muse firmware version.

5. **Sampling rate**: Muse 2 streams at 256 Hz.  If your board reports a
   different rate (check `BoardShim.get_sampling_rate(board_id)`), the
   spectral features will shift.  Resample before feature extraction.

## Latency Budget

| Stage              | Typical Time |
|--------------------|-------------|
| Buffer read        | < 0.1 ms    |
| Preprocessing      | ~ 2 ms      |
| Feature extraction | ~ 5 ms      |
| Model prediction   | ~ 1 ms      |
| Smoothing          | < 0.1 ms    |
| WebSocket send     | < 1 ms      |
| **Total**          | **~10 ms**  |

Target loop interval is 100 ms (10 Hz), so there is ~90 ms of headroom.
