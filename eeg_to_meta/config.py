"""Runtime configuration for Muse EEG → Meta Quest inference pipeline.

All constants mirror the training pipeline (ml_pipeline/config.py) to
guarantee preprocessing parity.  Change values here ONLY if you also
retrain the model.
"""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "ml_pipeline" / "models" / "realtime_models.pkl"

# ── Muse 2 hardware ────────────────────────────────────────────────
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
N_CHANNELS = 4
SFREQ = 256                    # Hz — must match training pipeline

FRONTAL_INDICES = [1, 2]       # AF7, AF8
TEMPORAL_INDICES = [0, 3]      # TP9, TP10

# ── Preprocessing — identical to ml_pipeline/config.py ─────────────
MI_BANDPASS = (1.0, 40.0)      # Hz
BLINK_BANDPASS = (0.5, 10.0)   # Hz
NOTCH_FREQ = 60.0              # Hz — North America powerline
CLIP_THRESHOLD_UV = 999.0      # Muse ADC rails at ±1000 µV
FLAT_THRESHOLD_UV = 0.5        # std below this = lost contact

# ── Window geometry ─────────────────────────────────────────────────
MI_WINDOW_S = 2.0              # seconds — motor imagery analysis window
BLINK_WINDOW_S = 0.5           # seconds — blink detection window
MI_WINDOW_SAMPLES = int(MI_WINDOW_S * SFREQ)         # 512
BLINK_WINDOW_SAMPLES = int(BLINK_WINDOW_S * SFREQ)   # 128
STRIDE_S = 0.1                 # inference runs every 100 ms
STRIDE_SAMPLES = int(STRIDE_S * SFREQ)               # ~26

# ── Feature dimensions (from trained model inspection) ──────────────
MI_SPECTRAL_FEATURES = 78      # per-channel band powers + Hjorth + stats
MI_CSP_FEATURES = 4            # from 4×4 CSP matrix
MI_TOTAL_FEATURES = 82         # 78 + 4
BLINK_TOTAL_FEATURES = 75

# ── Frequency bands — must match ml_pipeline/config.py ──────────────
MI_BANDS = {
    "mu":        (8, 12),
    "alpha":     (8, 13),
    "beta":      (13, 30),
    "low_beta":  (13, 20),
    "high_beta": (20, 30),
}

BANDS = {
    "delta":     (1, 4),
    "theta":     (4, 8),
    "mu":        (8, 12),
    "alpha":     (8, 13),
    "beta":      (13, 30),
    "low_beta":  (13, 20),
    "high_beta": (20, 30),
    "gamma":     (30, 45),
}

# ── Inference thresholds ────────────────────────────────────────────
BLINK_CONFIDENCE_THRESHOLD = 0.60
MI_CONFIDENCE_THRESHOLD = 0.55  # default fallback for any MI class

# Per-class thresholds (4-class MI + blink). Override at runtime via
# eeg_to_meta/main.py --mi-threshold / --blink-threshold or by setting
# values directly on this dict from pipeline.cli.
CLASS_THRESHOLDS: dict[str, float] = {
    "mi_forward": MI_CONFIDENCE_THRESHOLD,
    "mi_backward": MI_CONFIDENCE_THRESHOLD,
    "mi_rotate_left": MI_CONFIDENCE_THRESHOLD,
    "mi_rotate_right": MI_CONFIDENCE_THRESHOLD,
    "intentional_blink": BLINK_CONFIDENCE_THRESHOLD,
}

# ── Smoothing / debounce ───────────────────────────────────────────
SMOOTHING_WINDOW = 5           # majority-vote over last N predictions
BLINK_COOLDOWN_S = 1.0         # suppress repeat blink triggers
MI_COOLDOWN_S = 0.3            # suppress rapid MI toggles
STABILITY_MIN_AGREE = 3        # out of SMOOTHING_WINDOW to be "stable"

# ── WebSocket ──────────────────────────────────────────────────────
# Default localhost so predictions are not exposed on every interface.
# For Quest/Unity on another device on the LAN, run main.py with --ws-host 0.0.0.0
# (still plaintext / unauthenticated — use only on trusted networks).
WS_HOST = "127.0.0.1"
WS_PORT = 8765

# ── Class labels ───────────────────────────────────────────────────
# Canonical labels emitted on the prediction WebSocket. The 4-class MI
# pipeline adds forward / backward / rotate_left / rotate_right while
# keeping the legacy left / right keys around so the older 2-class
# realtime path keeps working until the new inference engine lands.
CLASS_NAMES = {
    "idle": "idle",
    "blink": "intentional_blink",
    # Legacy 2-class motor imagery (kept for backward compatibility).
    "left": "left_motor_imagery",
    "right": "right_motor_imagery",
    # 4-class navigation MI (used by the final-realtime pipeline).
    "forward": "mi_forward",
    "backward": "mi_backward",
    "rotate_left": "mi_rotate_left",
    "rotate_right": "mi_rotate_right",
}

# Advisory hints sent to Unity (debug bindings only — FreeNavController
# uses the model output directly and does not require these keys).
KEY_HINTS = {
    "left_motor_imagery": "A",
    "right_motor_imagery": "D",
    "mi_forward": "W",
    "mi_backward": "S",
    "mi_rotate_left": "A",
    "mi_rotate_right": "D",
    "intentional_blink": "SPACE",
    "idle": "—",
}

# ── BrainFlow board IDs ───────────────────────────────────────────
MUSE_2_BOARD_ID = 22           # BoardIds.MUSE_2_BOARD
SYNTHETIC_BOARD_ID = -1        # BoardIds.SYNTHETIC_BOARD
