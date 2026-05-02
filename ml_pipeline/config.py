"""Central configuration for the BCI ML pipeline.

All tunable parameters, file paths, and constants live here.
Values for ASR cutoff and spectral thresholds are derived from:
  Delorme & Martin (2021) "Automated Data Cleaning for the Muse EEG"
  IEEE BIBM, doi:10.1109/BIBM52615.2021.9669415
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_DIR = Path(__file__).parent

DATA_DIRS = [
    PROJECT_ROOT / "bci" / "data",
    PROJECT_ROOT / "References",
]

MODELS_DIR = PIPELINE_DIR / "models"
RESULTS_DIR = PIPELINE_DIR / "results"

# ── Muse 2 channel mapping ────────────────────────────────────────
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
CH_COLS = ["EEG_0", "EEG_1", "EEG_2", "EEG_3"]
N_CHANNELS = 4

FRONTAL_INDICES = [1, 2]       # AF7, AF8
TEMPORAL_INDICES = [0, 3]      # TP9, TP10

# ── Sampling ──────────────────────────────────────────────────────
TARGET_SFREQ = 256             # resample all data to this

# ── Trial types ───────────────────────────────────────────────────
# Legacy 2-class L/R hand motor imagery (still used by the old realtime
# bundle path; new collector writes 4-class trial types instead).
MI_TYPES = ["motor_imagery_left", "motor_imagery_right"]
BLINK_TYPE = "blink_intentional"
LABEL_MAP = {"motor_imagery_left": 0, "motor_imagery_right": 1, "blink_intentional": 2}

# 4-class navigation MI used by the final-realtime pipeline (forward,
# backward, rotate-left, rotate-right). Trial-type strings and integer
# label order MUST stay aligned with pipeline.labels.MI_LABEL_TO_INT.
MI_TYPES_4CLASS = [
    "motor_imagery_forward",
    "motor_imagery_backward",
    "motor_imagery_rotate_left",
    "motor_imagery_rotate_right",
]
LABEL_MAP_4CLASS = {
    "motor_imagery_forward": 0,
    "motor_imagery_backward": 1,
    "motor_imagery_rotate_left": 2,
    "motor_imagery_rotate_right": 3,
}
MI_CLASS_NAMES_4CLASS = [
    "mi_forward",
    "mi_backward",
    "mi_rotate_left",
    "mi_rotate_right",
]

# ── Preprocessing — Channel quality (Delorme paper Table 1) ──────
SPECTRAL_THRESHOLD_DB = 25.0   # log10(µV²)/Hz in 5-55 Hz band
SPECTRAL_FREQ_RANGE = (5, 55)  # Hz — best per paper
CHANNEL_BAD_FRACTION = 0.50    # flag channel if bad >50% of trial

# ── Preprocessing — ASR (Delorme paper Table 3) ──────────────────
ASR_CUTOFF_MI = 20             # more conservative than paper's 11 for MI
ASR_CUTOFF_REST = 11           # paper's validated optimal for resting

# ── Preprocessing — Filtering ────────────────────────────────────
MI_BANDPASS = (1.0, 40.0)      # Hz — motor imagery path
MI_FEATURE_BAND = (8, 30)      # mu + beta for feature extraction
BLINK_BANDPASS = (0.5, 10.0)   # Hz — blink detection path
NOTCH_FREQ = 60.0              # Hz — North America powerline

# ── Preprocessing — Artifact thresholds ──────────────────────────
CLIP_THRESHOLD_UV = 999.0      # Muse ADC rails at ±1000 µV
ARTIFACT_PTP_MI = 250.0        # peak-to-peak rejection for MI epochs (µV)
ARTIFACT_PTP_BLINK = 500.0     # blink epochs tolerate higher amplitude
FLAT_THRESHOLD_UV = 0.5        # std below this = lost contact

# ── Epoch extraction ─────────────────────────────────────────────
MI_EPOCH_MIN_S = 2.0           # minimum usable MI trial length
MI_EPOCH_FIXED_S = 2.0         # standardised epoch length used for features
MI_PRE_STIMULUS_S = 0.0        # no pre-stimulus baseline in this paradigm
BLINK_EPOCH_S = 0.5            # ~128 samples at 256 Hz
BLINK_NEG_PER_MI = 2           # negative windows sampled per MI trial

# ── Feature extraction — frequency bands ─────────────────────────
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

MI_BANDS = {
    "mu":        (8, 12),
    "alpha":     (8, 13),
    "beta":      (13, 30),
    "low_beta":  (13, 20),
    "high_beta": (20, 30),
}

# ── Model training ───────────────────────────────────────────────
CV_FOLDS = 5
RANDOM_STATE = 42

# ── Real-time inference ──────────────────────────────────────────
RT_BLINK_WINDOW_S = 0.5
RT_MI_WINDOW_S = 2.0
RT_UPDATE_INTERVAL_MS = 100
RT_BLINK_COOLDOWN_S = 1.0
RT_MI_COOLDOWN_S = 0.3
RT_BLINK_THRESHOLD = 0.60
RT_MI_THRESHOLD = 0.55

# Per-class realtime thresholds for the 4-class pipeline. Tweak per
# direction if one class is consistently noisier than the others.
RT_PER_CLASS_THRESHOLDS = {
    "mi_forward": RT_MI_THRESHOLD,
    "mi_backward": RT_MI_THRESHOLD,
    "mi_rotate_left": RT_MI_THRESHOLD,
    "mi_rotate_right": RT_MI_THRESHOLD,
    "intentional_blink": RT_BLINK_THRESHOLD,
}
