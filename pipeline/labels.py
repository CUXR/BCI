"""Canonical label taxonomy for the final-realtime BCI pipeline.

Five output classes (4 motor-imagery directions + blink) plus an internal
"idle" used by the inference loop when nothing crosses threshold.

This module is the single source of truth for:
    - trial_type strings written to trial_log_*.csv during collection
    - integer marker codes injected into the BrainFlow stream / muxed CSV
    - canonical labels emitted on the prediction WebSocket
    - default per-class confidence thresholds for realtime gating
    - direction strings used by the Unity MarkerWebSocketClient

Keep this in sync with:
    [unity/environment_1/Assets/scripts/AutoMoveDataCollectionController.cs]
    [pipeline/collector.py]
    [eeg_to_meta/config.py]
"""

from __future__ import annotations

from typing import Dict, List


# ── Trial-type strings (written to trial_log_*.csv) ─────────────────
TRIAL_MI_FORWARD = "motor_imagery_forward"
TRIAL_MI_BACKWARD = "motor_imagery_backward"
TRIAL_MI_ROTATE_LEFT = "motor_imagery_rotate_left"
TRIAL_MI_ROTATE_RIGHT = "motor_imagery_rotate_right"
TRIAL_BLINK = "blink_intentional"

MI_TRIAL_TYPES: List[str] = [
    TRIAL_MI_FORWARD,
    TRIAL_MI_BACKWARD,
    TRIAL_MI_ROTATE_LEFT,
    TRIAL_MI_ROTATE_RIGHT,
]

ALL_TRIAL_TYPES: List[str] = MI_TRIAL_TYPES + [TRIAL_BLINK]


# ── Canonical class labels emitted on the prediction WebSocket ─────
LABEL_MI_FORWARD = "mi_forward"
LABEL_MI_BACKWARD = "mi_backward"
LABEL_MI_ROTATE_LEFT = "mi_rotate_left"
LABEL_MI_ROTATE_RIGHT = "mi_rotate_right"
LABEL_BLINK = "intentional_blink"
LABEL_IDLE = "idle"

MI_CLASS_LABELS: List[str] = [
    LABEL_MI_FORWARD,
    LABEL_MI_BACKWARD,
    LABEL_MI_ROTATE_LEFT,
    LABEL_MI_ROTATE_RIGHT,
]

ALL_CLASS_LABELS: List[str] = MI_CLASS_LABELS + [LABEL_BLINK]


# Stable integer index for the MI multiclass head (must match training).
MI_LABEL_TO_INT: Dict[str, int] = {
    LABEL_MI_FORWARD: 0,
    LABEL_MI_BACKWARD: 1,
    LABEL_MI_ROTATE_LEFT: 2,
    LABEL_MI_ROTATE_RIGHT: 3,
}
MI_INT_TO_LABEL: Dict[int, str] = {v: k for k, v in MI_LABEL_TO_INT.items()}


# ── Trial-type → canonical-label mapping ───────────────────────────
TRIAL_TYPE_TO_LABEL: Dict[str, str] = {
    TRIAL_MI_FORWARD: LABEL_MI_FORWARD,
    TRIAL_MI_BACKWARD: LABEL_MI_BACKWARD,
    TRIAL_MI_ROTATE_LEFT: LABEL_MI_ROTATE_LEFT,
    TRIAL_MI_ROTATE_RIGHT: LABEL_MI_ROTATE_RIGHT,
    TRIAL_BLINK: LABEL_BLINK,
}


# ── BrainFlow / CSV marker codes ───────────────────────────────────
# Codes deliberately differ from the legacy 2-class L/R MI scheme
# (left_mi=1, right_mi=2) because the new pipeline records different
# semantics; data collected by pipeline.collector is gitignored and
# loaded only by the new trainer, so there is no in-flight collision.
MARKER_CODE_MI_FORWARD = 11
MARKER_CODE_MI_BACKWARD = 12
MARKER_CODE_MI_ROTATE_LEFT = 13
MARKER_CODE_MI_ROTATE_RIGHT = 14
MARKER_CODE_BLINK = 10  # matches legacy psychopy blink code

TRIAL_TYPE_TO_MARKER_CODE: Dict[str, int] = {
    TRIAL_MI_FORWARD: MARKER_CODE_MI_FORWARD,
    TRIAL_MI_BACKWARD: MARKER_CODE_MI_BACKWARD,
    TRIAL_MI_ROTATE_LEFT: MARKER_CODE_MI_ROTATE_LEFT,
    TRIAL_MI_ROTATE_RIGHT: MARKER_CODE_MI_ROTATE_RIGHT,
    TRIAL_BLINK: MARKER_CODE_BLINK,
}

MARKER_CODE_TO_TRIAL_TYPE: Dict[int, str] = {
    v: k for k, v in TRIAL_TYPE_TO_MARKER_CODE.items()
}


# ── Unity AutoMove direction strings ↔ trial types ─────────────────
# AutoMoveDataCollectionController emits one of these direction values
# along with action_cue / action_start / action_end events. The marker
# server translates them to MI trial types (blinks are cued differently).
UNITY_DIRECTION_FORWARD = "forward"
UNITY_DIRECTION_BACKWARD = "backward"
UNITY_DIRECTION_LEFT_ROTATE = "left_rotate"
UNITY_DIRECTION_RIGHT_ROTATE = "right_rotate"

UNITY_DIRECTION_TO_TRIAL_TYPE: Dict[str, str] = {
    UNITY_DIRECTION_FORWARD: TRIAL_MI_FORWARD,
    UNITY_DIRECTION_BACKWARD: TRIAL_MI_BACKWARD,
    UNITY_DIRECTION_LEFT_ROTATE: TRIAL_MI_ROTATE_LEFT,
    UNITY_DIRECTION_RIGHT_ROTATE: TRIAL_MI_ROTATE_RIGHT,
}

TRIAL_TYPE_TO_UNITY_DIRECTION: Dict[str, str] = {
    v: k for k, v in UNITY_DIRECTION_TO_TRIAL_TYPE.items()
}


# ── Default per-class confidence thresholds for realtime gating ────
# Override at runtime via `pipeline.cli realtime --mi-threshold ...`
# or `--blink-threshold ...`; per-class fine tuning lives in
# eeg_to_meta/config.CLASS_THRESHOLDS.
DEFAULT_MI_THRESHOLD = 0.55
DEFAULT_BLINK_THRESHOLD = 0.60

DEFAULT_CLASS_THRESHOLDS: Dict[str, float] = {
    LABEL_MI_FORWARD: DEFAULT_MI_THRESHOLD,
    LABEL_MI_BACKWARD: DEFAULT_MI_THRESHOLD,
    LABEL_MI_ROTATE_LEFT: DEFAULT_MI_THRESHOLD,
    LABEL_MI_ROTATE_RIGHT: DEFAULT_MI_THRESHOLD,
    LABEL_BLINK: DEFAULT_BLINK_THRESHOLD,
}
