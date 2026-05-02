"""Real-time EEG preprocessing — mirrors ml_pipeline/preprocessing.py.

CRITICAL: Every filter, threshold, and parameter here MUST match the
training pipeline exactly.  If you change anything here, the model's
feature space will shift and predictions will be meaningless.

Two separate paths are provided:
    preprocess_mi()    — for 2 s motor-imagery windows
    preprocess_blink() — for 0.5 s blink-detection windows
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter

from config import (
    MI_BANDPASS, BLINK_BANDPASS, NOTCH_FREQ,
    CLIP_THRESHOLD_UV, FLAT_THRESHOLD_UV,
)


# ═══════════════════════════════════════════════════════════════════
# Shared low-level filters
# ═══════════════════════════════════════════════════════════════════

def _interpolate_nans(sig: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaN / Inf values in a 1-D signal."""
    arr = sig.copy().astype(np.float64)
    bad = np.isnan(arr) | np.isinf(arr)
    if not np.any(bad):
        return arr
    valid = np.where(~bad)[0]
    if len(valid) < 2:
        arr[bad] = 0.0
        return arr
    arr[bad] = np.interp(np.where(bad)[0], valid, arr[valid])
    return arr


def _interpolate_clipped(sig: np.ndarray, threshold: float = CLIP_THRESHOLD_UV) -> np.ndarray:
    """Replace ADC-clipped samples with linear interpolation."""
    arr = sig.copy()
    clipped = np.abs(arr) >= threshold
    if not np.any(clipped):
        return arr
    valid = np.where(~clipped)[0]
    if len(valid) < 2:
        return arr
    arr[clipped] = np.interp(np.where(clipped)[0], valid, arr[valid])
    return arr


def _bandpass(sig: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    lo = max(low / nyq, 0.001)
    hi = min(high / nyq, 0.999)
    b, a = sp_signal.butter(order, [lo, hi], btype="band")
    return sp_signal.filtfilt(b, a, sig)


def _notch(sig: np.ndarray, sfreq: float, freq: float = NOTCH_FREQ, quality: float = 30.0) -> np.ndarray:
    b, a = sp_signal.iirnotch(freq, quality, sfreq)
    return sp_signal.filtfilt(b, a, sig)


def _moving_artifact_removal(sig: np.ndarray, sfreq: float, threshold_std: float = 3.0) -> np.ndarray:
    """Statistical z-score artifact removal with interpolation."""
    win = int(sfreq * 0.5)
    if len(sig) < win:
        return sig
    rolling_mean = uniform_filter(sig, size=win, mode="constant")
    rolling_var = uniform_filter(sig ** 2, size=win, mode="constant") - rolling_mean ** 2
    rolling_std = np.sqrt(np.maximum(rolling_var, 0))
    z = np.abs(sig - rolling_mean) / (rolling_std + 1e-8)
    artifact = z > threshold_std
    if np.any(artifact):
        valid = np.where(~artifact)[0]
        if len(valid) > 1:
            return np.interp(np.arange(len(sig)), valid, sig[valid])
    return sig


# ═══════════════════════════════════════════════════════════════════
# Public preprocessing pipelines
# ═══════════════════════════════════════════════════════════════════

def preprocess_mi(window: np.ndarray, sfreq: float) -> np.ndarray:
    """Preprocess a motor-imagery window — must match training exactly.

    Pipeline (mirrors ml_pipeline/preprocessing.py::preprocess_mi_epoch):
        1. NaN / clip interpolation
        2. Linear detrend
        3. 60 Hz notch
        4. 1–40 Hz bandpass
        5. Moving z-score artifact removal (3σ)
        6. ASR is skipped in real-time (matches training fallback when
           asrpy is unavailable — if you trained with ASR installed,
           consider adding it here too)

    Args:
        window: (n_channels, n_samples)  e.g. (4, 512)
        sfreq:  sampling rate in Hz

    Returns:
        cleaned: same shape as input
    """
    n_ch, n_samp = window.shape
    out = np.zeros_like(window, dtype=np.float64)

    for ch in range(n_ch):
        sig = window[ch].astype(np.float64)
        sig = _interpolate_nans(sig)
        sig = _interpolate_clipped(sig)
        sig = sp_signal.detrend(sig)
        if n_samp > int(sfreq * 0.5):
            sig = _notch(sig, sfreq)
            sig = _bandpass(sig, sfreq, MI_BANDPASS[0], MI_BANDPASS[1])
        sig = _moving_artifact_removal(sig, sfreq, threshold_std=3.0)
        out[ch] = sig

    # NOTE: ASR is NOT applied here.  During training the fallback path
    # (asrpy not installed) also skipped ASR, so this is consistent.
    # If you retrain with ASR enabled, uncomment and implement:
    # out = _apply_asr(out, sfreq, cutoff=20)

    return out


def preprocess_blink(window: np.ndarray, sfreq: float) -> np.ndarray:
    """Preprocess a blink-detection window — must match training exactly.

    Pipeline (mirrors ml_pipeline/preprocessing.py::preprocess_blink_epoch):
        1. NaN interpolation (NO clip interpolation — blinks are large)
        2. Linear detrend
        3. 0.5–10 Hz bandpass
        NO ASR — would destroy the blink waveform

    Args:
        window: (n_channels, n_samples)  e.g. (4, 128)
        sfreq:  sampling rate in Hz

    Returns:
        cleaned: same shape as input
    """
    n_ch, n_samp = window.shape
    out = np.zeros_like(window, dtype=np.float64)

    for ch in range(n_ch):
        sig = window[ch].astype(np.float64)
        sig = _interpolate_nans(sig)
        sig = sp_signal.detrend(sig)
        if n_samp > int(sfreq * 0.25):
            try:
                sig = _bandpass(sig, sfreq, BLINK_BANDPASS[0], BLINK_BANDPASS[1])
            except Exception:
                pass
        out[ch] = sig

    return out
