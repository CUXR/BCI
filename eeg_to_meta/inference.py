"""Feature extraction + model prediction for real-time windows.

Mirrors the exact feature-extraction logic from:
    ml_pipeline/features.py  →  extract_mi_features()
    ml_pipeline/features.py  →  extract_blink_features()

CRITICAL: The number and ORDER of features must be identical to
training.  If you change anything here you must retrain.
"""

import logging
import time

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from config import (
    SFREQ, MI_BANDS, BANDS, N_CHANNELS,
    FRONTAL_INDICES, TEMPORAL_INDICES,
    BLINK_TOTAL_FEATURES,
    BLINK_CONFIDENCE_THRESHOLD, MI_CONFIDENCE_THRESHOLD, CLASS_THRESHOLDS,
    BLINK_COOLDOWN_S, MI_COOLDOWN_S,
    CLASS_NAMES,
)
from preprocess import preprocess_mi, preprocess_blink
from model_loader import ModelBundle

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Shared helpers  (duplicated from ml_pipeline/features.py)
# ═══════════════════════════════════════════════════════════════════

def _bandpower(sig, sfreq, fmin, fmax):
    nperseg = min(int(sfreq * 2), len(sig))
    if nperseg < int(sfreq * 0.25):
        return 0.0
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
    idx = (freqs >= fmin) & (freqs <= fmax)
    return float(np.trapezoid(psd[idx], freqs[idx])) if np.any(idx) else 0.0


def _relative_bandpower(sig, sfreq, fmin, fmax, total_range=(1, 40)):
    nperseg = min(int(sfreq * 2), len(sig))
    if nperseg < int(sfreq * 0.25):
        return 0.0
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
    band = (freqs >= fmin) & (freqs <= fmax)
    total = (freqs >= total_range[0]) & (freqs <= total_range[1])
    total_power = np.trapezoid(psd[total], freqs[total])
    if total_power <= 0:
        return 0.0
    return float(np.trapezoid(psd[band], freqs[band]) / total_power)


def _hjorth(sig):
    activity = np.var(sig)
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    var_d1 = np.var(d1)
    var_d2 = np.var(d2)
    mobility = np.sqrt(var_d1 / (activity + 1e-12))
    mob_d1 = np.sqrt(var_d2 / (var_d1 + 1e-12))
    complexity = mob_d1 / (mobility + 1e-12)
    return activity, mobility, complexity


def _zcr(sig):
    return float(np.mean(np.diff(np.sign(sig)) != 0))


# ═══════════════════════════════════════════════════════════════════
# MI feature extraction  (mirrors extract_mi_features)
# ═══════════════════════════════════════════════════════════════════

def _extract_mi_spectral_features(epoch_2d: np.ndarray, sfreq: float, ch_indices: list) -> np.ndarray:
    """78-dim spectral + time-domain feature vector for MI."""
    n_ch = epoch_2d.shape[0]
    features = []
    per_ch_powers = {band: [] for band in MI_BANDS}

    for ch_i in range(n_ch):
        sig = epoch_2d[ch_i]

        for band_name, (fmin, fmax) in MI_BANDS.items():
            abp = _bandpower(sig, sfreq, fmin, fmax)
            rbp = _relative_bandpower(sig, sfreq, fmin, fmax)
            features.extend([abp, rbp])
            per_ch_powers[band_name].append(abp)

        act, mob, comp = _hjorth(sig)
        features.extend([act, mob, comp])

        features.extend([_zcr(sig), float(kurtosis(sig)), float(skew(sig))])

        alpha_p = _bandpower(sig, sfreq, 8, 13)
        beta_p = _bandpower(sig, sfreq, 13, 30)
        features.append(alpha_p / (beta_p + 1e-12))

    # Lateralization asymmetry
    pairs = []
    if 0 in ch_indices and 3 in ch_indices:
        pairs.append((ch_indices.index(0), ch_indices.index(3)))
    if 1 in ch_indices and 2 in ch_indices:
        pairs.append((ch_indices.index(1), ch_indices.index(2)))

    for l_idx, r_idx in pairs:
        for band_name in MI_BANDS:
            lp = per_ch_powers[band_name][l_idx]
            rp = per_ch_powers[band_name][r_idx]
            denom = rp + lp
            features.append((rp - lp) / denom if denom > 0 else 0.0)

    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
# Blink feature extraction  (mirrors extract_blink_features)
# ═══════════════════════════════════════════════════════════════════

def _extract_blink_features(epoch_2d: np.ndarray, sfreq: float, ch_indices: list) -> np.ndarray:
    """75-dim feature vector for blink detection."""
    n_ch, n_samp = epoch_2d.shape
    features = []
    ch_abs_mean = []

    for ch_i in range(n_ch):
        sig = epoch_2d[ch_i]

        ptp = float(np.ptp(sig))
        rms = float(np.sqrt(np.mean(sig ** 2)))
        max_abs = float(np.max(np.abs(sig)))
        features.extend([ptp, rms, max_abs])
        ch_abs_mean.append(np.mean(np.abs(sig)))

        act, mob, comp = _hjorth(sig)
        features.extend([act, mob, comp])

        features.extend([
            float(skew(sig)),
            float(kurtosis(sig)),
            _zcr(sig),
            float(np.argmax(np.abs(sig)) / max(n_samp - 1, 1)),
        ])

        for _band_name, (fmin, fmax) in BANDS.items():
            features.append(_bandpower(sig, sfreq, fmin, fmax))

    # Cross-channel: frontal correlation (AF7–AF8)
    if 1 in ch_indices and 2 in ch_indices:
        i_af7, i_af8 = ch_indices.index(1), ch_indices.index(2)
        corr = np.corrcoef(epoch_2d[i_af7], epoch_2d[i_af8])[0, 1]
        features.append(float(corr) if np.isfinite(corr) else 0.0)
    else:
        features.append(0.0)

    # Temporal correlation (TP9–TP10)
    if 0 in ch_indices and 3 in ch_indices:
        i_tp9, i_tp10 = ch_indices.index(0), ch_indices.index(3)
        corr = np.corrcoef(epoch_2d[i_tp9], epoch_2d[i_tp10])[0, 1]
        features.append(float(corr) if np.isfinite(corr) else 0.0)
    else:
        features.append(0.0)

    # Frontal–temporal amplitude ratio
    if all(i in ch_indices for i in [0, 1, 2, 3]):
        frontal_amp = (ch_abs_mean[ch_indices.index(1)] + ch_abs_mean[ch_indices.index(2)]) / 2
        temporal_amp = (ch_abs_mean[ch_indices.index(0)] + ch_abs_mean[ch_indices.index(3)]) / 2
        features.append(frontal_amp / (temporal_amp + 1e-12))
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
# Inference engine
# ═══════════════════════════════════════════════════════════════════

class InferenceEngine:
    """Runs the full preprocess → features → predict cycle."""

    def __init__(self, model: ModelBundle):
        self.model = model
        self.ch_indices = list(range(N_CHANNELS))   # [0, 1, 2, 3]
        self._last_blink_time = 0.0
        self._last_mi_time = 0.0

    def classify(self, mi_window: np.ndarray | None, blink_window: np.ndarray | None):
        """Run full inference on the latest windows.

        Args:
            mi_window:    (4, 512)  or None if buffer not ready
            blink_window: (4, 128)  or None if buffer not ready

        Returns:
            dict with keys:
                label          : str   ("idle" / "intentional_blink" / …)
                internal_label : str   ("idle" / "blink" / "left" / "right")
                confidence     : float
                raw_probs      : dict  {class_name: float}
                latency_ms     : float (wall-clock time for this call)
        """
        t0 = time.perf_counter()
        now = time.time()

        result = {
            "label": CLASS_NAMES["idle"],
            "internal_label": "idle",
            "confidence": 0.0,
            "raw_probs": {},
            "latency_ms": 0.0,
        }

        # ── blink first (higher priority, shorter window) ──────────
        if blink_window is not None and (now - self._last_blink_time) >= BLINK_COOLDOWN_S:
            blink_clean = preprocess_blink(blink_window, SFREQ)
            blink_feat = _extract_blink_features(blink_clean, SFREQ, self.ch_indices)
            blink_feat = np.nan_to_num(blink_feat, nan=0.0, posinf=0.0, neginf=0.0)

            if blink_feat.shape[0] != BLINK_TOTAL_FEATURES:
                log.warning("Blink feature count %d != expected %d", blink_feat.shape[0], BLINK_TOTAL_FEATURES)
            else:
                try:
                    pred, probas = self.model.predict_blink(blink_feat)
                    blink_prob = float(probas[1]) if len(probas) > 1 else 0.0
                    result["raw_probs"]["blink"] = blink_prob
                    result["raw_probs"]["non_blink"] = 1.0 - blink_prob

                    blink_threshold = float(
                        CLASS_THRESHOLDS.get(CLASS_NAMES["blink"], BLINK_CONFIDENCE_THRESHOLD)
                    )
                    if blink_prob >= blink_threshold:
                        self._last_blink_time = now
                        result["label"] = CLASS_NAMES["blink"]
                        result["internal_label"] = "blink"
                        result["confidence"] = blink_prob
                        result["latency_ms"] = (time.perf_counter() - t0) * 1000
                        return result
                except Exception as exc:
                    log.debug("Blink prediction failed: %s", exc)

        # ── motor imagery ──────────────────────────────────────────
        if mi_window is not None and (now - self._last_mi_time) >= MI_COOLDOWN_S:
            mi_clean = preprocess_mi(mi_window, SFREQ)
            mi_feat = _extract_mi_spectral_features(mi_clean, SFREQ, self.ch_indices)
            mi_feat = np.nan_to_num(mi_feat, nan=0.0, posinf=0.0, neginf=0.0)

            csp_feat = self.model.apply_csp(mi_clean)
            if csp_feat.size > 0:
                full_feat = np.concatenate([mi_feat, csp_feat])
            else:
                full_feat = mi_feat
            full_feat = np.nan_to_num(full_feat, nan=0.0, posinf=0.0, neginf=0.0)

            mi_expected = self.model.mi_expected_features
            if full_feat.shape[0] != mi_expected:
                log.warning("MI feature count %d != expected %d", full_feat.shape[0], mi_expected)
            else:
                try:
                    pred, probas = self.model.predict_mi(full_feat)
                    label_names = self.model.mi_label_names()
                    for i, p in enumerate(probas):
                        key = label_names[i] if i < len(label_names) else f"class_{i}"
                        result["raw_probs"][key] = float(p)

                    confidence = float(probas[pred]) if len(probas) else 0.0
                    pred_label = (
                        label_names[pred]
                        if pred < len(label_names)
                        else CLASS_NAMES.get("idle", "idle")
                    )
                    threshold = float(CLASS_THRESHOLDS.get(pred_label, MI_CONFIDENCE_THRESHOLD))

                    if confidence >= threshold:
                        self._last_mi_time = now
                        result["label"] = pred_label
                        result["internal_label"] = pred_label
                    result["confidence"] = confidence
                except Exception as exc:
                    log.debug("MI prediction failed: %s", exc)

        result["latency_ms"] = (time.perf_counter() - t0) * 1000
        return result
