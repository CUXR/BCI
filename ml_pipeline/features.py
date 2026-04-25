"""Feature extraction for motor imagery and blink classification.

Motor imagery: band powers, CSP, Hjorth parameters, lateralization asymmetry
Blink detection: amplitude, waveform shape, cross-channel correlation
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from scipy.linalg import eigh

from config import (
    TARGET_SFREQ, CH_NAMES, MI_BANDS, BANDS,
    FRONTAL_INDICES, TEMPORAL_INDICES,
)


# ═══════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════

def _bandpower(sig, sfreq, fmin, fmax):
    """Absolute band power via Welch PSD."""
    nperseg = min(int(sfreq * 2), len(sig))
    if nperseg < int(sfreq * 0.25):
        return 0.0
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
    idx = (freqs >= fmin) & (freqs <= fmax)
    return float(np.trapz(psd[idx], freqs[idx])) if np.any(idx) else 0.0


def _relative_bandpower(sig, sfreq, fmin, fmax, total_range=(1, 40)):
    """Band power relative to total power in total_range."""
    nperseg = min(int(sfreq * 2), len(sig))
    if nperseg < int(sfreq * 0.25):
        return 0.0
    freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)
    band = (freqs >= fmin) & (freqs <= fmax)
    total = (freqs >= total_range[0]) & (freqs <= total_range[1])
    total_power = np.trapz(psd[total], freqs[total])
    if total_power <= 0:
        return 0.0
    return float(np.trapz(psd[band], freqs[band]) / total_power)


def _hjorth(sig):
    """Hjorth activity, mobility, complexity."""
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
    """Zero-crossing rate."""
    return float(np.mean(np.diff(np.sign(sig)) != 0))


# ═══════════════════════════════════════════════════════════════════
# Motor imagery features
# ═══════════════════════════════════════════════════════════════════

def extract_mi_features(epoch_2d, sfreq=None, ch_indices=None):
    """Extract feature vector for a single MI epoch.

    Args:
        epoch_2d: (n_channels, n_samples)
        sfreq: sampling rate (default TARGET_SFREQ)
        ch_indices: original channel indices used

    Returns:
        features: 1-D numpy array
    """
    sfreq = sfreq or TARGET_SFREQ
    n_ch, n_samp = epoch_2d.shape
    if ch_indices is None:
        ch_indices = list(range(n_ch))

    features = []
    per_ch_powers = {band: [] for band in MI_BANDS}

    for ch_i in range(n_ch):
        sig = epoch_2d[ch_i]

        # Band powers (absolute + relative)
        for band_name, (fmin, fmax) in MI_BANDS.items():
            abp = _bandpower(sig, sfreq, fmin, fmax)
            rbp = _relative_bandpower(sig, sfreq, fmin, fmax)
            features.extend([abp, rbp])
            per_ch_powers[band_name].append(abp)

        # Hjorth
        act, mob, comp = _hjorth(sig)
        features.extend([act, mob, comp])

        # Statistical
        features.extend([_zcr(sig), float(kurtosis(sig)), float(skew(sig))])

        # Alpha/beta ratio
        alpha_p = _bandpower(sig, sfreq, 8, 13)
        beta_p = _bandpower(sig, sfreq, 13, 30)
        features.append(alpha_p / (beta_p + 1e-12))

    # Lateralization asymmetry: (Right - Left) / (Right + Left) per band
    # TP9(idx 0) vs TP10(idx 3), AF7(idx 1) vs AF8(idx 2)
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


def extract_mi_features_batch(epochs, sfreq=None, ch_indices=None):
    """Extract MI features for a list of epochs. Returns (n_epochs, n_features)."""
    if not epochs:
        return np.empty((0, 0))
    X = np.array([extract_mi_features(ep, sfreq, ch_indices) for ep in epochs])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ═══════════════════════════════════════════════════════════════════
# Common Spatial Patterns (CSP)
# ═══════════════════════════════════════════════════════════════════

def compute_csp(epochs, labels, n_components=None):
    """Compute CSP filters and log-variance features.

    Args:
        epochs: list of (n_channels, n_samples) arrays
        labels: list of 0/1 labels
        n_components: CSP components (default: n_channels)

    Returns:
        csp_features: (n_epochs, n_components) log-variance features
        W: (n_channels, n_components) spatial filter matrix
    """
    labels = np.array(labels)
    n_ch = epochs[0].shape[0]
    n_components = n_components or n_ch

    covs = {0: [], 1: []}
    for ep, lab in zip(epochs, labels):
        C = np.cov(ep)
        if C.ndim == 0:
            C = np.array([[C]])
        trace = np.trace(C)
        if trace > 0:
            C /= trace
        covs[lab].append(C)

    if not covs[0] or not covs[1]:
        return np.zeros((len(epochs), 1)), np.eye(n_ch)[:, :1]

    C0 = np.mean(covs[0], axis=0)
    C1 = np.mean(covs[1], axis=0)
    C_sum = C0 + C1 + np.eye(n_ch) * 1e-6

    try:
        eigenvalues, eigenvectors = eigh(C0, C_sum)
    except np.linalg.LinAlgError:
        return np.zeros((len(epochs), 1)), np.eye(n_ch)[:, :1]

    sorted_idx = np.argsort(eigenvalues)
    n_half = n_components // 2
    n_other = n_components - n_half
    selected = np.concatenate([sorted_idx[:n_half], sorted_idx[-n_other:]])
    W = eigenvectors[:, selected]

    csp_features = np.zeros((len(epochs), len(selected)))
    for i, ep in enumerate(epochs):
        projected = W.T @ ep  # (n_components, n_samples)
        csp_features[i] = np.log(np.var(projected, axis=1) + 1e-12)

    return csp_features, W


def apply_csp(epochs, W):
    """Apply pre-computed CSP spatial filters."""
    features = np.zeros((len(epochs), W.shape[1]))
    for i, ep in enumerate(epochs):
        projected = W.T @ ep
        features[i] = np.log(np.var(projected, axis=1) + 1e-12)
    return features


# ═══════════════════════════════════════════════════════════════════
# Blink features
# ═══════════════════════════════════════════════════════════════════

def extract_blink_features(epoch_2d, sfreq=None, ch_indices=None):
    """Extract feature vector for a single blink detection epoch.

    Args:
        epoch_2d: (n_channels, n_samples)

    Returns:
        features: 1-D numpy array
    """
    sfreq = sfreq or TARGET_SFREQ
    n_ch, n_samp = epoch_2d.shape
    if ch_indices is None:
        ch_indices = list(range(n_ch))

    features = []
    ch_rms = []
    ch_abs_mean = []

    for ch_i in range(n_ch):
        sig = epoch_2d[ch_i]

        ptp = float(np.ptp(sig))
        rms = float(np.sqrt(np.mean(sig ** 2)))
        max_abs = float(np.max(np.abs(sig)))
        features.extend([ptp, rms, max_abs])
        ch_rms.append(rms)
        ch_abs_mean.append(np.mean(np.abs(sig)))

        act, mob, comp = _hjorth(sig)
        features.extend([act, mob, comp])

        features.extend([
            float(skew(sig)),
            float(kurtosis(sig)),
            _zcr(sig),
            float(np.argmax(np.abs(sig)) / max(n_samp - 1, 1)),
        ])

        for band_name, (fmin, fmax) in BANDS.items():
            features.append(_bandpower(sig, sfreq, fmin, fmax))

    # Cross-channel: frontal correlation (AF7-AF8)
    if 1 in ch_indices and 2 in ch_indices:
        i_af7, i_af8 = ch_indices.index(1), ch_indices.index(2)
        corr = np.corrcoef(epoch_2d[i_af7], epoch_2d[i_af8])[0, 1]
        features.append(float(corr) if np.isfinite(corr) else 0.0)
    else:
        features.append(0.0)

    # Temporal correlation (TP9-TP10)
    if 0 in ch_indices and 3 in ch_indices:
        i_tp9, i_tp10 = ch_indices.index(0), ch_indices.index(3)
        corr = np.corrcoef(epoch_2d[i_tp9], epoch_2d[i_tp10])[0, 1]
        features.append(float(corr) if np.isfinite(corr) else 0.0)
    else:
        features.append(0.0)

    # Frontal-temporal amplitude ratio
    if all(i in ch_indices for i in [0, 1, 2, 3]):
        frontal_amp = (ch_abs_mean[ch_indices.index(1)] + ch_abs_mean[ch_indices.index(2)]) / 2
        temporal_amp = (ch_abs_mean[ch_indices.index(0)] + ch_abs_mean[ch_indices.index(3)]) / 2
        features.append(frontal_amp / (temporal_amp + 1e-12))
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float64)


def extract_blink_features_batch(epochs, sfreq=None, ch_indices=None):
    """Extract blink features for a list of epochs."""
    if not epochs:
        return np.empty((0, 0))
    X = np.array([extract_blink_features(ep, sfreq, ch_indices) for ep in epochs])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
