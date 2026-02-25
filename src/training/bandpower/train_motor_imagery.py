"""Motor Imagery Classification Pipeline.

Trains and evaluates L/R motor imagery classifiers using EEG data from
Muse 2 headband (4 channels: TP9, AF7, AF8, TP10 @ 256 Hz).

Compares 3 channel configurations:
  (a) TP9+TP10 only (cleanest temporal channels)
  (b) All 4 channels with aggressive artifact rejection
  (c) Auto-selected by per-channel SNR scoring

Evaluates 4 classifiers with within-subject 5-fold CV and Leave-One-Subject-Out CV.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_filters import EEGFilter

# ---------- constants (match analyze_all.py) ----------
DATA_DIR = PROJECT_ROOT / "data"
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
CH_COLS = ["EEG_0", "EEG_1", "EEG_2", "EEG_3"]
MI_TYPES = ["motor_imagery_left", "motor_imagery_right"]

BANDS = {
    "mu": (8, 12),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
}

# Channel configs to compare
CHANNEL_CONFIGS = {
    "temporal_only": [0, 3],          # TP9, TP10
    "all_4ch": [0, 1, 2, 3],          # all channels
    # "auto" is computed per-subject based on quality scores
}

MIN_EPOCH_SECONDS = 2.0
MAX_EPOCH_SECONDS = 5.0
ARTIFACT_THRESHOLD_UV = 200.0


# ============================================================
# Data loading (reuses patterns from analyze_all.py)
# ============================================================

def find_subject_data(data_dir):
    """Discover all subject directories with EEG data."""
    subjects = {}
    for sub_dir in sorted(data_dir.glob("sub*")):
        eeg_files = list(sub_dir.glob("eeg_data_*.csv"))
        trial_files = list(sub_dir.glob("trial_log_*.csv"))
        if eeg_files and trial_files:
            meta_file = sub_dir / "metadata.yaml"
            name = sub_dir.name
            if meta_file.exists():
                for line in meta_file.read_text().splitlines():
                    if line.startswith("participant:"):
                        name = f"{sub_dir.name} ({line.split(':')[1].strip()})"
            subjects[sub_dir.name] = {
                "dir": sub_dir,
                "eeg": eeg_files[0],
                "trials": trial_files[0],
                "label": name,
            }
    return subjects


def load_subject(info):
    """Load EEG and trial data for a subject."""
    eeg = pd.read_csv(info["eeg"])
    trials = pd.read_csv(info["trials"])
    dt = np.diff(eeg["timestamp"].values[:1000])
    fs = 1.0 / np.median(dt)
    return eeg, trials, fs


# ============================================================
# Channel quality assessment
# ============================================================

def compute_channel_quality(eeg, trials, fs, ch_indices=None):
    """Compute per-channel quality metrics across motor imagery trials.

    Returns a dict of {ch_name: {metric: value, ..., composite: float}}.
    Higher composite = better quality.
    """
    if ch_indices is None:
        ch_indices = list(range(4))

    mi_trials = trials[trials["trial_type"].isin(MI_TYPES)]
    quality = {}

    for ch_idx in ch_indices:
        col = CH_COLS[ch_idx]
        name = CH_NAMES[ch_idx]

        # Collect all MI segments for this channel
        segments = []
        for _, trial in mi_trials.iterrows():
            mask = (eeg["timestamp"] >= trial["start_time"]) & (
                eeg["timestamp"] <= trial["end_time"])
            seg = eeg.loc[mask, col].values
            if len(seg) > int(fs):
                segments.append(seg)

        if not segments:
            quality[name] = {"rms": np.nan, "kurtosis": np.nan,
                             "spectral_flatness": np.nan, "mu_snr": np.nan,
                             "artifact_rate": np.nan, "composite": 0.0}
            continue

        all_data = np.concatenate(segments)

        # RMS amplitude (lower = cleaner for EEG)
        rms = np.sqrt(np.mean(all_data ** 2))

        # Kurtosis (high = heavy tails / artifacts)
        kurt = kurtosis(all_data)

        # Spectral flatness (1 = white noise, 0 = tonal; lower = more structured)
        psds = []
        for seg in segments:
            nperseg = min(int(fs * 2), len(seg))
            if nperseg >= int(fs * 0.5):
                freqs, psd = welch(seg, fs=fs, nperseg=nperseg)
                # Focus on 1-40 Hz range
                band_mask = (freqs >= 1) & (freqs <= 40)
                if np.any(band_mask):
                    psds.append(psd[band_mask])
        if psds:
            min_len = min(len(p) for p in psds)
            mean_psd = np.mean([p[:min_len] for p in psds], axis=0)
            # Spectral flatness = geometric mean / arithmetic mean
            log_psd = np.log(mean_psd + 1e-12)
            spectral_flatness = np.exp(np.mean(log_psd)) / (np.mean(mean_psd) + 1e-12)
        else:
            spectral_flatness = 1.0

        # Mu-band (8-12 Hz) SNR: mu power / total power in 1-40 Hz
        mu_snrs = []
        for seg in segments:
            nperseg = min(int(fs * 2), len(seg))
            if nperseg >= int(fs * 0.5):
                freqs, psd = welch(seg, fs=fs, nperseg=nperseg)
                mu_mask = (freqs >= 8) & (freqs <= 12)
                total_mask = (freqs >= 1) & (freqs <= 40)
                total_power = np.trapz(psd[total_mask], freqs[total_mask])
                if total_power > 0:
                    mu_power = np.trapz(psd[mu_mask], freqs[mu_mask])
                    mu_snrs.append(mu_power / total_power)
        mu_snr = np.mean(mu_snrs) if mu_snrs else 0.0

        # Artifact rate: fraction of samples exceeding threshold
        artifact_rate = np.mean(np.abs(all_data) > ARTIFACT_THRESHOLD_UV)

        # Composite score (higher = better)
        # Penalize: high RMS, high kurtosis, high spectral flatness, high artifact rate
        # Reward: high mu SNR
        rms_score = np.clip(1.0 - rms / 500.0, 0, 1)
        kurt_score = np.clip(1.0 - abs(kurt) / 20.0, 0, 1)
        flat_score = np.clip(1.0 - spectral_flatness, 0, 1)
        artifact_score = 1.0 - artifact_rate
        mu_score = np.clip(mu_snr * 5.0, 0, 1)  # Boost mu contribution

        composite = (rms_score * 0.25 + kurt_score * 0.15 + flat_score * 0.2 +
                     artifact_score * 0.2 + mu_score * 0.2)

        quality[name] = {
            "rms": rms,
            "kurtosis": kurt,
            "spectral_flatness": spectral_flatness,
            "mu_snr": mu_snr,
            "artifact_rate": artifact_rate,
            "composite": composite,
        }

    return quality


def select_channels_by_quality(quality, min_channels=2):
    """Select channels with above-median composite quality score."""
    scores = {ch: q["composite"] for ch, q in quality.items()
              if not np.isnan(q["composite"])}
    if len(scores) <= min_channels:
        return list(range(4))

    median_score = np.median(list(scores.values()))
    selected = [i for i, ch in enumerate(CH_NAMES)
                if ch in scores and scores[ch] >= median_score]

    # Ensure minimum channel count
    if len(selected) < min_channels:
        sorted_chs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [CH_NAMES.index(ch) for ch, _ in sorted_chs[:min_channels]]

    return sorted(selected)


# ============================================================
# Epoch extraction
# ============================================================

def extract_mi_epochs(eeg, trials, fs, ch_indices, eeg_filter):
    """Extract motor imagery epochs using trial_log timestamps.

    Returns:
        epochs: list of 2D arrays (n_samples, n_channels)
        labels: list of int (0=left, 1=right)
        trial_ids: list of trial_id strings
    """
    mi_trials = trials[trials["trial_type"].isin(MI_TYPES)]
    epochs = []
    labels = []
    trial_ids = []

    min_samples = int(MIN_EPOCH_SECONDS * fs)
    max_samples = int(MAX_EPOCH_SECONDS * fs)

    for _, trial in mi_trials.iterrows():
        mask = (eeg["timestamp"] >= trial["start_time"]) & (
            eeg["timestamp"] <= trial["end_time"])
        segment = eeg.loc[mask, [CH_COLS[i] for i in ch_indices]].values

        # Discard trials shorter than minimum
        if len(segment) < min_samples:
            continue

        # Truncate to maximum
        if len(segment) > max_samples:
            segment = segment[:max_samples]

        # Apply preprocessing per channel
        processed = np.zeros_like(segment, dtype=np.float64)
        n_clean_channels = 0
        for col_i in range(segment.shape[1]):
            ch_data = segment[:, col_i].astype(np.float64)

            # Apply artifact removal (interpolates bad samples) then filter
            ch_data = eeg_filter.apply_moving_artifact_removal(ch_data, threshold=3.0)
            ch_data = eeg_filter.apply_comprehensive_filter(ch_data)
            processed[:, col_i] = ch_data

            # Track how many channels are reasonably clean after filtering
            if np.std(ch_data) < 500:
                n_clean_channels += 1

        # Reject epoch only if fewer than 2 channels are usable
        if n_clean_channels < min(2, segment.shape[1]):
            continue

        label = 0 if trial["trial_type"] == "motor_imagery_left" else 1
        epochs.append(processed)
        labels.append(label)
        trial_ids.append(trial["trial_id"])

    return epochs, labels, trial_ids


# ============================================================
# Feature extraction
# ============================================================

def _bandpower(signal, fs, fmin, fmax):
    """Compute absolute band power using Welch's method."""
    nperseg = min(int(fs * 2), len(signal))
    if nperseg < int(fs * 0.5):
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return 0.0
    return np.trapz(psd[idx], freqs[idx])


def _relative_bandpower(signal, fs, fmin, fmax):
    """Compute relative band power (band / total 1-40 Hz)."""
    nperseg = min(int(fs * 2), len(signal))
    if nperseg < int(fs * 0.5):
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    band_idx = (freqs >= fmin) & (freqs <= fmax)
    total_idx = (freqs >= 1) & (freqs <= 40)
    total_power = np.trapz(psd[total_idx], freqs[total_idx])
    if total_power == 0:
        return 0.0
    return np.trapz(psd[band_idx], freqs[band_idx]) / total_power


def _hjorth_parameters(signal):
    """Compute Hjorth activity, mobility, complexity."""
    activity = np.var(signal)
    d1 = np.diff(signal)
    d2 = np.diff(d1)
    var_d1 = np.var(d1)
    var_d2 = np.var(d2)

    mobility = np.sqrt(var_d1 / (activity + 1e-12))
    mob_d1 = np.sqrt(var_d2 / (var_d1 + 1e-12))
    complexity = mob_d1 / (mobility + 1e-12)

    return activity, mobility, complexity


def _zero_crossing_rate(signal):
    """Fraction of consecutive samples that cross zero."""
    return np.mean(np.diff(np.sign(signal)) != 0)


def extract_features_single_epoch(epoch, fs, ch_indices):
    """Extract features from a single epoch (n_samples, n_channels).

    Returns a 1D feature vector.
    """
    n_ch = epoch.shape[1]
    features = []
    ch_names_used = [CH_NAMES[i] for i in ch_indices]

    # Per-channel features
    per_ch_powers = {}  # {band_name: [power_ch0, power_ch1, ...]}
    for band_name in BANDS:
        per_ch_powers[band_name] = []

    for ch_i in range(n_ch):
        sig = epoch[:, ch_i]

        # Band powers (absolute + relative)
        for band_name, (fmin, fmax) in BANDS.items():
            abs_bp = _bandpower(sig, fs, fmin, fmax)
            rel_bp = _relative_bandpower(sig, fs, fmin, fmax)
            features.extend([abs_bp, rel_bp])
            per_ch_powers[band_name].append(abs_bp)

        # Time-domain: Hjorth parameters
        activity, mobility, complexity = _hjorth_parameters(sig)
        features.extend([activity, mobility, complexity])

        # Zero-crossing rate, kurtosis, skewness
        zcr = _zero_crossing_rate(sig)
        kurt = kurtosis(sig)
        sk = skew(sig)
        features.extend([zcr, kurt, sk])

        # Alpha/beta ratio
        alpha_power = _bandpower(sig, fs, 8, 13)
        beta_power = _bandpower(sig, fs, 13, 30)
        ab_ratio = alpha_power / (beta_power + 1e-12)
        features.append(ab_ratio)

    # Lateralization asymmetry indices
    # For electrode pairs: TP9(left)↔TP10(right), AF7(left)↔AF8(right)
    # Asymmetry = (R - L) / (R + L) for each band
    pairs = []
    if 0 in ch_indices and 3 in ch_indices:
        # TP9 (index in ch_indices) ↔ TP10
        l_idx = ch_indices.index(0)
        r_idx = ch_indices.index(3)
        pairs.append((l_idx, r_idx, "TP9_TP10"))
    if 1 in ch_indices and 2 in ch_indices:
        # AF7 ↔ AF8
        l_idx = ch_indices.index(1)
        r_idx = ch_indices.index(2)
        pairs.append((l_idx, r_idx, "AF7_AF8"))

    for l_idx, r_idx, pair_name in pairs:
        for band_name in BANDS:
            l_power = per_ch_powers[band_name][l_idx]
            r_power = per_ch_powers[band_name][r_idx]
            denom = r_power + l_power
            if denom > 0:
                asym = (r_power - l_power) / denom
            else:
                asym = 0.0
            features.append(asym)

    return np.array(features, dtype=np.float64)


def extract_features_all(epochs, fs, ch_indices):
    """Extract features from all epochs. Returns (X, feature_names)."""
    if not epochs:
        return np.empty((0, 0)), []

    X = np.array([extract_features_single_epoch(ep, fs, ch_indices)
                  for ep in epochs])

    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X


# ============================================================
# Common Spatial Patterns (CSP)
# ============================================================

def compute_csp(epochs, labels, n_components=None):
    """Compute CSP spatial filters and project epochs.

    Args:
        epochs: list of (n_samples, n_channels) arrays
        labels: list of 0/1
        n_components: number of CSP components (default: n_channels - 1)

    Returns:
        csp_features: (n_epochs, n_components) log-variance features
        W: spatial filter matrix for applying to new data
    """
    labels = np.array(labels)
    n_ch = epochs[0].shape[1]

    if n_components is None:
        n_components = min(n_ch, max(1, n_ch - 1))

    # Compute class-conditional covariance matrices
    covs = {0: [], 1: []}
    for ep, lab in zip(epochs, labels):
        # Normalize to unit trace
        C = np.cov(ep.T)
        if C.ndim == 0:
            C = np.array([[C]])
        C /= np.trace(C) + 1e-12
        covs[lab].append(C)

    if not covs[0] or not covs[1]:
        return np.zeros((len(epochs), 1)), np.eye(n_ch)[:, :1]

    C0 = np.mean(covs[0], axis=0)
    C1 = np.mean(covs[1], axis=0)

    # Solve generalized eigenvalue problem
    C_composite = C0 + C1
    # Regularize to avoid singular matrix
    C_composite += np.eye(n_ch) * 1e-6

    try:
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(C0, C_composite)
    except np.linalg.LinAlgError:
        return np.zeros((len(epochs), 1)), np.eye(n_ch)[:, :1]

    # Sort by eigenvalue (most discriminative at edges)
    sorted_idx = np.argsort(eigenvalues)
    # Take first and last components (most discriminative for each class)
    if n_components >= n_ch:
        selected_idx = sorted_idx
    else:
        n_half = n_components // 2
        n_other = n_components - n_half
        selected_idx = np.concatenate([sorted_idx[:n_half], sorted_idx[-n_other:]])

    W = eigenvectors[:, selected_idx]

    # Project and compute log-variance features
    csp_features = np.zeros((len(epochs), len(selected_idx)))
    for i, ep in enumerate(epochs):
        projected = ep @ W  # (n_samples, n_components)
        variances = np.var(projected, axis=0)
        csp_features[i] = np.log(variances + 1e-12)

    return csp_features, W


def apply_csp(epochs, W):
    """Apply pre-computed CSP filters to epochs."""
    csp_features = np.zeros((len(epochs), W.shape[1]))
    for i, ep in enumerate(epochs):
        projected = ep @ W
        variances = np.var(projected, axis=0)
        csp_features[i] = np.log(variances + 1e-12)
    return csp_features


# ============================================================
# Classifiers
# ============================================================

def get_classifiers():
    """Return dict of classifier pipelines."""
    return {
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "SVM_linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0)),
        ]),
        "SVM_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]),
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=3,
                random_state=42)),
        ]),
    }


# ============================================================
# Evaluation strategies
# ============================================================

def evaluate_within_subject(X, y, subject_ids, classifiers):
    """5-fold stratified CV per subject.

    Returns dict: {subject: {clf_name: accuracy}}.
    """
    unique_subs = sorted(set(subject_ids))
    results = {}

    for sub in unique_subs:
        mask = np.array([s == sub for s in subject_ids])
        X_sub = X[mask]
        y_sub = y[mask]

        if len(np.unique(y_sub)) < 2 or len(y_sub) < 5:
            results[sub] = {name: np.nan for name in classifiers}
            continue

        n_folds = min(5, min(np.bincount(y_sub)))
        if n_folds < 2:
            results[sub] = {name: np.nan for name in classifiers}
            continue

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        results[sub] = {}

        for clf_name, clf in classifiers.items():
            try:
                y_pred = cross_val_predict(clf, X_sub, y_sub, cv=cv)
                acc = accuracy_score(y_sub, y_pred)
            except Exception:
                acc = np.nan
            results[sub][clf_name] = acc

    return results


def evaluate_loso(X, y, subject_ids, classifiers):
    """Leave-One-Subject-Out cross-validation.

    Returns dict: {test_subject: {clf_name: accuracy}}.
    """
    unique_subs = sorted(set(subject_ids))
    results = {}

    for test_sub in unique_subs:
        test_mask = np.array([s == test_sub for s in subject_ids])
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            results[test_sub] = {name: np.nan for name in classifiers}
            continue

        results[test_sub] = {}
        for clf_name, clf in classifiers.items():
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
            except Exception:
                acc = np.nan
            results[test_sub][clf_name] = acc

    return results


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline():
    """Run the full classification pipeline."""
    print("=" * 70)
    print("Motor Imagery Classification Pipeline")
    print("=" * 70)

    # --- Load all subjects ---
    subjects = find_subject_data(DATA_DIR)
    if not subjects:
        print("ERROR: No subject data found in", DATA_DIR)
        return

    print(f"\nFound {len(subjects)} subjects: {list(subjects.keys())}")

    subject_data = {}
    for sub_id, info in subjects.items():
        eeg, trials, fs = load_subject(info)
        n_mi = len(trials[trials["trial_type"].isin(MI_TYPES)])
        print(f"  {info['label']}: {n_mi} MI trials, Fs={fs:.0f} Hz")
        subject_data[sub_id] = (eeg, trials, fs, info)

    fs = list(subject_data.values())[0][2]  # Use first subject's fs

    # --- Channel quality assessment ---
    print(f"\n{'=' * 70}")
    print("Channel Quality Assessment")
    print("=" * 70)

    all_quality = {}
    for sub_id, (eeg, trials, fs_sub, info) in subject_data.items():
        quality = compute_channel_quality(eeg, trials, fs_sub)
        all_quality[sub_id] = quality

    # Print quality table
    print(f"\n{'Subject':<10}", end="")
    for ch in CH_NAMES:
        print(f"  {ch:>12}", end="")
    print()
    print("-" * 62)

    for sub_id in sorted(all_quality.keys()):
        q = all_quality[sub_id]
        print(f"{sub_id:<10}", end="")
        for ch in CH_NAMES:
            print(f"  {q[ch]['composite']:>12.3f}", end="")
        print()

    # Detail rows
    for metric in ["rms", "kurtosis", "spectral_flatness", "mu_snr", "artifact_rate"]:
        print(f"\n  {metric}:")
        for sub_id in sorted(all_quality.keys()):
            q = all_quality[sub_id]
            print(f"    {sub_id:<8}", end="")
            for ch in CH_NAMES:
                val = q[ch][metric]
                if metric == "rms":
                    print(f"  {val:>10.1f}", end="")
                else:
                    print(f"  {val:>10.4f}", end="")
            print()

    # --- Build channel configs including auto ---
    configs = dict(CHANNEL_CONFIGS)
    # Auto config: per-subject, pick channels above median quality
    # For pipeline simplicity, use the union of all subjects' selected channels
    auto_channels = set()
    for sub_id, quality in all_quality.items():
        selected = select_channels_by_quality(quality)
        auto_channels.update(selected)
    configs["auto"] = sorted(auto_channels)

    print(f"\nChannel configurations:")
    for config_name, ch_idx in configs.items():
        ch_names = [CH_NAMES[i] for i in ch_idx]
        print(f"  {config_name}: {ch_names}")

    # --- Run classification for each channel config ---
    all_results = {}

    for config_name, ch_indices in configs.items():
        print(f"\n{'=' * 70}")
        print(f"Config: {config_name} — channels: {[CH_NAMES[i] for i in ch_indices]}")
        print("=" * 70)

        eeg_filter = EEGFilter(sampling_rate=int(fs), notch_freq=60.0)

        # Extract epochs and features per subject
        all_epochs = []
        all_labels = []
        all_sub_ids = []

        for sub_id, (eeg, trials, fs_sub, info) in subject_data.items():
            epochs, labels, trial_ids = extract_mi_epochs(
                eeg, trials, fs_sub, ch_indices, eeg_filter)
            print(f"  {sub_id}: {len(epochs)} valid epochs "
                  f"(L={labels.count(0)}, R={labels.count(1)})")

            all_epochs.extend(epochs)
            all_labels.extend(labels)
            all_sub_ids.extend([sub_id] * len(epochs))

        if not all_epochs:
            print("  No valid epochs — skipping config")
            continue

        # Pad epochs to same length for CSP
        max_len = max(ep.shape[0] for ep in all_epochs)
        padded_epochs = []
        for ep in all_epochs:
            if ep.shape[0] < max_len:
                pad_width = ((0, max_len - ep.shape[0]), (0, 0))
                ep = np.pad(ep, pad_width, mode="edge")
            padded_epochs.append(ep)

        y = np.array(all_labels)
        sub_ids_arr = all_sub_ids

        # Extract spectral/time-domain features
        X_feat = extract_features_all(padded_epochs, fs, ch_indices)

        # Compute CSP features
        csp_features, W_csp = compute_csp(padded_epochs, all_labels)

        # Combine: spectral features + CSP features
        X = np.hstack([X_feat, csp_features])

        print(f"\n  Feature matrix: {X.shape[0]} epochs × {X.shape[1]} features")
        print(f"    Spectral/time features: {X_feat.shape[1]}")
        print(f"    CSP features: {csp_features.shape[1]}")
        print(f"    Class balance: L={np.sum(y == 0)}, R={np.sum(y == 1)}")

        classifiers = get_classifiers()

        # Within-subject CV
        print(f"\n  --- Within-Subject 5-Fold CV ---")
        ws_results = evaluate_within_subject(X, y, sub_ids_arr, classifiers)

        for sub_id in sorted(ws_results.keys()):
            accs = ws_results[sub_id]
            parts = [f"{name}: {acc:.1%}" if not np.isnan(acc) else f"{name}: N/A"
                     for name, acc in accs.items()]
            print(f"    {sub_id}: {', '.join(parts)}")

        # Mean across subjects
        ws_means = {}
        for clf_name in classifiers:
            vals = [ws_results[s][clf_name] for s in ws_results
                    if not np.isnan(ws_results[s][clf_name])]
            ws_means[clf_name] = np.mean(vals) if vals else np.nan

        print(f"    {'Mean':<6}: ", end="")
        print(", ".join(f"{name}: {acc:.1%}" if not np.isnan(acc) else f"{name}: N/A"
                        for name, acc in ws_means.items()))

        # LOSO CV
        print(f"\n  --- Leave-One-Subject-Out CV ---")
        loso_results = evaluate_loso(X, y, sub_ids_arr, classifiers)

        for sub_id in sorted(loso_results.keys()):
            accs = loso_results[sub_id]
            parts = [f"{name}: {acc:.1%}" if not np.isnan(acc) else f"{name}: N/A"
                     for name, acc in accs.items()]
            print(f"    {sub_id} (test): {', '.join(parts)}")

        loso_means = {}
        for clf_name in classifiers:
            vals = [loso_results[s][clf_name] for s in loso_results
                    if not np.isnan(loso_results[s][clf_name])]
            loso_means[clf_name] = np.mean(vals) if vals else np.nan

        print(f"    {'Mean':<6}: ", end="")
        print(", ".join(f"{name}: {acc:.1%}" if not np.isnan(acc) else f"{name}: N/A"
                        for name, acc in loso_means.items()))

        all_results[config_name] = {
            "channels": [CH_NAMES[i] for i in ch_indices],
            "n_epochs": len(all_epochs),
            "n_features": X.shape[1],
            "within_subject": {
                sub: {clf: float(f"{acc:.4f}") if not np.isnan(acc) else None
                      for clf, acc in accs.items()}
                for sub, accs in ws_results.items()
            },
            "within_subject_mean": {
                clf: float(f"{acc:.4f}") if not np.isnan(acc) else None
                for clf, acc in ws_means.items()
            },
            "loso": {
                sub: {clf: float(f"{acc:.4f}") if not np.isnan(acc) else None
                      for clf, acc in accs.items()}
                for sub, accs in loso_results.items()
            },
            "loso_mean": {
                clf: float(f"{acc:.4f}") if not np.isnan(acc) else None
                for clf, acc in loso_means.items()
            },
        }

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY — Best Configurations")
    print("=" * 70)

    # Within-subject best
    best_ws_acc = 0
    best_ws = ("", "")
    for config, res in all_results.items():
        for clf, acc in res["within_subject_mean"].items():
            if acc is not None and acc > best_ws_acc:
                best_ws_acc = acc
                best_ws = (config, clf)

    print(f"\n  Best within-subject: {best_ws[1]} with {best_ws[0]} "
          f"channels — {best_ws_acc:.1%}")

    # LOSO best
    best_loso_acc = 0
    best_loso = ("", "")
    for config, res in all_results.items():
        for clf, acc in res["loso_mean"].items():
            if acc is not None and acc > best_loso_acc:
                best_loso_acc = acc
                best_loso = (config, clf)

    print(f"  Best LOSO:           {best_loso[1]} with {best_loso[0]} "
          f"channels — {best_loso_acc:.1%}")

    # Save results
    results_file = DATA_DIR / "classification_results.json"

    output = {
        "channel_quality": {
            sub: {ch: {k: float(f"{v:.6f}") if isinstance(v, float) else v
                       for k, v in metrics.items()}
                  for ch, metrics in quality.items()}
            for sub, quality in all_quality.items()
        },
        "configs": all_results,
        "best_within_subject": {
            "config": best_ws[0], "classifier": best_ws[1],
            "accuracy": float(f"{best_ws_acc:.4f}"),
        },
        "best_loso": {
            "config": best_loso[0], "classifier": best_loso[1],
            "accuracy": float(f"{best_loso_acc:.4f}"),
        },
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {results_file}")


if __name__ == "__main__":
    run_pipeline()
