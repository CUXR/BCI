"""Blink Detection Classifier Pipeline.

Trains and evaluates intentional blink detectors using EEG data from
Muse 2 headband (4 channels: TP9, AF7, AF8, TP10 @ 256 Hz).

Positive class: intentional blink epochs (~500ms windows from trial log).
Negative class: random 500ms windows sampled from motor imagery trials.

Compares 2 channel configurations:
  (a) AF7+AF8 only (frontal, closest to eyes)
  (b) All 4 channels

Evaluates 4 classifiers + a simple peak-to-peak threshold baseline,
with within-subject 5-fold CV and Leave-One-Subject-Out CV.
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_filters import EEGFilter

# ---------- constants ----------
DATA_DIR = PROJECT_ROOT / "data"
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
CH_COLS = ["EEG_0", "EEG_1", "EEG_2", "EEG_3"]
MI_TYPES = ["motor_imagery_left", "motor_imagery_right"]

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "mu": (8, 12),
    "beta": (13, 30),
}

CHANNEL_CONFIGS = {
    "frontal_only": [1, 2],        # AF7, AF8
    "all_4ch": [0, 1, 2, 3],      # TP9, AF7, AF8, TP10
}

BLINK_EPOCH_SECONDS = 0.5  # ~128 samples at 256 Hz
NEG_PER_MI_TRIAL = 2       # negative windows sampled per MI trial


# ============================================================
# Data loading (reused from train_motor_imagery.py)
# ============================================================

def find_subject_data(data_dir):
    """Discover all subject directories with EEG data (recursive).

    Walks ``data_dir`` for any ``sub*`` directory containing both an
    ``eeg_data_*.csv`` and a ``trial_log_*.csv`` so the Drive layout
    (``data/Muse_Data/Phase{1,2}/sub*/``) works alongside a flat
    ``data/sub*/``. Keys disambiguate with the parent folder when a bare
    ``subNN`` would collide (e.g. Phase1/sub05 vs Phase2/sub05).
    """
    subjects = {}
    for sub_dir in sorted(data_dir.rglob("sub*")):
        if not sub_dir.is_dir():
            continue
        eeg_files = list(sub_dir.glob("eeg_data_*.csv"))
        trial_files = list(sub_dir.glob("trial_log_*.csv"))
        if not (eeg_files and trial_files):
            continue
        meta_file = sub_dir / "metadata.yaml"
        label = sub_dir.name
        if meta_file.exists():
            for line in meta_file.read_text().splitlines():
                if line.startswith("participant:"):
                    label = f"{sub_dir.name} ({line.split(':')[1].strip()})"
        key = sub_dir.name
        if key in subjects:
            key = f"{sub_dir.parent.name}_{sub_dir.name}"
        subjects[key] = {
            "dir": sub_dir,
            "eeg": eeg_files[0],
            "trials": trial_files[0],
            "label": label,
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
# Epoch extraction
# ============================================================

def extract_blink_epochs(eeg, trials, fs, ch_indices, eeg_filter):
    """Extract blink epochs using trial_log timestamps.

    Returns:
        epochs: list of 2D arrays (n_samples, n_channels)
    """
    blink_trials = trials[trials["trial_type"] == "blink_intentional"]
    epochs = []

    for _, trial in blink_trials.iterrows():
        mask = (eeg["timestamp"] >= trial["start_time"]) & (
            eeg["timestamp"] <= trial["end_time"])
        segment = eeg.loc[mask, [CH_COLS[i] for i in ch_indices]].values

        if len(segment) < 10:  # too short to be useful
            continue

        # Preprocess per channel
        processed = np.zeros_like(segment, dtype=np.float64)
        bad_epoch = False
        for col_i in range(segment.shape[1]):
            ch_data = segment[:, col_i].astype(np.float64)
            # Replace NaN/Inf before filtering
            ch_data = np.nan_to_num(ch_data, nan=0.0, posinf=0.0, neginf=0.0)
            ch_data = eeg_filter.apply_moving_artifact_removal(ch_data, threshold=3.0)
            ch_data = np.nan_to_num(ch_data, nan=0.0, posinf=0.0, neginf=0.0)
            ch_data = eeg_filter.apply_comprehensive_filter(ch_data)
            ch_data = np.nan_to_num(ch_data, nan=0.0, posinf=0.0, neginf=0.0)
            processed[:, col_i] = ch_data

        epochs.append(processed)

    return epochs


def extract_nonblink_epochs(eeg, trials, fs, ch_indices, eeg_filter,
                            target_samples, n_per_trial=NEG_PER_MI_TRIAL,
                            rng=None):
    """Sample random non-blink windows from motor imagery trials.

    For each MI trial, samples n_per_trial non-overlapping windows of
    target_samples length. Uses same preprocessing as blink epochs.

    Returns:
        epochs: list of 2D arrays (n_samples, n_channels)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    mi_trials = trials[trials["trial_type"].isin(MI_TYPES)]
    epochs = []

    for _, trial in mi_trials.iterrows():
        mask = (eeg["timestamp"] >= trial["start_time"]) & (
            eeg["timestamp"] <= trial["end_time"])
        segment = eeg.loc[mask, [CH_COLS[i] for i in ch_indices]].values

        if len(segment) < target_samples + 10:
            continue

        # Pick n_per_trial random non-overlapping start positions
        max_start = len(segment) - target_samples
        if max_start < 1:
            continue

        starts = []
        attempts = 0
        while len(starts) < n_per_trial and attempts < 100:
            s = rng.integers(0, max_start)
            # Check no overlap with existing windows
            overlap = False
            for existing in starts:
                if abs(s - existing) < target_samples:
                    overlap = True
                    break
            if not overlap:
                starts.append(s)
            attempts += 1

        for s in starts:
            window = segment[s:s + target_samples]
            processed = np.zeros_like(window, dtype=np.float64)
            for col_i in range(window.shape[1]):
                ch_data = window[:, col_i].astype(np.float64)
                ch_data = np.nan_to_num(ch_data, nan=0.0, posinf=0.0, neginf=0.0)
                ch_data = eeg_filter.apply_moving_artifact_removal(
                    ch_data, threshold=3.0)
                ch_data = np.nan_to_num(ch_data, nan=0.0, posinf=0.0, neginf=0.0)
                ch_data = eeg_filter.apply_comprehensive_filter(ch_data)
                ch_data = np.nan_to_num(ch_data, nan=0.0, posinf=0.0, neginf=0.0)
                processed[:, col_i] = ch_data
            epochs.append(processed)

    return epochs


# ============================================================
# Feature extraction
# ============================================================

def _bandpower(signal_data, fs, fmin, fmax):
    """Compute absolute band power using Welch's method with small nperseg."""
    nperseg = min(int(fs * 0.5), len(signal_data))
    if nperseg < 16:
        return 0.0
    freqs, psd = welch(signal_data, fs=fs, nperseg=nperseg)
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return 0.0
    return np.trapz(psd[idx], freqs[idx])


def _hjorth_parameters(signal_data):
    """Compute Hjorth activity, mobility, complexity."""
    activity = np.var(signal_data)
    d1 = np.diff(signal_data)
    d2 = np.diff(d1)
    var_d1 = np.var(d1)
    var_d2 = np.var(d2)
    mobility = np.sqrt(var_d1 / (activity + 1e-12))
    mob_d1 = np.sqrt(var_d2 / (var_d1 + 1e-12))
    complexity = mob_d1 / (mobility + 1e-12)
    return activity, mobility, complexity


def _zero_crossing_rate(signal_data):
    """Fraction of consecutive samples that cross zero."""
    return np.mean(np.diff(np.sign(signal_data)) != 0)


def extract_blink_features(epoch, fs, ch_indices):
    """Extract time-domain + spectral features from a single epoch.

    Per channel:
        - Peak-to-peak amplitude, RMS, max absolute value
        - Hjorth parameters (activity, mobility, complexity)
        - Skewness, kurtosis
        - Zero-crossing rate
        - Position of peak (argmax / n_samples)
        - Band powers (delta, theta, mu, beta)

    Cross-channel:
        - Frontal correlation (AF7-AF8)
        - Temporal correlation (TP9-TP10)
        - Frontal-temporal amplitude ratio

    Returns a 1D feature vector.
    """
    n_samples = epoch.shape[0]
    n_ch = epoch.shape[1]
    features = []

    ch_rms = []
    ch_abs_mean = []

    for ch_i in range(n_ch):
        sig = epoch[:, ch_i]

        # Amplitude features
        ptp = np.ptp(sig)  # peak-to-peak
        rms = np.sqrt(np.mean(sig ** 2))
        max_abs = np.max(np.abs(sig))
        features.extend([ptp, rms, max_abs])

        ch_rms.append(rms)
        ch_abs_mean.append(np.mean(np.abs(sig)))

        # Hjorth parameters
        activity, mobility, complexity = _hjorth_parameters(sig)
        features.extend([activity, mobility, complexity])

        # Shape features
        sk = skew(sig)
        kurt = kurtosis(sig)
        zcr = _zero_crossing_rate(sig)
        peak_pos = np.argmax(np.abs(sig)) / max(n_samples - 1, 1)
        features.extend([sk, kurt, zcr, peak_pos])

        # Band powers (with small nperseg for short epochs)
        for band_name, (fmin, fmax) in BANDS.items():
            bp = _bandpower(sig, fs, fmin, fmax)
            features.append(bp)

    # Cross-channel features
    # Frontal correlation (AF7-AF8) if both present
    if 1 in ch_indices and 2 in ch_indices:
        af7_idx = ch_indices.index(1)
        af8_idx = ch_indices.index(2)
        corr = np.corrcoef(epoch[:, af7_idx], epoch[:, af8_idx])[0, 1]
        features.append(corr if np.isfinite(corr) else 0.0)
    else:
        features.append(0.0)

    # Temporal correlation (TP9-TP10) if both present
    if 0 in ch_indices and 3 in ch_indices:
        tp9_idx = ch_indices.index(0)
        tp10_idx = ch_indices.index(3)
        corr = np.corrcoef(epoch[:, tp9_idx], epoch[:, tp10_idx])[0, 1]
        features.append(corr if np.isfinite(corr) else 0.0)
    else:
        features.append(0.0)

    # Frontal-temporal amplitude ratio (if all 4 channels present)
    if 0 in ch_indices and 1 in ch_indices and 2 in ch_indices and 3 in ch_indices:
        af7_idx = ch_indices.index(1)
        af8_idx = ch_indices.index(2)
        tp9_idx = ch_indices.index(0)
        tp10_idx = ch_indices.index(3)
        frontal_amp = (ch_abs_mean[af7_idx] + ch_abs_mean[af8_idx]) / 2
        temporal_amp = (ch_abs_mean[tp9_idx] + ch_abs_mean[tp10_idx]) / 2
        ft_ratio = frontal_amp / (temporal_amp + 1e-12)
        features.append(ft_ratio)
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float64)


def extract_features_all(epochs, fs, ch_indices):
    """Extract features from all epochs. Returns X array."""
    if not epochs:
        return np.empty((0, 0))
    X = np.array([extract_blink_features(ep, fs, ch_indices) for ep in epochs])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


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
# Threshold baseline classifier
# ============================================================

def threshold_baseline(X_train, y_train, X_test, ptp_col_indices):
    """Simple threshold classifier: blink if max peak-to-peak > threshold.

    Optimizes threshold on training data using the peak-to-peak amplitude
    columns (one per channel). Takes max across channels.
    """
    # Max peak-to-peak across channels for each sample
    train_max_ptp = np.max(X_train[:, ptp_col_indices], axis=1)
    test_max_ptp = np.max(X_test[:, ptp_col_indices], axis=1)

    # Find optimal threshold on training data
    best_acc = 0.0
    best_thresh = 0.0
    for pct in np.linspace(0, 100, 200):
        thresh = np.percentile(train_max_ptp, pct)
        y_pred_train = (train_max_ptp > thresh).astype(int)
        acc = accuracy_score(y_train, y_pred_train)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    y_pred = (test_max_ptp > best_thresh).astype(int)
    return y_pred, best_thresh


# ============================================================
# Evaluation
# ============================================================

def evaluate_within_subject(X, y, subject_ids, classifiers, ptp_col_indices):
    """5-fold stratified CV per subject. Includes threshold baseline.

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
            results[sub]["Threshold"] = np.nan
            continue

        n_folds = min(5, min(np.bincount(y_sub)))
        if n_folds < 2:
            results[sub] = {name: np.nan for name in classifiers}
            results[sub]["Threshold"] = np.nan
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

        # Threshold baseline via CV
        thresh_preds = np.zeros_like(y_sub)
        for train_idx, test_idx in cv.split(X_sub, y_sub):
            preds, _ = threshold_baseline(
                X_sub[train_idx], y_sub[train_idx],
                X_sub[test_idx], ptp_col_indices)
            thresh_preds[test_idx] = preds
        results[sub]["Threshold"] = accuracy_score(y_sub, thresh_preds)

    return results


def evaluate_loso(X, y, subject_ids, classifiers, ptp_col_indices):
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
            results[test_sub]["Threshold"] = np.nan
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

        # Threshold baseline
        y_pred_thresh, _ = threshold_baseline(
            X_train, y_train, X_test, ptp_col_indices)
        results[test_sub]["Threshold"] = accuracy_score(y_test, y_pred_thresh)

    return results


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline():
    """Run the full blink detection pipeline."""
    print("=" * 70)
    print("Blink Detection Classification Pipeline")
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
        n_blink = len(trials[trials["trial_type"] == "blink_intentional"])
        n_mi = len(trials[trials["trial_type"].isin(MI_TYPES)])
        print(f"  {info['label']}: {n_blink} blink trials, "
              f"{n_mi} MI trials, Fs={fs:.0f} Hz")
        subject_data[sub_id] = (eeg, trials, fs, info)

    fs = list(subject_data.values())[0][2]
    target_samples = int(BLINK_EPOCH_SECONDS * fs)

    # --- Run classification for each channel config ---
    all_results = {}

    for config_name, ch_indices in CHANNEL_CONFIGS.items():
        print(f"\n{'=' * 70}")
        print(f"Config: {config_name} — channels: "
              f"{[CH_NAMES[i] for i in ch_indices]}")
        print("=" * 70)

        eeg_filter = EEGFilter(sampling_rate=int(fs), notch_freq=60.0)

        all_epochs = []
        all_labels = []
        all_sub_ids = []
        sub_epoch_counts = {}

        for sub_id, (eeg, trials, fs_sub, info) in subject_data.items():
            # Blink epochs (positive class = 1)
            blink_eps = extract_blink_epochs(
                eeg, trials, fs_sub, ch_indices, eeg_filter)

            # Non-blink epochs from MI trials (negative class = 0)
            nonblink_eps = extract_nonblink_epochs(
                eeg, trials, fs_sub, ch_indices, eeg_filter,
                target_samples=target_samples,
                n_per_trial=NEG_PER_MI_TRIAL)

            n_pos = len(blink_eps)
            n_neg = len(nonblink_eps)

            quality_flag = ""
            if n_pos < 5:
                quality_flag = " [LOW BLINK COUNT]"

            print(f"  {sub_id}: {n_pos} blink + {n_neg} non-blink epochs"
                  f"{quality_flag}")

            all_epochs.extend(blink_eps)
            all_labels.extend([1] * n_pos)
            all_sub_ids.extend([sub_id] * n_pos)

            all_epochs.extend(nonblink_eps)
            all_labels.extend([0] * n_neg)
            all_sub_ids.extend([sub_id] * n_neg)

            sub_epoch_counts[sub_id] = {"blink": n_pos, "non_blink": n_neg}

        if not all_epochs:
            print("  No valid epochs — skipping config")
            continue

        # Pad all epochs to same length
        max_len = max(ep.shape[0] for ep in all_epochs)
        padded_epochs = []
        for ep in all_epochs:
            if ep.shape[0] < max_len:
                pad_width = ((0, max_len - ep.shape[0]), (0, 0))
                ep = np.pad(ep, pad_width, mode="edge")
            padded_epochs.append(ep)

        y = np.array(all_labels)
        sub_ids_arr = all_sub_ids

        # Extract features
        X = extract_features_all(padded_epochs, fs, ch_indices)

        # Identify peak-to-peak column indices for threshold baseline
        # Features per channel: ptp(0), rms(1), max_abs(2), hjorth*3(3-5),
        #   skew(6), kurt(7), zcr(8), peak_pos(9), bands*4(10-13)
        # = 14 features per channel
        n_per_ch = 14
        ptp_col_indices = [ch_i * n_per_ch for ch_i in range(len(ch_indices))]

        print(f"\n  Feature matrix: {X.shape[0]} epochs x {X.shape[1]} features")
        print(f"    Class balance: blink={np.sum(y == 1)}, "
              f"non-blink={np.sum(y == 0)}")

        classifiers = get_classifiers()

        # Within-subject CV
        print(f"\n  --- Within-Subject CV ---")
        ws_results = evaluate_within_subject(
            X, y, sub_ids_arr, classifiers, ptp_col_indices)

        all_clf_names = list(classifiers.keys()) + ["Threshold"]

        for sub_id in sorted(ws_results.keys()):
            accs = ws_results[sub_id]
            parts = [f"{name}: {acc:.1%}" if not np.isnan(acc) else f"{name}: N/A"
                     for name, acc in accs.items()]
            print(f"    {sub_id}: {', '.join(parts)}")

        # Mean across subjects
        ws_means = {}
        for clf_name in all_clf_names:
            vals = [ws_results[s][clf_name] for s in ws_results
                    if not np.isnan(ws_results[s][clf_name])]
            ws_means[clf_name] = np.mean(vals) if vals else np.nan

        print(f"    {'Mean':<6}: ", end="")
        print(", ".join(f"{name}: {acc:.1%}" if not np.isnan(acc)
                        else f"{name}: N/A"
                        for name, acc in ws_means.items()))

        # LOSO CV
        print(f"\n  --- Leave-One-Subject-Out CV ---")
        loso_results = evaluate_loso(
            X, y, sub_ids_arr, classifiers, ptp_col_indices)

        for sub_id in sorted(loso_results.keys()):
            accs = loso_results[sub_id]
            parts = [f"{name}: {acc:.1%}" if not np.isnan(acc) else f"{name}: N/A"
                     for name, acc in accs.items()]
            print(f"    {sub_id} (test): {', '.join(parts)}")

        loso_means = {}
        for clf_name in all_clf_names:
            vals = [loso_results[s][clf_name] for s in loso_results
                    if not np.isnan(loso_results[s][clf_name])]
            loso_means[clf_name] = np.mean(vals) if vals else np.nan

        print(f"    {'Mean':<6}: ", end="")
        print(", ".join(f"{name}: {acc:.1%}" if not np.isnan(acc)
                        else f"{name}: N/A"
                        for name, acc in loso_means.items()))

        # Best classifier classification report (LOSO)
        best_loso_clf = max(
            [(name, acc) for name, acc in loso_means.items()
             if acc is not None and not np.isnan(acc)],
            key=lambda x: x[1], default=(None, 0)
        )

        if best_loso_clf[0] and best_loso_clf[0] != "Threshold":
            print(f"\n  --- Classification Report (LOSO, best={best_loso_clf[0]}) ---")
            clf = classifiers[best_loso_clf[0]]
            # Full LOSO predictions
            y_pred_all = np.zeros_like(y)
            unique_subs = sorted(set(sub_ids_arr))
            for test_sub in unique_subs:
                test_mask = np.array([s == test_sub for s in sub_ids_arr])
                train_mask = ~test_mask
                try:
                    clf.fit(X[train_mask], y[train_mask])
                    y_pred_all[test_mask] = clf.predict(X[test_mask])
                except Exception:
                    y_pred_all[test_mask] = 0
            print(classification_report(
                y, y_pred_all, target_names=["non-blink", "blink"]))

        all_results[config_name] = {
            "channels": [CH_NAMES[i] for i in ch_indices],
            "n_epochs": len(all_epochs),
            "n_features": X.shape[1],
            "epoch_counts_per_subject": sub_epoch_counts,
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
    print("SUMMARY — Blink Detection Results")
    print("=" * 70)

    for config, res in all_results.items():
        print(f"\n  [{config}] channels={res['channels']}")
        print(f"    Within-subject (mean): ", end="")
        print(", ".join(
            f"{clf}: {acc:.1%}" if acc is not None else f"{clf}: N/A"
            for clf, acc in res["within_subject_mean"].items()))
        print(f"    LOSO (mean):           ", end="")
        print(", ".join(
            f"{clf}: {acc:.1%}" if acc is not None else f"{clf}: N/A"
            for clf, acc in res["loso_mean"].items()))

    # Best overall
    best_ws_acc = 0
    best_ws = ("", "")
    best_loso_acc = 0
    best_loso = ("", "")
    for config, res in all_results.items():
        for clf, acc in res["within_subject_mean"].items():
            if acc is not None and acc > best_ws_acc:
                best_ws_acc = acc
                best_ws = (config, clf)
        for clf, acc in res["loso_mean"].items():
            if acc is not None and acc > best_loso_acc:
                best_loso_acc = acc
                best_loso = (config, clf)

    print(f"\n  Best within-subject: {best_ws[1]} with {best_ws[0]} "
          f"— {best_ws_acc:.1%}")
    print(f"  Best LOSO:           {best_loso[1]} with {best_loso[0]} "
          f"— {best_loso_acc:.1%}")

    # Save results
    results_file = DATA_DIR / "blink_detection_results.json"
    output = {
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
