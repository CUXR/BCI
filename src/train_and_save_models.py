"""Train and save models for the real-time BCI feedback system.

Trains on all available subject data and saves a model bundle:
  - Blink detector: LDA on all 4 channels (best: 88.7% within-subject)
  - Motor imagery: RF on all 4 channels (best cross-subject generalization)

Usage:
    python src/train_and_save_models.py

Output:
    models/realtime_models.pkl
"""

import sys
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.bandpower import train_motor_imagery as mi_mod
import train_blink_detector as blink_mod
from eeg_filters import EEGFilter

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_FILE = MODELS_DIR / "realtime_models.pkl"

CH_INDICES = [0, 1, 2, 3]  # all 4 channels: TP9, AF7, AF8, TP10


def train_blink_model(subject_data, fs):
    """Train blink detector (LDA) on all subjects."""
    print("\n" + "=" * 60)
    print("Training Blink Detector (LDA, all_4ch)")
    print("=" * 60)

    eeg_filter = EEGFilter(sampling_rate=int(fs), notch_freq=60.0)
    target_samples = int(blink_mod.BLINK_EPOCH_SECONDS * fs)

    all_epochs = []
    all_labels = []

    for sub_id, (eeg, trials, fs_sub, info) in subject_data.items():
        blink_eps = blink_mod.extract_blink_epochs(
            eeg, trials, fs_sub, CH_INDICES, eeg_filter)
        nonblink_eps = blink_mod.extract_nonblink_epochs(
            eeg, trials, fs_sub, CH_INDICES, eeg_filter,
            target_samples=target_samples)

        n_pos, n_neg = len(blink_eps), len(nonblink_eps)
        flag = " [LOW BLINK COUNT]" if n_pos < 5 else ""
        print(f"  {sub_id}: {n_pos} blink + {n_neg} non-blink{flag}")

        all_epochs.extend(blink_eps)
        all_labels.extend([1] * n_pos)
        all_epochs.extend(nonblink_eps)
        all_labels.extend([0] * n_neg)

    if not all_epochs:
        print("ERROR: No blink epochs extracted!")
        return None

    # Pad to same length
    max_len = max(ep.shape[0] for ep in all_epochs)
    padded = []
    for ep in all_epochs:
        if ep.shape[0] < max_len:
            ep = np.pad(ep, ((0, max_len - ep.shape[0]), (0, 0)), mode="edge")
        padded.append(ep)

    y = np.array(all_labels)
    X = blink_mod.extract_features_all(padded, fs, CH_INDICES)

    print(f"  Features: {X.shape[1]}, Epochs: {X.shape[0]} "
          f"(blink={np.sum(y == 1)}, non-blink={np.sum(y == 0)})")

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
    pipeline.fit(X, y)

    train_acc = np.mean(pipeline.predict(X) == y)
    print(f"  Training accuracy: {train_acc:.1%}")

    return pipeline


def train_mi_model(subject_data, fs):
    """Train motor imagery classifier (RF) on all subjects."""
    print("\n" + "=" * 60)
    print("Training Motor Imagery Classifier (RF, all_4ch)")
    print("=" * 60)

    eeg_filter = EEGFilter(sampling_rate=int(fs), notch_freq=60.0)

    all_epochs = []
    all_labels = []

    for sub_id, (eeg, trials, fs_sub, info) in subject_data.items():
        epochs, labels, trial_ids = mi_mod.extract_mi_epochs(
            eeg, trials, fs_sub, CH_INDICES, eeg_filter)
        print(f"  {sub_id}: {len(epochs)} epochs "
              f"(L={labels.count(0)}, R={labels.count(1)})")
        all_epochs.extend(epochs)
        all_labels.extend(labels)

    if not all_epochs:
        print("ERROR: No MI epochs extracted!")
        return None, None

    # Pad to same length
    max_len = max(ep.shape[0] for ep in all_epochs)
    padded = []
    for ep in all_epochs:
        if ep.shape[0] < max_len:
            ep = np.pad(ep, ((0, max_len - ep.shape[0]), (0, 0)), mode="edge")
        padded.append(ep)

    y = np.array(all_labels)
    X_feat = mi_mod.extract_features_all(padded, fs, CH_INDICES)
    csp_features, W_csp = mi_mod.compute_csp(padded, all_labels)
    X = np.hstack([X_feat, csp_features])

    print(f"  Features: {X.shape[1]} "
          f"(spectral={X_feat.shape[1]}, CSP={csp_features.shape[1]})")
    print(f"  Epochs: {X.shape[0]} (L={np.sum(y == 0)}, R={np.sum(y == 1)})")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=3,
            random_state=42)),
    ])
    pipeline.fit(X, y)

    train_acc = np.mean(pipeline.predict(X) == y)
    print(f"  Training accuracy: {train_acc:.1%}")

    return pipeline, W_csp


def main():
    print("=" * 60)
    print("BCI Model Training for Real-Time Feedback")
    print("=" * 60)

    subjects = mi_mod.find_subject_data(DATA_DIR)
    if not subjects:
        print("ERROR: No subject data found in", DATA_DIR)
        return

    print(f"Found {len(subjects)} subjects: {list(subjects.keys())}")

    subject_data = {}
    for sub_id, info in subjects.items():
        eeg, trials, fs = mi_mod.load_subject(info)
        subject_data[sub_id] = (eeg, trials, fs, info)

    fs = list(subject_data.values())[0][2]

    blink_model = train_blink_model(subject_data, fs)
    mi_model, mi_csp_W = train_mi_model(subject_data, fs)

    MODELS_DIR.mkdir(exist_ok=True)

    bundle = {
        "blink_pipeline": blink_model,
        "blink_ch_indices": CH_INDICES,
        "mi_pipeline": mi_model,
        "mi_ch_indices": CH_INDICES,
        "mi_csp_W": mi_csp_W,
        "sampling_rate": int(fs),
        "ch_names": ["TP9", "AF7", "AF8", "TP10"],
        "training_date": datetime.now().isoformat(),
        "n_subjects": len(subjects),
        "subjects": list(subjects.keys()),
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n{'=' * 60}")
    print(f"Model bundle saved to {OUTPUT_FILE}")
    print(f"  Blink model: {'OK' if blink_model else 'FAILED'}")
    print(f"  MI model:    {'OK' if mi_model else 'FAILED'}")
    print(f"  Subjects:    {list(subjects.keys())}")


if __name__ == "__main__":
    main()
