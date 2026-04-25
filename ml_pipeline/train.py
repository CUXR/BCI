#!/usr/bin/env python3
"""Main training script for the BCI ML pipeline.

Loads all subject data, runs preprocessing, trains models, evaluates
with within-subject CV and LOSO, then saves a production model bundle.

Usage:
    cd ml_pipeline
    python train.py
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from config import TARGET_SFREQ, RESULTS_DIR, MODELS_DIR
from dataset import build_dataset
from features import (
    extract_mi_features_batch, compute_csp, apply_csp,
    extract_blink_features_batch,
)
from models import (
    get_mi_classifiers, get_blink_classifiers,
    evaluate_within_subject, evaluate_loso, print_results,
    train_final_model, save_model_bundle,
)


def train_mi(dataset):
    """Train and evaluate motor imagery classifiers."""
    print("\n" + "=" * 70)
    print("MOTOR IMAGERY CLASSIFICATION (left vs right)")
    print("=" * 70)

    epochs = dataset["mi_epochs"]
    labels = dataset["mi_labels"]
    subjects = dataset["mi_subjects"]

    if not epochs:
        print("  No MI epochs available.")
        return None, None, None

    print(f"\n  Total epochs: {len(epochs)}")
    print(f"  Left: {sum(1 for l in labels if l == 0)}, "
          f"Right: {sum(1 for l in labels if l == 1)}")
    print(f"  Subjects: {sorted(set(subjects))}")

    # Extract spectral/time-domain features
    print("\n  Extracting features...")
    X_feat = extract_mi_features_batch(epochs, TARGET_SFREQ)
    print(f"    Spectral/time features: {X_feat.shape[1]}")

    # CSP features
    csp_features, W_csp = compute_csp(epochs, labels)
    print(f"    CSP features: {csp_features.shape[1]}")

    X = np.hstack([X_feat, csp_features])
    y = np.array(labels)
    print(f"    Total feature vector: {X.shape[1]}")

    # Evaluate
    classifiers = get_mi_classifiers()

    ws_results = evaluate_within_subject(X, y, subjects, classifiers)
    ws_means = print_results(ws_results, "Within-Subject CV")

    loso_results = evaluate_loso(X, y, subjects, classifiers)
    loso_means = print_results(loso_results, "Leave-One-Subject-Out CV")

    # Train final ensemble on all data
    print("\n  Training final ensemble model on all data...")
    final_model = train_final_model(X, y, model_type="mi")
    train_acc = np.mean(final_model.predict(X) == y)
    print(f"    Training accuracy: {train_acc:.1%}")

    return final_model, W_csp, {
        "within_subject": ws_results,
        "loso": loso_results,
        "ws_means": ws_means,
        "loso_means": loso_means,
        "n_features": X.shape[1],
        "n_epochs": len(epochs),
    }


def train_blink(dataset):
    """Train and evaluate blink detection classifiers."""
    print("\n" + "=" * 70)
    print("BLINK DETECTION (blink vs non-blink)")
    print("=" * 70)

    pos_epochs = dataset["blink_pos_epochs"]
    neg_epochs = dataset["blink_neg_epochs"]

    if not pos_epochs:
        print("  No blink epochs available.")
        return None, None

    all_epochs = pos_epochs + neg_epochs
    all_labels = [1] * len(pos_epochs) + [0] * len(neg_epochs)

    # Build subject IDs for blink epochs
    blink_subjects = dataset["blink_subjects"]
    # If subject tracking is shorter than combined epochs, extend
    if len(blink_subjects) < len(all_epochs):
        blink_subjects = blink_subjects + ["unknown"] * (len(all_epochs) - len(blink_subjects))

    print(f"\n  Blink (positive): {len(pos_epochs)}")
    print(f"  Non-blink (negative): {len(neg_epochs)}")

    # Extract features
    print("\n  Extracting features...")
    X = extract_blink_features_batch(all_epochs, TARGET_SFREQ)
    y = np.array(all_labels)
    print(f"    Feature vector: {X.shape[1]}")

    # Evaluate
    classifiers = get_blink_classifiers()

    ws_results = evaluate_within_subject(X, y, blink_subjects, classifiers)
    ws_means = print_results(ws_results, "Within-Subject CV")

    loso_results = evaluate_loso(X, y, blink_subjects, classifiers)
    loso_means = print_results(loso_results, "Leave-One-Subject-Out CV")

    # Train final ensemble
    print("\n  Training final ensemble model on all data...")
    final_model = train_final_model(X, y, model_type="blink")
    train_acc = np.mean(final_model.predict(X) == y)
    print(f"    Training accuracy: {train_acc:.1%}")

    return final_model, {
        "within_subject": ws_results,
        "loso": loso_results,
        "ws_means": ws_means,
        "loso_means": loso_means,
        "n_features": X.shape[1],
        "n_epochs": len(all_epochs),
    }


def main():
    print("=" * 70)
    print("BCI ML Pipeline — Training")
    print("=" * 70)

    # Load all data
    print("\nLoading data...")
    dataset = build_dataset(verbose=True)
    if dataset is None:
        print("\nERROR: No data found. Place subject folders (sub01, sub02, ...) in:")
        from config import DATA_DIRS
        for d in DATA_DIRS:
            print(f"  {d}")
        sys.exit(1)

    # Print dataset summary
    print(f"\n{'=' * 70}")
    print("Dataset Summary")
    print("=" * 70)
    for sub_id, info in dataset["subjects_info"].items():
        print(f"  {sub_id}: {info['n_sessions']} sessions, "
              f"{info['n_mi_epochs']} MI epochs, "
              f"{info['n_blink_pos']} blink+, {info['n_blink_neg']} blink-")

    # Train MI model
    mi_model, mi_csp_W, mi_results = train_mi(dataset)

    # Train blink model
    blink_model, blink_results = train_blink(dataset)

    # Save model bundle
    subjects_list = list(dataset["subjects_info"].keys())
    out_path = save_model_bundle(
        mi_pipeline=mi_model,
        mi_csp_W=mi_csp_W,
        blink_pipeline=blink_model,
        subjects_list=subjects_list,
    )

    print(f"\n{'=' * 70}")
    print("SAVED")
    print("=" * 70)
    print(f"  Model bundle: {out_path}")
    print(f"  MI model:     {'OK' if mi_model else 'FAILED'}")
    print(f"  Blink model:  {'OK' if blink_model else 'FAILED'}")
    print(f"  Subjects:     {subjects_list}")

    # Save detailed results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "training_results.json"

    def _clean_results(res):
        if res is None:
            return None
        cleaned = {}
        for k, v in res.items():
            if isinstance(v, dict):
                cleaned[k] = {}
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        cleaned[k][k2] = {
                            k3: (float(f"{v3:.4f}") if isinstance(v3, float) and not np.isnan(v3) else None)
                            for k3, v3 in v2.items()
                        }
                    elif isinstance(v2, float):
                        cleaned[k][k2] = float(f"{v2:.4f}") if not np.isnan(v2) else None
                    else:
                        cleaned[k][k2] = v2
            else:
                cleaned[k] = v
        return cleaned

    output = {
        "mi": _clean_results(mi_results),
        "blink": _clean_results(blink_results),
        "subjects": subjects_list,
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results:      {results_path}")

    print(f"\nTo run real-time BCI:")
    print(f"  python realtime.py              # with Muse 2 headband")
    print(f"  python realtime.py --simulate   # synthetic data for testing")


if __name__ == "__main__":
    main()
