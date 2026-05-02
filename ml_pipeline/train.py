#!/usr/bin/env python3
"""Main training script for the BCI ML pipeline.

Loads all subject data, runs preprocessing, trains models, evaluates
with within-subject CV and LOSO, then saves a production model bundle.

Two variants:
    --variant 2class  (legacy)  binary L/R MI + blink — reads psychopy
                                style trial logs (start_time/end_time,
                                EEG_0..EEG_3 columns) via dataset.py.
    --variant 4class  (default) navigation MI (forward / backward /
                                rotate-left / rotate-right) + blink —
                                reads pipeline.collector output
                                (per-phase trial rows, TP9/AF7/AF8/TP10
                                columns) via dataset_4class.py. Drops
                                CSP (binary-only) in favour of richer
                                spectral + Hjorth + asymmetry features.

Usage:
    cd ml_pipeline
    python train.py                       # 4-class (collector data)
    python train.py --variant 2class      # legacy psychopy data
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from config import (                                                # type: ignore[import-not-found]
    CH_NAMES, MODELS_DIR, RANDOM_STATE, RESULTS_DIR, TARGET_SFREQ,
)
from dataset import build_dataset                                   # type: ignore[import-not-found]
from features import (                                              # type: ignore[import-not-found]
    extract_mi_features_batch, compute_csp, apply_csp,
    extract_blink_features_batch,
)
from models import (                                                # type: ignore[import-not-found]
    get_mi_classifiers, get_blink_classifiers,
    evaluate_within_subject, evaluate_loso, print_results,
    train_final_model, save_model_bundle,
)
from sklearn.ensemble import RandomForestClassifier                 # noqa: E402
from sklearn.metrics import confusion_matrix                        # noqa: E402
from sklearn.pipeline import Pipeline                               # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402

# 4-class loader is in a separate module so the legacy 2-class path
# stays untouched.
from dataset_4class import build_dataset_4class                     # type: ignore[import-not-found]


BASE_4CLASS_BUNDLE_NAME = "realtime_models_4class.pkl"
BASE_4CLASS_MODEL_VERSION = "base_4class_v1"
MI_CLASS_NAMES_4CLASS = ["mi_forward", "mi_backward", "mi_rotate_left", "mi_rotate_right"]


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


def train_mi_4class(dataset):
    """Train and evaluate 4-class motor imagery classifiers (no CSP).

    CSP is a binary construct; for the 4-class navigation head we lean
    on RandomForest with the rich spectral + Hjorth + asymmetry feature
    set (matches the user spec "RandomForest for MI").
    """
    print("\n" + "=" * 70)
    print("MOTOR IMAGERY CLASSIFICATION (4-class navigation)")
    print("=" * 70)

    epochs = dataset["mi_epochs"]
    labels = dataset["mi_labels"]
    subjects = dataset["mi_subjects"]
    class_names = dataset["mi_class_names"]

    if not epochs:
        print("  No 4-class MI epochs available.")
        return None, None

    print(f"\n  Total epochs: {len(epochs)}")
    counts = {name: sum(1 for l in labels if l == i)
              for i, name in enumerate(class_names)}
    print(f"  Class counts: {counts}")
    print(f"  Subjects: {sorted(set(subjects))}")

    print("\n  Extracting features...")
    X = extract_mi_features_batch(epochs, TARGET_SFREQ)
    y = np.array(labels)
    print(f"    Spectral/time features: {X.shape[1]}")

    classifiers = get_mi_classifiers()

    ws_results = evaluate_within_subject(X, y, subjects, classifiers)
    ws_means = print_results(ws_results, "Within-Subject CV (4-class)")

    loso_results = evaluate_loso(X, y, subjects, classifiers)
    loso_means = print_results(loso_results, "Leave-One-Subject-Out CV (4-class)")

    print("\n  Training final RandomForest on all data...")
    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=2,
            random_state=RANDOM_STATE,
        )),
    ])
    final.fit(X, y)
    train_acc = float(np.mean(final.predict(X) == y))
    print(f"    Training accuracy: {train_acc:.1%}")

    cm = confusion_matrix(y, final.predict(X), labels=list(range(len(class_names))))
    print("\n    Confusion matrix (rows=true, cols=pred):")
    header = "    " + " " * 12 + "  ".join(f"{n[:6]:>6}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        cells = "  ".join(f"{int(c):>6}" for c in row)
        print(f"    {class_names[i][:10]:>10}  {cells}")

    return final, {
        "within_subject": ws_results,
        "loso": loso_results,
        "ws_means": ws_means,
        "loso_means": loso_means,
        "n_features": X.shape[1],
        "n_epochs": len(epochs),
        "class_counts": counts,
        "train_accuracy": train_acc,
        "confusion_matrix": cm.tolist(),
    }


def _save_4class_bundle(mi_pipeline, blink_pipeline, *, subjects_list, dataset, out_path):
    """Save base 4-class bundle through the unified models.save_model_bundle()."""
    return save_model_bundle(
        mi_pipeline=mi_pipeline,
        mi_csp_W=None,
        blink_pipeline=blink_pipeline,
        subjects_list=subjects_list,
        class_names=list(MI_CLASS_NAMES_4CLASS),
        blink_class_names=["non_blink", "intentional_blink"],
        model_version=BASE_4CLASS_MODEL_VERSION,
        model_kind="base_4class",
        mi_label_map=dict(dataset["mi_label_map"]),
        output_path=out_path,
        extra_info={
            "n_mi_epochs": len(dataset["mi_epochs"]),
            "n_blink_pos": len(dataset["blink_pos_epochs"]),
            "n_blink_neg": len(dataset["blink_neg_epochs"]),
        },
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="BCI base-model trainer")
    p.add_argument(
        "--variant",
        choices=["2class", "4class"],
        default="4class",
        help="Which trial taxonomy to train on (default: 4class — "
             "reads pipeline.collector output)",
    )
    return p.parse_args(argv)


def _run_2class():
    print("=" * 70)
    print("BCI ML Pipeline — Training (2-class legacy variant)")
    print("=" * 70)

    print("\nLoading data...")
    dataset = build_dataset(verbose=True)
    if dataset is None:
        print("\nERROR: No 2-class data found. Place subject folders in:")
        from config import DATA_DIRS                                # type: ignore[import-not-found]  # noqa: PLC0415
        for d in DATA_DIRS:
            print(f"  {d}")
        sys.exit(1)

    print(f"\n{'=' * 70}\nDataset Summary\n{'=' * 70}")
    for sub_id, info in dataset["subjects_info"].items():
        print(f"  {sub_id}: {info['n_sessions']} sessions, "
              f"{info['n_mi_epochs']} MI epochs, "
              f"{info['n_blink_pos']} blink+, {info['n_blink_neg']} blink-")

    mi_model, mi_csp_W, mi_results = train_mi(dataset)
    blink_model, blink_results = train_blink(dataset)

    subjects_list = list(dataset["subjects_info"].keys())
    out_path = save_model_bundle(
        mi_pipeline=mi_model,
        mi_csp_W=mi_csp_W,
        blink_pipeline=blink_model,
        subjects_list=subjects_list,
    )
    return out_path, mi_results, blink_results, subjects_list


def _run_4class():
    print("=" * 70)
    print("BCI ML Pipeline — Training (4-class navigation MI)")
    print("=" * 70)

    print("\nLoading collector data...")
    dataset = build_dataset_4class(verbose=True)
    if dataset is None:
        print(
            "\nERROR: No 4-class collector data found. Run "
            "`python -m pipeline collect --participant N --name ... --runs M` "
            "first, then retry training."
        )
        sys.exit(1)

    print(f"\n{'=' * 70}\nDataset Summary\n{'=' * 70}")
    for sub_id, info in dataset["subjects_info"].items():
        print(f"  {sub_id}: {info['n_sessions']} sessions, "
              f"{info['n_mi_epochs']} MI epochs, "
              f"{info['n_blink_pos']} blink+, {info['n_blink_neg']} blink-")

    mi_model, mi_results = train_mi_4class(dataset)
    blink_model, blink_results = train_blink(dataset)

    subjects_list = list(dataset["subjects_info"].keys())
    out_path = MODELS_DIR / BASE_4CLASS_BUNDLE_NAME
    _save_4class_bundle(
        mi_pipeline=mi_model,
        blink_pipeline=blink_model,
        subjects_list=subjects_list,
        dataset=dataset,
        out_path=out_path,
    )
    return out_path, mi_results, blink_results, subjects_list


def main(argv=None):
    args = _parse_args(argv)

    if args.variant == "2class":
        out_path, mi_results, blink_results, subjects_list = _run_2class()
    else:
        out_path, mi_results, blink_results, subjects_list = _run_4class()

    print(f"\n{'=' * 70}\nSAVED\n{'=' * 70}")
    print(f"  Variant:      {args.variant}")
    print(f"  Model bundle: {out_path}")
    print(f"  Subjects:     {subjects_list}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"training_results_{args.variant}.json"

    def _clean(res):
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

    with open(results_path, "w") as f:
        json.dump({
            "variant": args.variant,
            "mi": _clean(mi_results),
            "blink": _clean(blink_results),
            "subjects": subjects_list,
        }, f, indent=2, default=str)
    print(f"  Results:      {results_path}")

    print("\nTo run real-time BCI:")
    if args.variant == "4class":
        print("  python -m pipeline realtime --participant <N>   # personalised + live")
    else:
        print("  python realtime.py              # with Muse 2 headband")
        print("  python realtime.py --simulate   # synthetic data for testing")


if __name__ == "__main__":
    main()
