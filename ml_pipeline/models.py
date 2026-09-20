"""Model definitions, training, and evaluation.

Provides:
  - Individual classifiers (LDA, SVM, RF)
  - Voting ensemble for robustness
  - Within-subject CV and Leave-One-Subject-Out CV
  - 3-class unified model (left / right / blink)
  - Model serialization for real-time use
"""

import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, GradientBoostingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

from config import CV_FOLDS, RANDOM_STATE, MODELS_DIR




# ── Bundle schema/version constants ───────────────────────────────
DEFAULT_BUNDLE_FILENAME = "realtime_models.pkl"
BASE_4CLASS_BUNDLE_FILENAME = "realtime_models_4class.pkl"
PERSONALIZED_BUNDLE_SUBDIR = "participants"
LEGACY_MODEL_VERSION = "legacy_2class"
BASE_4CLASS_MODEL_VERSION = "base_4class_v1"
PERSONALIZED_MODEL_VERSION = "personalized_v1"


# ═══════════════════════════════════════════════════════════════════
# Classifier definitions
# ═══════════════════════════════════════════════════════════════════

def get_mi_classifiers():
    """Motor imagery classifiers — binary (left vs right)."""
    return {
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "SVM_linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0, probability=True, random_state=RANDOM_STATE)),
        ]),
        "SVM_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                        random_state=RANDOM_STATE)),
        ]),
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=2,
                random_state=RANDOM_STATE)),
        ]),
        "GBM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=RANDOM_STATE)),
        ]),
    }


def get_blink_classifiers():
    """Blink detection classifiers — binary (blink vs non-blink)."""
    return {
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "SVM_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                        random_state=RANDOM_STATE)),
        ]),
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=2,
                random_state=RANDOM_STATE)),
        ]),
    }


def build_ensemble(classifiers_dict, voting="soft"):
    """Build a VotingClassifier from a dict of pipelines."""
    estimators = [(name, clf) for name, clf in classifiers_dict.items()]
    return VotingClassifier(estimators=estimators, voting=voting)


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_within_subject(X, y, subject_ids, classifiers, n_folds=None):
    """Per-subject stratified k-fold CV.

    Returns:
        results: {subject_id: {clf_name: accuracy}}
    """
    n_folds = n_folds or CV_FOLDS
    unique_subs = sorted(set(subject_ids))
    results = {}

    for sub in unique_subs:
        mask = np.array([s == sub for s in subject_ids])
        X_sub, y_sub = X[mask], y[mask]

        if len(np.unique(y_sub)) < 2 or len(y_sub) < 4:
            results[sub] = {name: np.nan for name in classifiers}
            continue

        k = min(n_folds, min(np.bincount(y_sub)))
        if k < 2:
            results[sub] = {name: np.nan for name in classifiers}
            continue

        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        results[sub] = {}

        for name, clf in classifiers.items():
            try:
                y_pred = cross_val_predict(clf, X_sub, y_sub, cv=cv)
                results[sub][name] = float(accuracy_score(y_sub, y_pred))
            except Exception:
                results[sub][name] = np.nan

    return results


def evaluate_loso(X, y, subject_ids, classifiers):
    """Leave-One-Subject-Out cross-validation.

    Returns:
        results: {test_subject: {clf_name: accuracy}}
    """
    unique_subs = sorted(set(subject_ids))
    if len(unique_subs) < 2:
        return {}

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
        for name, clf in classifiers.items():
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                results[test_sub][name] = float(accuracy_score(y_test, y_pred))
            except Exception:
                results[test_sub][name] = np.nan

    return results


def print_results(results, title=""):
    """Pretty-print evaluation results."""
    if not results:
        print(f"  {title}: No results (need >= 2 subjects for LOSO)")
        return

    clf_names = list(next(iter(results.values())).keys())

    print(f"\n  --- {title} ---")
    for sub in sorted(results.keys()):
        parts = []
        for name in clf_names:
            acc = results[sub][name]
            parts.append(f"{name}: {acc:.1%}" if not np.isnan(acc) else f"{name}: N/A")
        print(f"    {sub}: {', '.join(parts)}")

    means = {}
    for name in clf_names:
        vals = [results[s][name] for s in results if not np.isnan(results[s][name])]
        means[name] = np.mean(vals) if vals else np.nan

    print(f"    {'Mean':<8}: ", end="")
    print(", ".join(
        f"{n}: {a:.1%}" if not np.isnan(a) else f"{n}: N/A"
        for n, a in means.items()
    ))

    return means


# ═══════════════════════════════════════════════════════════════════
# Training & saving
# ═══════════════════════════════════════════════════════════════════

def train_final_model(X, y, model_type="mi"):
    """Train the final production model (ensemble).

    Args:
        X: feature matrix
        y: labels
        model_type: "mi" or "blink"

    Returns:
        pipeline: trained sklearn pipeline with predict_proba
    """
    if model_type == "mi":
        base_clfs = {
            "lda": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]),
            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                            random_state=RANDOM_STATE)),
            ]),
            "rf": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200, max_depth=6, min_samples_leaf=2,
                    random_state=RANDOM_STATE)),
            ]),
        }
    else:
        base_clfs = {
            "lda": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]),
            "rf": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=2,
                    random_state=RANDOM_STATE)),
            ]),
        }

    ensemble = build_ensemble(base_clfs, voting="soft")
    ensemble.fit(X, y)
    return ensemble


def _default_output_path(model_kind: str = "legacy_2class", participant_id: int | None = None) -> Path:
    """Resolve default bundle location by model kind."""
    if model_kind == "personalized":
        if participant_id is None:
            raise ValueError("participant_id is required when model_kind='personalized'")
        return MODELS_DIR / PERSONALIZED_BUNDLE_SUBDIR / f"sub{participant_id:02d}.pkl"
    if model_kind == "base_4class":
        return MODELS_DIR / BASE_4CLASS_BUNDLE_FILENAME
    return MODELS_DIR / DEFAULT_BUNDLE_FILENAME


def bundle_class_names(bundle: dict) -> list[str]:
    """Return canonical MI class names from any bundle shape."""
    names = bundle.get("class_names")
    if isinstance(names, list) and names:
        return [str(n) for n in names]

    # Legacy 2-class fallback
    return ["left_motor_imagery", "right_motor_imagery"]


def save_model_bundle(
    mi_pipeline,
    mi_csp_W,
    blink_pipeline,
    subjects_list,
    *,
    class_names: list[str] | None = None,
    blink_class_names: list[str] | None = None,
    model_version: str | None = None,
    model_kind: str = "legacy_2class",
    mi_label_map: dict | None = None,
    participant_id: int | None = None,
    output_path: Path | str | None = None,
    extra_info: dict | None = None,
):
    """Save all models as a single pickle bundle.

    Backward-compatible defaults preserve the legacy bundle shape.
    """
    out_path = Path(output_path) if output_path else _default_output_path(
        model_kind=model_kind,
        participant_id=participant_id,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if model_version is None:
        if model_kind == "base_4class":
            model_version = BASE_4CLASS_MODEL_VERSION
        elif model_kind == "personalized":
            model_version = PERSONALIZED_MODEL_VERSION
        else:
            model_version = LEGACY_MODEL_VERSION

    bundle = {
        "model_version": model_version,
        "model_kind": model_kind,
        "mi_pipeline": mi_pipeline,
        "mi_csp_W": mi_csp_W,
        "mi_ch_indices": list(range(4)),
        "blink_pipeline": blink_pipeline,
        "blink_ch_indices": list(range(4)),
        "sampling_rate": 256,
        "ch_names": ["TP9", "AF7", "AF8", "TP10"],
        "training_date": datetime.now().isoformat(),
        "n_subjects": len(subjects_list),
        "subjects": subjects_list,
        "class_names": class_names or ["left_motor_imagery", "right_motor_imagery"],
        "blink_class_names": blink_class_names or ["non_blink", "intentional_blink"],
    }

    if mi_label_map is not None:
        bundle["mi_label_map"] = mi_label_map
    if participant_id is not None:
        bundle["participant_id"] = int(participant_id)
    if extra_info:
        bundle.update(extra_info)

    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    return out_path


def load_model_bundle(path=None):
    """Load a saved model bundle with backward-compatible defaults."""
    path = Path(path) if path else (MODELS_DIR / DEFAULT_BUNDLE_FILENAME)
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    # Backfill schema for legacy bundles.
    bundle.setdefault("model_version", LEGACY_MODEL_VERSION)
    bundle.setdefault("model_kind", "legacy_2class")
    bundle.setdefault("class_names", ["left_motor_imagery", "right_motor_imagery"])
    bundle.setdefault("blink_class_names", ["non_blink", "intentional_blink"])
    return bundle
