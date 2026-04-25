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


def save_model_bundle(mi_pipeline, mi_csp_W, blink_pipeline,
                      subjects_list, extra_info=None):
    """Save all models as a single pickle bundle.

    Returns:
        path to saved file
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
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
    }

    if extra_info:
        bundle.update(extra_info)

    out_path = MODELS_DIR / "realtime_models.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    return out_path


def load_model_bundle(path=None):
    """Load a saved model bundle."""
    path = path or (MODELS_DIR / "realtime_models.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
