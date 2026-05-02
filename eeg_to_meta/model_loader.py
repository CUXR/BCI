"""Load and validate the trained .pkl model bundle.

The bundle produced by ml_pipeline/train.py contains:
    mi_pipeline      : VotingClassifier  (left=0 vs right=1)
    mi_csp_W         : ndarray (4, 4)    CSP spatial filter
    mi_ch_indices    : [0, 1, 2, 3]
    blink_pipeline   : VotingClassifier  (non-blink=0 vs blink=1)
    blink_ch_indices : [0, 1, 2, 3]
    sampling_rate    : 256
    ch_names         : ['TP9', 'AF7', 'AF8', 'TP10']
"""

import pickle
import logging
from pathlib import Path

import numpy as np

from config import (
    MODEL_PATH, MI_TOTAL_FEATURES, BLINK_TOTAL_FEATURES, N_CHANNELS, SFREQ,
)

log = logging.getLogger(__name__)


class ModelBundle:
    """Wrapper around the serialised model bundle."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else MODEL_PATH
        self.bundle: dict | None = None
        self.mi_pipeline = None
        self.mi_csp_W: np.ndarray | None = None
        self.blink_pipeline = None

    # ── loading ─────────────────────────────────────────────────────

    def load(self) -> bool:
        if not self.path.exists():
            log.error("Model file not found: %s", self.path)
            return False

        try:
            with open(self.path, "rb") as fh:
                self.bundle = pickle.load(fh)
        except Exception as exc:
            log.error("Failed to unpickle model: %s", exc)
            return False

        self.mi_pipeline = self.bundle.get("mi_pipeline")
        self.mi_csp_W = self.bundle.get("mi_csp_W")
        self.blink_pipeline = self.bundle.get("blink_pipeline")

        self._print_info()
        ok = self._validate()
        return ok

    # ── prediction helpers ──────────────────────────────────────────

    def predict_mi(self, features: np.ndarray):
        """Predict motor imagery class + probabilities.

        Args:
            features: (1, MI_TOTAL_FEATURES) or (MI_TOTAL_FEATURES,)

        Returns:
            (predicted_label: int, probas: np.ndarray)
            label 0 = left, 1 = right
        """
        X = features.reshape(1, -1)
        if X.shape[1] != MI_TOTAL_FEATURES:
            log.warning(
                "MI feature shape mismatch: got %d, expected %d",
                X.shape[1], MI_TOTAL_FEATURES,
            )
        probas = self.mi_pipeline.predict_proba(X)[0]
        return int(np.argmax(probas)), probas

    def predict_blink(self, features: np.ndarray):
        """Predict blink vs non-blink + probabilities.

        Args:
            features: (1, BLINK_TOTAL_FEATURES) or (BLINK_TOTAL_FEATURES,)

        Returns:
            (predicted_label: int, probas: np.ndarray)
            label 0 = non-blink, 1 = blink
        """
        X = features.reshape(1, -1)
        if X.shape[1] != BLINK_TOTAL_FEATURES:
            log.warning(
                "Blink feature shape mismatch: got %d, expected %d",
                X.shape[1], BLINK_TOTAL_FEATURES,
            )
        probas = self.blink_pipeline.predict_proba(X)[0]
        return int(np.argmax(probas)), probas

    def apply_csp(self, epoch_2d: np.ndarray) -> np.ndarray:
        """Apply CSP spatial filter to a (4, n_samples) MI epoch.

        Returns:
            csp_features: (n_components,) log-variance features.
        """
        if self.mi_csp_W is None:
            return np.zeros(N_CHANNELS)
        projected = self.mi_csp_W.T @ epoch_2d     # (n_comp, n_samples)
        return np.log(np.var(projected, axis=1) + 1e-12)

    # ── diagnostics ─────────────────────────────────────────────────

    def _print_info(self):
        b = self.bundle
        log.info("Model loaded from %s", self.path)
        # Do not log subject IDs from the bundle — they may be PII and end up in log aggregators.
        log.info(
            "  Trained %s on %d subject(s)",
            b.get("training_date", "?")[:10],
            b.get("n_subjects", 0),
        )
        log.info("  Sampling rate:  %d Hz", b.get("sampling_rate", 0))
        log.info("  Channels:       %s", b.get("ch_names", []))
        log.info("  MI features:    %d expected", MI_TOTAL_FEATURES)
        log.info("  Blink features: %d expected", BLINK_TOTAL_FEATURES)

        if self.mi_csp_W is not None:
            log.info("  CSP matrix:     %s", self.mi_csp_W.shape)
        else:
            log.warning("  CSP matrix:     MISSING — MI accuracy will degrade")

    def _validate(self) -> bool:
        ok = True

        if self.mi_pipeline is None:
            log.error("  mi_pipeline is None — MI classification disabled")
            ok = False
        else:
            expected = getattr(self.mi_pipeline, "n_features_in_", None)
            if expected and expected != MI_TOTAL_FEATURES:
                log.error(
                    "  MI model trained on %d features but config says %d",
                    expected, MI_TOTAL_FEATURES,
                )
                ok = False

        if self.blink_pipeline is None:
            log.error("  blink_pipeline is None — blink detection disabled")
            ok = False
        else:
            expected = getattr(self.blink_pipeline, "n_features_in_", None)
            if expected and expected != BLINK_TOTAL_FEATURES:
                log.error(
                    "  Blink model trained on %d features but config says %d",
                    expected, BLINK_TOTAL_FEATURES,
                )
                ok = False

        sr = self.bundle.get("sampling_rate", 0)
        if sr != SFREQ:
            log.error("  Model expects %d Hz but config says %d Hz", sr, SFREQ)
            ok = False

        return ok
