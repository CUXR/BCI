import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from classifier import classify, LABELS


def test_classify_returns_valid_label_and_confidence():
    window = np.random.randn(4, 256)
    label, confidence = classify(window, [0, 1, 2, 3])
    assert label in LABELS
    assert 0.0 <= confidence <= 1.0


def test_classify_output_types():
    window = np.random.randn(4, 256)
    label, confidence = classify(window, [0, 1, 2, 3])
    assert isinstance(label, str)
    assert isinstance(confidence, float)
