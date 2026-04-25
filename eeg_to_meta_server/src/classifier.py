import random

LABELS = ["LEFT", "RIGHT", "FORWARD", "BACKWARD"]


def classify(window, channels):
    """Classify an EEG window.

    Args:
        window: numpy array of shape (n_channels, n_samples)
        channels: list of EEG channel indices

    Returns:
        (label, confidence) where label is one of LABELS
    """
    label = random.choice(LABELS)
    confidence = round(random.uniform(0.5, 1.0), 2)
    return label, confidence
