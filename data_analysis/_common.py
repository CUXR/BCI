"""Shared discovery + loading utilities for data_analysis scripts.

The generation logic itself lives in raw/, signal_level/, and ml/ — this
module only owns the parts that would otherwise be copy-pasted across them
(subject discovery, CSV loading, channel/trial constants).
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
CH_COLS = ["EEG_0", "EEG_1", "EEG_2", "EEG_3"]

MI_TYPES = ["motor_imagery_left", "motor_imagery_right"]
TRIAL_COLORS = {
    "motor_imagery_left": "blue",
    "motor_imagery_right": "red",
    "blink_intentional": "green",
}


def find_subject_data(data_dir=DATA_DIR):
    """Recursively discover sub*/ directories with EEG + trial CSVs.

    Walks ``data_dir`` for any ``sub*`` directory containing both an
    ``eeg_data_*.csv`` and a ``trial_log_*.csv``. The Drive download
    expands to ``data/Muse_Data/Phase{1,2}/sub*/``, so recursion lets the
    same code handle either the nested or a flattened layout. Subject keys
    disambiguate with the parent folder when a bare ``subNN`` would
    collide (e.g. ``Phase1_sub05`` vs ``Phase2_sub05``).
    """
    subjects = {}
    for sub_dir in sorted(data_dir.rglob("sub*")):
        if not sub_dir.is_dir():
            continue
        eeg_files = sorted(sub_dir.glob("eeg_data_*.csv"))
        trial_files = sorted(sub_dir.glob("trial_log_*.csv"))
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
    """Return (eeg_df, trials_df, sampling_rate) for one subject."""
    eeg = pd.read_csv(info["eeg"])
    trials = pd.read_csv(info["trials"])
    dt = np.diff(eeg["timestamp"].values[:1000])
    fs = 1.0 / np.median(dt)
    return eeg, trials, fs


def output_dir(script_file, subject_key):
    """Return a per-subject outputs/<key>/ dir under the calling script.

    The directory is created if missing. Each generator writes its plots
    here so generated artifacts never leak into data/ (which mirrors the
    Drive download).
    """
    out = Path(script_file).resolve().parent / "outputs" / subject_key
    out.mkdir(parents=True, exist_ok=True)
    return out
