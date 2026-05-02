"""Dataset loaders for the 4-class navigation MI pipeline.

Reads the schema produced by `pipeline.collector` (per-phase trial rows
+ named EEG channel columns TP9 / AF7 / AF8 / TP10) and produces:

    mi_epochs:    list of (4, n_samples) arrays, fixed length
    mi_labels:    list of int (0..3 — see LABEL_MAP_4CLASS)
    mi_subjects:  list of subject ids parallel to mi_epochs
    blink_pos_*:  positive blink epochs (event-centred ±BLINK_EPOCH_S/2)
    blink_neg_*:  random non-blink windows sampled from MI trials

The legacy 2-class loaders in `dataset.py` are untouched so existing
psychopy data keeps training the old bundle. This module is a deliberate
sibling that recognises only the new collector schema.

Used by:
    * ml_pipeline/train.py  (when --variant 4class)
    * pipeline/personalize.py  (per-participant fits — uses a slimmed
      copy at the moment; will switch to this module once both surfaces
      are exercised end-to-end)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from config import (                                                # type: ignore[import-not-found]
    ARTIFACT_PTP_MI,
    BLINK_EPOCH_S,
    BLINK_NEG_PER_MI,
    DATA_DIRS,
    FLAT_THRESHOLD_UV,
    LABEL_MAP_4CLASS,
    MI_EPOCH_FIXED_S,
    MI_EPOCH_MIN_S,
    MI_TYPES_4CLASS,
    RANDOM_STATE,
    TARGET_SFREQ,
)
from preprocessing import (                                         # type: ignore[import-not-found]
    interpolate_nans,
    preprocess_blink_epoch,
    preprocess_mi_epoch,
    score_channel_quality_epoch,
)


# Channel column names written by pipeline.collector.
COLLECTOR_CH_COLS = ["TP9", "AF7", "AF8", "TP10"]

# Trial-type string for blink trials (kept here to avoid importing the
# pipeline package — circular dep with personalize).
BLINK_TRIAL_TYPE = "blink_intentional"


# ── Discovery / loading ────────────────────────────────────────────


@dataclass
class SessionPaths:
    sub_id: str
    session_dir: Path
    eeg_csv: Path
    trial_log_csv: Path


def find_all_subjects(data_roots: Optional[Iterable[Path]] = None) -> dict[str, list[SessionPaths]]:
    """Scan one or more data roots for sub<NN>/session_*/ pairs.

    Returns:
        {sub_id: [SessionPaths, ...]}
    Subject ids are the directory names (e.g. "sub05").
    """
    roots: list[Path]
    if data_roots is None:
        roots = [Path(d) for d in DATA_DIRS]
    else:
        roots = [Path(d) for d in data_roots]

    out: dict[str, list[SessionPaths]] = {}
    for root in roots:
        if not root.exists():
            continue
        for sub_dir in sorted(root.glob("sub*")):
            if not sub_dir.is_dir():
                continue
            sessions: list[SessionPaths] = []
            for session_dir in sorted(sub_dir.glob("session_*")):
                eeg = session_dir / "eeg_data.csv"
                trial_log = session_dir / "trial_log.csv"
                if eeg.is_file() and trial_log.is_file():
                    sessions.append(SessionPaths(
                        sub_id=sub_dir.name,
                        session_dir=session_dir,
                        eeg_csv=eeg,
                        trial_log_csv=trial_log,
                    ))
            if sessions:
                out.setdefault(sub_dir.name, []).extend(sessions)
    return out


def load_session(paths: SessionPaths) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Read one collector session into (eeg_df, trials_df, estimated_sfreq)."""
    eeg_df = pd.read_csv(paths.eeg_csv)
    trials_df = pd.read_csv(paths.trial_log_csv)
    for col in COLLECTOR_CH_COLS:
        if col in eeg_df.columns:
            eeg_df[col] = pd.to_numeric(eeg_df[col], errors="coerce")
    ts = eeg_df["timestamp"].values[:2000]
    dt = np.diff(ts)
    dt = dt[dt > 0]
    sfreq = float(1.0 / np.median(dt)) if len(dt) > 0 else float(TARGET_SFREQ)
    return eeg_df, trials_df, sfreq


def aggregate_trials(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-phase rows into one row per (run_index, trial_index, trial_type).

    Produces columns trial_type / run_index / trial_index / start_time /
    end_time / event_marker. Times are sourced from `ts_recv` so they
    line up with the EEG `timestamp` column.
    """
    if trials_df.empty:
        return pd.DataFrame(columns=[
            "trial_type", "run_index", "trial_index",
            "start_time", "end_time", "event_marker",
        ])

    rows = []
    grouped = trials_df.groupby(["run_index", "trial_index", "trial_type"], sort=True)
    for (run_idx, trial_idx, trial_type), group in grouped:
        phases = {
            row["phase"]: float(row["ts_recv"])
            for _, row in group.iterrows()
            if pd.notna(row.get("ts_recv"))
        }
        cue_t = phases.get("cue")
        start_t = phases.get("start", cue_t)
        end_t = phases.get("end")
        if start_t is None:
            continue
        if end_t is None:
            end_t = start_t + MI_EPOCH_FIXED_S

        rows.append({
            "trial_type": trial_type,
            "run_index": int(run_idx),
            "trial_index": int(trial_idx),
            "start_time": start_t,
            "end_time": end_t,
            "event_marker": phases.get("start", cue_t or start_t),
        })
    return pd.DataFrame(rows)


# ── Segment utilities ──────────────────────────────────────────────


def _segment(eeg_df: pd.DataFrame, t0: float, t1: float) -> np.ndarray:
    mask = (eeg_df["timestamp"] >= t0) & (eeg_df["timestamp"] <= t1)
    arr = eeg_df.loc[mask, COLLECTOR_CH_COLS].values.astype(np.float64)
    if arr.size == 0:
        return np.zeros((len(COLLECTOR_CH_COLS), 0))
    for c in range(arr.shape[1]):
        arr[:, c] = interpolate_nans(arr[:, c])
    return arr.T  # (n_channels, n_samples)


def _resample_if_needed(data_2d: np.ndarray, sfreq_orig: float) -> tuple[np.ndarray, float]:
    target = float(TARGET_SFREQ)
    if data_2d.shape[1] < 4:
        return data_2d, target
    if abs(sfreq_orig - target) < 2:
        return data_2d, target
    import mne                                                      # noqa: PLC0415
    mne.set_log_level("ERROR")
    info = mne.create_info(
        ch_names=[f"ch{i}" for i in range(data_2d.shape[0])],
        sfreq=sfreq_orig, ch_types="eeg",
    )
    raw = mne.io.RawArray(data_2d, info, verbose=False)
    raw.resample(target, verbose=False)
    return raw.get_data(), target


# ── Epoch extraction ───────────────────────────────────────────────


def extract_mi_epochs_4class(
    eeg_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    sfreq: float,
) -> tuple[list[np.ndarray], list[int], list[dict]]:
    """4-class MI epoch extraction (forward / backward / rotate-left / rotate-right)."""
    aggregated = aggregate_trials(trials_df)
    mi = aggregated[aggregated["trial_type"].isin(MI_TYPES_4CLASS)]
    if mi.empty:
        return [], [], []

    fixed_samples = int(MI_EPOCH_FIXED_S * TARGET_SFREQ)
    min_samples = int(MI_EPOCH_MIN_S * sfreq)

    epochs: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict] = []

    for _, trial in mi.iterrows():
        seg = _segment(eeg_df, trial["start_time"], trial["end_time"])
        if seg.shape[1] < min_samples:
            continue
        seg, actual_sfreq = _resample_if_needed(seg, sfreq)
        if seg.shape[1] > fixed_samples:
            seg = seg[:, :fixed_samples]
        elif seg.shape[1] < fixed_samples:
            pad = fixed_samples - seg.shape[1]
            seg = np.pad(seg, ((0, 0), (0, pad)), mode="edge")

        cleaned = preprocess_mi_epoch(seg, actual_sfreq)
        ch_quality = score_channel_quality_epoch(cleaned, actual_sfreq)
        if int(np.sum(ch_quality)) < max(2, len(COLLECTOR_CH_COLS) // 2):
            continue
        good = cleaned[ch_quality]
        ptp = float(np.ptp(good, axis=1).max())
        if ptp > ARTIFACT_PTP_MI:
            continue
        if np.any(np.std(cleaned[ch_quality], axis=1) < FLAT_THRESHOLD_UV):
            continue

        epochs.append(cleaned)
        labels.append(LABEL_MAP_4CLASS[trial["trial_type"]])
        meta.append({
            "trial_type": trial["trial_type"],
            "run_index": int(trial["run_index"]),
            "trial_index": int(trial["trial_index"]),
            "ptp_uv": ptp,
        })

    return epochs, labels, meta


def extract_blink_pos_epochs(
    eeg_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    sfreq: float,
    window_s: float = BLINK_EPOCH_S,
) -> list[np.ndarray]:
    """Positive blink epochs centred on each `intentional_blink` event."""
    aggregated = aggregate_trials(trials_df)
    blinks = aggregated[aggregated["trial_type"] == BLINK_TRIAL_TYPE]
    if blinks.empty:
        return []

    target_samples = int(window_s * TARGET_SFREQ)
    epochs: list[np.ndarray] = []

    for _, trial in blinks.iterrows():
        center = float(trial["event_marker"])
        seg = _segment(eeg_df, center - window_s / 2, center + window_s / 2)
        if seg.shape[1] < 10:
            continue
        seg, actual_sfreq = _resample_if_needed(seg, sfreq)
        if seg.shape[1] > target_samples:
            seg = seg[:, :target_samples]
        elif seg.shape[1] < target_samples:
            pad = target_samples - seg.shape[1]
            seg = np.pad(seg, ((0, 0), (0, pad)), mode="edge")
        epochs.append(preprocess_blink_epoch(seg, actual_sfreq))
    return epochs


def extract_blink_neg_epochs(
    eeg_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    sfreq: float,
    window_s: float = BLINK_EPOCH_S,
    rng: Optional[np.random.Generator] = None,
) -> list[np.ndarray]:
    """Sample non-blink windows from inside MI trials (no overlap)."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    aggregated = aggregate_trials(trials_df)
    mi = aggregated[aggregated["trial_type"].isin(MI_TYPES_4CLASS)]
    target_samples = int(window_s * TARGET_SFREQ)

    epochs: list[np.ndarray] = []
    for _, trial in mi.iterrows():
        seg = _segment(eeg_df, trial["start_time"], trial["end_time"])
        if seg.shape[1] < target_samples + 10:
            continue
        seg, actual_sfreq = _resample_if_needed(seg, sfreq)
        if seg.shape[1] < target_samples + 10:
            continue
        max_start = seg.shape[1] - target_samples
        starts: list[int] = []
        attempts = 0
        while len(starts) < BLINK_NEG_PER_MI and attempts < 50:
            s = int(rng.integers(0, max_start))
            if all(abs(s - ex) >= target_samples for ex in starts):
                starts.append(s)
            attempts += 1
        for s in starts:
            window = seg[:, s : s + target_samples]
            epochs.append(preprocess_blink_epoch(window, actual_sfreq))
    return epochs


# ── Cross-subject builder ──────────────────────────────────────────


def build_dataset_4class(
    data_roots: Optional[Iterable[Path]] = None,
    verbose: bool = True,
) -> Optional[dict]:
    """Load every collector session under data_roots and assemble epochs.

    Returns a dict shaped like the 2-class `dataset.build_dataset` so
    `train.train_blink` can keep its current contract; new keys are
    `mi_class_names` (canonical labels for the 4 MI head outputs) and
    `mi_label_map` (trial_type → int).
    """
    subjects = find_all_subjects(data_roots)
    if not subjects:
        if verbose:
            print("No 4-class collector sessions found in:")
            for d in (data_roots or DATA_DIRS):
                print(f"  {d}")
        return None

    if verbose:
        print(f"Found {len(subjects)} subject(s) with 4-class data: "
              f"{sorted(subjects.keys())}")

    mi_epochs_all: list[np.ndarray] = []
    mi_labels_all: list[int] = []
    mi_subjects_all: list[str] = []

    blink_pos_all: list[np.ndarray] = []
    blink_pos_subjects: list[str] = []
    blink_neg_all: list[np.ndarray] = []
    blink_neg_subjects: list[str] = []

    subjects_info: dict[str, dict] = {}

    for sub_id, sessions in subjects.items():
        sub_mi = 0
        sub_blink_pos = 0
        sub_blink_neg = 0
        for sp in sessions:
            eeg_df, trials_df, sfreq = load_session(sp)

            mi_e, mi_l, _ = extract_mi_epochs_4class(eeg_df, trials_df, sfreq)
            mi_epochs_all.extend(mi_e)
            mi_labels_all.extend(mi_l)
            mi_subjects_all.extend([sub_id] * len(mi_e))
            sub_mi += len(mi_e)

            bp = extract_blink_pos_epochs(eeg_df, trials_df, sfreq)
            bn = extract_blink_neg_epochs(eeg_df, trials_df, sfreq)
            blink_pos_all.extend(bp)
            blink_pos_subjects.extend([sub_id] * len(bp))
            blink_neg_all.extend(bn)
            blink_neg_subjects.extend([sub_id] * len(bn))
            sub_blink_pos += len(bp)
            sub_blink_neg += len(bn)

            if verbose:
                counts = {n: sum(1 for l in mi_l if l == i)
                          for i, n in enumerate(["fwd", "bwd", "rotL", "rotR"])}
                print(f"  {sub_id}/{sp.session_dir.name}: "
                      f"MI={len(mi_e)} ({counts}), "
                      f"Blink={len(bp)}+/{len(bn)}- "
                      f"sfreq≈{sfreq:.1f}Hz")

        subjects_info[sub_id] = {
            "n_sessions": len(sessions),
            "n_mi_epochs": sub_mi,
            "n_blink_pos": sub_blink_pos,
            "n_blink_neg": sub_blink_neg,
        }

    blink_subjects_all = blink_pos_subjects + blink_neg_subjects

    return {
        "mi_epochs": mi_epochs_all,
        "mi_labels": mi_labels_all,
        "mi_subjects": mi_subjects_all,
        "mi_class_names": ["mi_forward", "mi_backward", "mi_rotate_left", "mi_rotate_right"],
        "mi_label_map": dict(LABEL_MAP_4CLASS),
        "blink_pos_epochs": blink_pos_all,
        "blink_neg_epochs": blink_neg_all,
        "blink_subjects": blink_subjects_all,
        "subjects_info": subjects_info,
    }
