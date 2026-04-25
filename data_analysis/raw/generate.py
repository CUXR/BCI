"""Generate per-subject *raw* visualization plots.

Produces three plots per subject directly from the CSVs in
``data/Muse_Data/Phase*/sub*/`` — no preprocessing beyond windowing:

  plot_01_continuous.png        — full-session 4-channel EEG with trial spans
  plot_02_averaged_epochs.png   — mean ± std epochs per trial type, per channel
  plot_04_blinks.png            — zoomed view of the blink block with markers

Outputs land under ``data_analysis/raw/outputs/<subject_key>/``. The
``data/`` tree is treated as read-only so the Drive download stays a
clean mirror.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import (
    CH_COLS,
    CH_NAMES,
    TRIAL_COLORS,
    find_subject_data,
    load_subject,
    output_dir,
)

EPOCH_DURATION = 5.0
TRIAL_TYPES = ["motor_imagery_left", "motor_imagery_right", "blink_intentional"]


def plot_continuous(eeg, trials, label, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"{label} - Continuous EEG with Trial Markers",
                 fontsize=14, fontweight="bold")
    t0 = eeg["timestamp"].iloc[0]
    t_rel = eeg["timestamp"] - t0

    for i, (col, name) in enumerate(zip(CH_COLS, CH_NAMES)):
        axes[i].plot(t_rel, eeg[col], linewidth=0.3, color="black", alpha=0.7)
        axes[i].set_ylabel(f"{name} (uV)")
        for _, trial in trials.iterrows():
            ts = trial["start_time"] - t0
            te = trial["end_time"] - t0
            color = TRIAL_COLORS.get(trial["trial_type"])
            if color:
                axes[i].axvspan(ts, te, alpha=0.15, color=color)

    axes[-1].set_xlabel("Time (s)")
    legend_elements = [Patch(facecolor=c, alpha=0.3, label=k.replace("_", " "))
                       for k, c in TRIAL_COLORS.items()]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_averaged_epochs(eeg, trials, fs, label, out_path):
    epoch_samples = int(EPOCH_DURATION * fs)
    fig, axes = plt.subplots(len(TRIAL_TYPES), 4, figsize=(18, 10))
    fig.suptitle(f"{label} - Averaged EEG Epochs by Trial Type",
                 fontsize=14, fontweight="bold")

    for row, ttype in enumerate(TRIAL_TYPES):
        subset = trials[trials["trial_type"] == ttype]
        epochs = []
        for _, trial in subset.iterrows():
            mask = (eeg["timestamp"] >= trial["start_time"]) & (
                eeg["timestamp"] < trial["start_time"] + EPOCH_DURATION)
            epoch = eeg.loc[mask, CH_COLS].values
            if len(epoch) >= epoch_samples * 0.8:
                if len(epoch) > epoch_samples:
                    epoch = epoch[:epoch_samples]
                elif len(epoch) < epoch_samples:
                    epoch = np.pad(
                        epoch,
                        ((0, epoch_samples - len(epoch)), (0, 0)),
                        mode="edge",
                    )
                epochs.append(epoch)

        if not epochs:
            for col_idx in range(4):
                axes[row][col_idx].text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    transform=axes[row][col_idx].transAxes,
                )
            continue

        epochs = np.array(epochs)
        mean_epoch = np.mean(epochs, axis=0)
        std_epoch = np.std(epochs, axis=0)
        t_epoch = np.linspace(0, EPOCH_DURATION, epoch_samples)
        color = TRIAL_COLORS[ttype]

        for col_idx, name in enumerate(CH_NAMES):
            ax = axes[row][col_idx]
            ax.plot(t_epoch, mean_epoch[:, col_idx], color=color, linewidth=1.5)
            ax.fill_between(
                t_epoch,
                mean_epoch[:, col_idx] - std_epoch[:, col_idx],
                mean_epoch[:, col_idx] + std_epoch[:, col_idx],
                alpha=0.2, color=color,
            )
            for ep in epochs:
                ax.plot(t_epoch, ep[:, col_idx], alpha=0.12,
                        linewidth=0.4, color=color)
            if row == 0:
                ax.set_title(name)
            if col_idx == 0:
                ax.set_ylabel(f"{ttype.replace('_', ' ')}\n(uV)")
            if row == len(TRIAL_TYPES) - 1:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_blinks(eeg, trials, label, out_path):
    blink_trials = trials[trials["trial_type"] == "blink_intentional"]
    if len(blink_trials) == 0:
        return False

    fig, axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"{label} - Blink Trials with Markers",
                 fontsize=14, fontweight="bold")

    blink_start = blink_trials["start_time"].min() - 1
    blink_end = blink_trials["end_time"].max() + 1
    mask = (eeg["timestamp"] >= blink_start) & (eeg["timestamp"] <= blink_end)
    blink_eeg = eeg.loc[mask].copy()
    t0 = blink_start

    for i, (col, name) in enumerate(zip(CH_COLS, CH_NAMES)):
        ax = axes[i]
        t_rel = blink_eeg["timestamp"] - t0
        ax.plot(t_rel, blink_eeg[col], linewidth=0.5, color="black", alpha=0.8)

        for _, trial in blink_trials.iterrows():
            ts = trial["start_time"] - t0
            te = trial["end_time"] - t0
            ax.axvspan(ts, te, alpha=0.12, color="green")
            ax.axvline(ts, color="blue", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axvline(te, color="orange", linewidth=0.8, linestyle="--", alpha=0.6)

            em = trial["event_marker"]
            if pd.notna(em):
                ax.axvline(float(em) - t0, color="red", linewidth=1.2,
                           linestyle="-", alpha=0.8)

        ax.set_ylabel(f"{name} (uV)")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    legend_elements = [
        Line2D([0], [0], color="blue", linestyle="--", linewidth=1, label="Trial start"),
        Line2D([0], [0], color="orange", linestyle="--", linewidth=1, label="Trial end"),
        Line2D([0], [0], color="red", linestyle="-", linewidth=1.5, label="Event marker"),
        Patch(facecolor="green", alpha=0.2, label="Trial window"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main():
    subjects = find_subject_data()
    print(f"Found {len(subjects)} subjects: {list(subjects.keys())}")

    for sub_id, info in subjects.items():
        print(f"\n[{sub_id}] {info['label']}")
        eeg, trials, fs = load_subject(info)
        out = output_dir(__file__, sub_id)

        plot_continuous(eeg, trials, info["label"], out / "plot_01_continuous.png")
        print("  plot_01_continuous.png")

        plot_averaged_epochs(eeg, trials, fs, info["label"],
                             out / "plot_02_averaged_epochs.png")
        print("  plot_02_averaged_epochs.png")

        if plot_blinks(eeg, trials, info["label"], out / "plot_04_blinks.png"):
            print("  plot_04_blinks.png")
        else:
            print("  plot_04_blinks.png — skipped (no blink trials)")


if __name__ == "__main__":
    main()
