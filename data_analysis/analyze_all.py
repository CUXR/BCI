"""Comprehensive BCI EEG analysis across all subjects.

Generates per-subject summary plots and cross-subject comparisons.
Based on sub01 analysis scripts (sanity_check_plot.py, sanity_check_bandpower.py,
plot_blinks_with_markers.py).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.signal import welch
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
CH_COLS = ["EEG_0", "EEG_1", "EEG_2", "EEG_3"]

BANDS = {
    "Delta (1-4)": (1, 4),
    "Theta (4-8)": (4, 8),
    "Alpha (8-13)": (8, 13),
    "Beta (13-30)": (13, 30),
    "Gamma (30-50)": (30, 50),
}
BAND_COLORS = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#b07aa1"]

TRIAL_COLORS = {
    "motor_imagery_left": "blue",
    "motor_imagery_right": "red",
    "blink_intentional": "green",
}

MI_TYPES = ["motor_imagery_left", "motor_imagery_right"]


def find_subject_data(data_dir):
    """Discover all subject directories with EEG data (recursive).

    Walks ``data_dir`` for any ``sub*`` directory containing both an
    ``eeg_data_*.csv`` and a ``trial_log_*.csv``. The download from Google
    Drive expands to ``data/Muse_Data/Phase{1,2}/sub*/``, so a recursive
    search lets the same code work whether the user keeps the nested
    layout or flattens it. Subject keys include the parent folder when a
    bare ``subNN`` name would collide (e.g. Phase1/sub05 vs Phase2/sub05).
    """
    subjects = {}
    for sub_dir in sorted(data_dir.rglob("sub*")):
        if not sub_dir.is_dir():
            continue
        eeg_files = list(sub_dir.glob("eeg_data_*.csv"))
        trial_files = list(sub_dir.glob("trial_log_*.csv"))
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
    """Load EEG and trial data for a subject."""
    eeg = pd.read_csv(info["eeg"])
    trials = pd.read_csv(info["trials"])
    dt = np.diff(eeg["timestamp"].values[:1000])
    fs = 1.0 / np.median(dt)
    return eeg, trials, fs


def compute_bandpower(signal, fs, fmin, fmax):
    nperseg = min(int(fs * 2), len(signal))
    if nperseg < int(fs * 0.5):
        return np.nan
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])


def compute_relative_bandpower(signal, fs, fmin, fmax):
    nperseg = min(int(fs * 2), len(signal))
    if nperseg < int(fs * 0.5):
        return np.nan
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    band_idx = (freqs >= fmin) & (freqs <= fmax)
    total_idx = (freqs >= 1) & (freqs <= 50)
    total_power = np.trapz(psd[total_idx], freqs[total_idx])
    if total_power == 0:
        return np.nan
    return np.trapz(psd[band_idx], freqs[band_idx]) / total_power


# ============================================================
# Discover subjects
# ============================================================
subjects = find_subject_data(DATA_DIR)
print(f"Found {len(subjects)} subjects: {list(subjects.keys())}")

all_bandpowers = {}  # subject -> {ttype -> {band -> {ch -> [values]}}}
all_rel_bandpowers = {}
all_summary = []

for sub_id, info in subjects.items():
    print(f"\n{'='*60}")
    print(f"Processing {info['label']}")
    print(f"{'='*60}")

    eeg, trials, fs = load_subject(info)
    out_dir = info["dir"]

    n_left = len(trials[trials["trial_type"] == "motor_imagery_left"])
    n_right = len(trials[trials["trial_type"] == "motor_imagery_right"])
    n_blink = len(trials[trials["trial_type"] == "blink_intentional"])
    duration = eeg["timestamp"].iloc[-1] - eeg["timestamp"].iloc[0]

    print(f"  Samples: {len(eeg)}, Fs: {fs:.1f} Hz, Duration: {duration:.1f}s")
    print(f"  Trials: {n_left} left MI, {n_right} right MI, {n_blink} blinks")

    all_summary.append({
        "subject": sub_id,
        "label": info["label"],
        "n_samples": len(eeg),
        "fs": fs,
        "duration_s": duration,
        "n_left": n_left,
        "n_right": n_right,
        "n_blink": n_blink,
    })

    # ==== Plot 1: Continuous EEG with trial markers ====
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"{info['label']} - Continuous EEG with Trial Markers", fontsize=14, fontweight="bold")
    t_rel = eeg["timestamp"] - eeg["timestamp"].iloc[0]

    for i, (col, name) in enumerate(zip(CH_COLS, CH_NAMES)):
        axes[i].plot(t_rel, eeg[col], linewidth=0.3, color="black", alpha=0.7)
        axes[i].set_ylabel(f"{name} (uV)")
        for _, trial in trials.iterrows():
            ts = trial["start_time"] - eeg["timestamp"].iloc[0]
            te = trial["end_time"] - eeg["timestamp"].iloc[0]
            c = TRIAL_COLORS[trial["trial_type"]]
            axes[i].axvspan(ts, te, alpha=0.15, color=c)

    axes[-1].set_xlabel("Time (s)")
    legend_elements = [Patch(facecolor=c, alpha=0.3, label=k.replace("_", " "))
                       for k, c in TRIAL_COLORS.items()]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_01_continuous.png", dpi=150)
    plt.close()
    print(f"  Saved plot_01_continuous.png")

    # ==== Plot 2: Averaged epochs ====
    EPOCH_DURATION = 5.0
    epoch_samples = int(EPOCH_DURATION * fs)
    trial_types = ["motor_imagery_left", "motor_imagery_right", "blink_intentional"]

    fig, axes = plt.subplots(len(trial_types), 4, figsize=(18, 10))
    fig.suptitle(f"{info['label']} - Averaged EEG Epochs by Trial Type", fontsize=14, fontweight="bold")

    for row, ttype in enumerate(trial_types):
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
                    epoch = np.pad(epoch, ((0, epoch_samples - len(epoch)), (0, 0)), mode="edge")
                epochs.append(epoch)

        if not epochs:
            for col_idx in range(4):
                axes[row][col_idx].text(0.5, 0.5, "No data", ha="center", va="center",
                                        transform=axes[row][col_idx].transAxes)
            continue

        epochs = np.array(epochs)
        mean_epoch = np.mean(epochs, axis=0)
        std_epoch = np.std(epochs, axis=0)
        t_epoch = np.linspace(0, EPOCH_DURATION, epoch_samples)

        for col_idx, (col, name) in enumerate(zip(CH_COLS, CH_NAMES)):
            ax = axes[row][col_idx]
            ax.plot(t_epoch, mean_epoch[:, col_idx], color=TRIAL_COLORS[ttype], linewidth=1.5)
            ax.fill_between(t_epoch,
                            mean_epoch[:, col_idx] - std_epoch[:, col_idx],
                            mean_epoch[:, col_idx] + std_epoch[:, col_idx],
                            alpha=0.2, color=TRIAL_COLORS[ttype])
            for ep in epochs:
                ax.plot(t_epoch, ep[:, col_idx], alpha=0.12, linewidth=0.4, color=TRIAL_COLORS[ttype])
            if row == 0:
                ax.set_title(name)
            if col_idx == 0:
                ax.set_ylabel(f"{ttype.replace('_', ' ')}\n(uV)")
            if row == len(trial_types) - 1:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(out_dir / "plot_02_averaged_epochs.png", dpi=150)
    plt.close()
    print(f"  Saved plot_02_averaged_epochs.png")

    # ==== Plot 3: PSD by trial type ====
    fig, axes = plt.subplots(len(trial_types), 4, figsize=(18, 10))
    fig.suptitle(f"{info['label']} - Power Spectral Density by Trial Type", fontsize=14, fontweight="bold")

    for row, ttype in enumerate(trial_types):
        subset = trials[trials["trial_type"] == ttype]
        for col_idx, (col, name) in enumerate(zip(CH_COLS, CH_NAMES)):
            ax = axes[row][col_idx]
            all_psds = []
            for _, trial in subset.iterrows():
                mask = (eeg["timestamp"] >= trial["start_time"]) & (
                    eeg["timestamp"] <= trial["end_time"])
                segment = eeg.loc[mask, col].values
                if len(segment) > int(fs):
                    nperseg = min(int(fs), len(segment))
                    freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
                    all_psds.append(psd)
                    ax.semilogy(freqs, psd, alpha=0.3, linewidth=0.5, color=TRIAL_COLORS[ttype])

            if all_psds:
                min_len = min(len(p) for p in all_psds)
                all_psds_trimmed = [p[:min_len] for p in all_psds]
                mean_psd = np.mean(all_psds_trimmed, axis=0)
                ax.semilogy(freqs[:min_len], mean_psd, linewidth=2, color=TRIAL_COLORS[ttype])

            ax.set_xlim(0, 50)
            if row == 0:
                ax.set_title(name)
            if col_idx == 0:
                ax.set_ylabel(f"{ttype.replace('_', ' ')}\nPSD (uV^2/Hz)")
            if row == len(trial_types) - 1:
                ax.set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(out_dir / "plot_03_psd.png", dpi=150)
    plt.close()
    print(f"  Saved plot_03_psd.png")

    # ==== Plot 4: Blink trials zoomed ====
    blink_trials = trials[trials["trial_type"] == "blink_intentional"]
    if len(blink_trials) > 0:
        fig, axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
        fig.suptitle(f"{info['label']} - Blink Trials with Markers", fontsize=14, fontweight="bold")

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
                    ax.axvline(float(em) - t0, color="red", linewidth=1.2, linestyle="-", alpha=0.8)

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
        plt.savefig(out_dir / "plot_04_blinks.png", dpi=150)
        plt.close()
        print(f"  Saved plot_04_blinks.png")

    # ==== Plot 5: Left vs Right MI comparison ====
    fig, axes = plt.subplots(2, 4, figsize=(18, 6))
    fig.suptitle(f"{info['label']} - Left vs Right Motor Imagery", fontsize=14, fontweight="bold")

    for ttype_idx, ttype in enumerate(MI_TYPES):
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
                    epoch = np.pad(epoch, ((0, epoch_samples - len(epoch)), (0, 0)), mode="edge")
                epochs.append(epoch)

        if epochs:
            epochs = np.array(epochs)
            mean_epoch = np.mean(epochs, axis=0)
            t_epoch = np.linspace(0, EPOCH_DURATION, epoch_samples)
            for col_idx, (col, name) in enumerate(zip(CH_COLS, CH_NAMES)):
                ax = axes[ttype_idx][col_idx]
                ax.plot(t_epoch, mean_epoch[:, col_idx], color=TRIAL_COLORS[ttype], linewidth=1.5)
                if ttype_idx == 0:
                    ax.set_title(name)
                ax.set_ylabel(f"{'Left' if 'left' in ttype else 'Right'}\n(uV)")
                if ttype_idx == 1:
                    ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(out_dir / "plot_05_left_vs_right.png", dpi=150)
    plt.close()
    print(f"  Saved plot_05_left_vs_right.png")

    # ==== Compute band powers for cross-subject comparison ====
    all_bandpowers[sub_id] = {}
    all_rel_bandpowers[sub_id] = {}

    for ttype in MI_TYPES:
        subset = trials[trials["trial_type"] == ttype]
        all_bandpowers[sub_id][ttype] = {}
        all_rel_bandpowers[sub_id][ttype] = {}

        for bname, (fmin, fmax) in BANDS.items():
            all_bandpowers[sub_id][ttype][bname] = {}
            all_rel_bandpowers[sub_id][ttype][bname] = {}

            for col_idx, col in enumerate(CH_COLS):
                trial_abs = []
                trial_rel = []
                for _, trial in subset.iterrows():
                    mask = (eeg["timestamp"] >= trial["start_time"]) & (
                        eeg["timestamp"] <= trial["end_time"])
                    sig = eeg.loc[mask, col].values
                    bp = compute_bandpower(sig, fs, fmin, fmax)
                    rp = compute_relative_bandpower(sig, fs, fmin, fmax)
                    if not np.isnan(bp):
                        trial_abs.append(bp)
                    if not np.isnan(rp):
                        trial_rel.append(rp)

                all_bandpowers[sub_id][ttype][bname][CH_NAMES[col_idx]] = trial_abs
                all_rel_bandpowers[sub_id][ttype][bname][CH_NAMES[col_idx]] = trial_rel

    # ==== Plot 6: Absolute band power ====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{info['label']} - Absolute Band Power by Trial Type", fontsize=14, fontweight="bold")

    for ax_idx, ttype in enumerate(MI_TYPES):
        ax = axes[ax_idx]
        x = np.arange(len(CH_NAMES))
        width = 0.15
        for b_idx, (bname, (fmin, fmax)) in enumerate(BANDS.items()):
            powers = []
            stds = []
            for ch in CH_NAMES:
                vals = all_bandpowers[sub_id][ttype][bname][ch]
                powers.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            offset = (b_idx - 2) * width
            ax.bar(x + offset, powers, width, yerr=stds, capsize=3,
                   label=bname, color=BAND_COLORS[b_idx], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(CH_NAMES)
        ax.set_ylabel("Power (uV^2)")
        ax.set_title(ttype.replace("_", " ").title())
        ax.legend(fontsize=7, loc="upper right")
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_dir / "plot_06_bandpower_absolute.png", dpi=150)
    plt.close()
    print(f"  Saved plot_06_bandpower_absolute.png")

    # ==== Plot 7: Relative band power ====
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{info['label']} - Relative Band Power", fontsize=14, fontweight="bold")

    for ax_idx, ttype in enumerate(MI_TYPES):
        ax = axes[ax_idx]
        bottoms = np.zeros(len(CH_NAMES))
        for b_idx, bname in enumerate(BANDS):
            vals = [np.mean(all_rel_bandpowers[sub_id][ttype][bname][ch])
                    if all_rel_bandpowers[sub_id][ttype][bname][ch] else 0
                    for ch in CH_NAMES]
            ax.bar(CH_NAMES, vals, bottom=bottoms, label=bname,
                   color=BAND_COLORS[b_idx], alpha=0.85)
            bottoms += np.array(vals)
        ax.set_ylabel("Relative Power")
        ax.set_title(ttype.replace("_", " ").title())
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_dir / "plot_07_bandpower_relative.png", dpi=150)
    plt.close()
    print(f"  Saved plot_07_bandpower_relative.png")

    # ==== Plot 8: Left vs Right band comparison ====
    fig, axes = plt.subplots(1, len(BANDS), figsize=(20, 5))
    fig.suptitle(f"{info['label']} - Left vs Right MI Band Power", fontsize=14, fontweight="bold")

    for b_idx, (bname, (fmin, fmax)) in enumerate(BANDS.items()):
        ax = axes[b_idx]
        x = np.arange(len(CH_NAMES))
        width = 0.35
        for t_idx, ttype in enumerate(MI_TYPES):
            powers = []
            stds = []
            for ch in CH_NAMES:
                vals = all_bandpowers[sub_id][ttype][bname][ch]
                powers.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            offset = (t_idx - 0.5) * width
            color = "blue" if "left" in ttype else "red"
            label = "Left MI" if "left" in ttype else "Right MI"
            ax.bar(x + offset, powers, width, yerr=stds, capsize=3,
                   label=label, color=color, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(CH_NAMES)
        ax.set_title(bname)
        ax.set_ylabel("Power (uV^2)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "plot_08_left_vs_right_bands.png", dpi=150)
    plt.close()
    print(f"  Saved plot_08_left_vs_right_bands.png")


# ============================================================
# Cross-subject comparison plots
# ============================================================
print(f"\n{'='*60}")
print("Generating cross-subject comparison plots")
print(f"{'='*60}")

sub_ids = list(subjects.keys())
sub_labels = [subjects[s]["label"] for s in sub_ids]

# ==== Cross-subject Plot 1: Summary table ====
summary_df = pd.DataFrame(all_summary)
print("\n--- Data Summary ---")
print(summary_df.to_string(index=False))

# ==== Cross-subject Plot 2: Alpha/Beta band power comparison (key BCI bands) ====
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Cross-Subject Comparison: Alpha & Beta Band Power (Motor Imagery)",
             fontsize=14, fontweight="bold")

for row, bname in enumerate(["Alpha (8-13)", "Beta (13-30)"]):
    for col, ttype in enumerate(MI_TYPES):
        ax = axes[row][col]
        x = np.arange(len(CH_NAMES))
        width = 0.8 / len(sub_ids)

        for s_idx, sub_id in enumerate(sub_ids):
            powers = []
            stds = []
            for ch in CH_NAMES:
                vals = all_bandpowers[sub_id][ttype][bname][ch]
                powers.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            offset = (s_idx - len(sub_ids) / 2 + 0.5) * width
            ax.bar(x + offset, powers, width, yerr=stds, capsize=2,
                   label=subjects[sub_id]["label"], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(CH_NAMES)
        ax.set_ylabel("Power (uV^2)")
        ax.set_title(f"{bname} - {ttype.replace('_', ' ').title()}")
        ax.legend(fontsize=6, loc="upper right")

plt.tight_layout()
plt.savefig(DATA_DIR / "cross_subject_alpha_beta.png", dpi=150)
plt.close()
print("Saved cross_subject_alpha_beta.png")

# ==== Cross-subject Plot 3: Left-Right asymmetry (lateralization index) ====
fig, axes = plt.subplots(1, len(BANDS), figsize=(22, 5))
fig.suptitle("Cross-Subject: Left-Right Power Asymmetry per Band\n"
             "(Positive = more power for Left MI, Negative = more for Right MI)",
             fontsize=13, fontweight="bold")

for b_idx, bname in enumerate(BANDS):
    ax = axes[b_idx]
    x = np.arange(len(CH_NAMES))
    width = 0.8 / len(sub_ids)

    for s_idx, sub_id in enumerate(sub_ids):
        asymmetry = []
        for ch in CH_NAMES:
            left_vals = all_bandpowers[sub_id]["motor_imagery_left"][bname][ch]
            right_vals = all_bandpowers[sub_id]["motor_imagery_right"][bname][ch]
            left_mean = np.mean(left_vals) if left_vals else 0
            right_mean = np.mean(right_vals) if right_vals else 0
            total = left_mean + right_mean
            if total > 0:
                asymmetry.append((left_mean - right_mean) / total)
            else:
                asymmetry.append(0)

        offset = (s_idx - len(sub_ids) / 2 + 0.5) * width
        ax.bar(x + offset, asymmetry, width, label=subjects[sub_id]["label"], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(CH_NAMES)
    ax.set_title(bname)
    ax.set_ylabel("Asymmetry Index")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    if b_idx == 0:
        ax.legend(fontsize=6, loc="lower left")

plt.tight_layout()
plt.savefig(DATA_DIR / "cross_subject_asymmetry.png", dpi=150)
plt.close()
print("Saved cross_subject_asymmetry.png")

# ==== Cross-subject Plot 4: Relative band power profiles ====
fig, axes = plt.subplots(len(sub_ids), 2, figsize=(14, 4 * len(sub_ids)))
fig.suptitle("Cross-Subject: Relative Band Power Profiles", fontsize=14, fontweight="bold")

for s_idx, sub_id in enumerate(sub_ids):
    for t_idx, ttype in enumerate(MI_TYPES):
        ax = axes[s_idx][t_idx] if len(sub_ids) > 1 else axes[t_idx]
        bottoms = np.zeros(len(CH_NAMES))
        for b_idx, bname in enumerate(BANDS):
            vals = [np.mean(all_rel_bandpowers[sub_id][ttype][bname][ch])
                    if all_rel_bandpowers[sub_id][ttype][bname][ch] else 0
                    for ch in CH_NAMES]
            ax.bar(CH_NAMES, vals, bottom=bottoms, label=bname,
                   color=BAND_COLORS[b_idx], alpha=0.85)
            bottoms += np.array(vals)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Relative Power")
        title = f"{subjects[sub_id]['label']} - {ttype.replace('_', ' ').title()}"
        ax.set_title(title, fontsize=10)
        if s_idx == 0 and t_idx == 0:
            ax.legend(fontsize=6, loc="upper right")

plt.tight_layout()
plt.savefig(DATA_DIR / "cross_subject_relative_bandpower.png", dpi=150)
plt.close()
print("Saved cross_subject_relative_bandpower.png")

# ==== Cross-subject Plot 5: Grand average PSD overlay ====
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Cross-Subject: Grand Average PSD (All Motor Imagery Trials Combined)",
             fontsize=14, fontweight="bold")

sub_colors = plt.cm.tab10(np.linspace(0, 1, len(sub_ids)))

for col_idx, (col, name) in enumerate(zip(CH_COLS, CH_NAMES)):
    ax = axes[col_idx]
    for s_idx, sub_id in enumerate(sub_ids):
        eeg, trials, fs = load_subject(subjects[sub_id])
        mi_trials = trials[trials["trial_type"].isin(MI_TYPES)]
        all_psds = []
        for _, trial in mi_trials.iterrows():
            mask = (eeg["timestamp"] >= trial["start_time"]) & (
                eeg["timestamp"] <= trial["end_time"])
            segment = eeg.loc[mask, col].values
            if len(segment) > int(fs):
                nperseg = min(int(fs), len(segment))
                freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
                all_psds.append(psd)

        if all_psds:
            min_len = min(len(p) for p in all_psds)
            all_psds_trimmed = [p[:min_len] for p in all_psds]
            mean_psd = np.mean(all_psds_trimmed, axis=0)
            ax.semilogy(freqs[:min_len], mean_psd, linewidth=2,
                        color=sub_colors[s_idx], label=subjects[sub_id]["label"])

    ax.set_xlim(0, 50)
    ax.set_title(name)
    ax.set_xlabel("Frequency (Hz)")
    if col_idx == 0:
        ax.set_ylabel("PSD (uV^2/Hz)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / "cross_subject_psd.png", dpi=150)
plt.close()
print("Saved cross_subject_psd.png")

print(f"\n{'='*60}")
print("All analyses complete!")
print(f"{'='*60}")
print(f"\nPer-subject plots saved in each sub*/  directory (8 plots each)")
print(f"Cross-subject comparison plots saved in {DATA_DIR}/:")
print(f"  - cross_subject_alpha_beta.png")
print(f"  - cross_subject_asymmetry.png")
print(f"  - cross_subject_relative_bandpower.png")
print(f"  - cross_subject_psd.png")
