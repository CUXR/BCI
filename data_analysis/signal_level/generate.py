"""Generate signal-level analyses (PSD, band power) — per subject + cross-subject.

Per subject (under ``outputs/<subject_key>/``):
  plot_03_psd.png                  — PSD per trial type per channel
  plot_06_bandpower_absolute.png   — abs band power per channel, per MI side
  plot_07_bandpower_relative.png   — relative band power per channel, per MI side
  plot_08_left_vs_right_bands.png  — L vs R MI band power side-by-side

Cross-subject (under ``outputs/cross_subject/``):
  cross_subject_psd.png                  — grand-average PSD overlay per channel
  cross_subject_alpha_beta.png           — alpha & beta band power, per subject
  cross_subject_asymmetry.png            — L–R asymmetry index per band per channel
  cross_subject_relative_bandpower.png   — relative-power profiles per subject
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import (
    CH_COLS,
    CH_NAMES,
    MI_TYPES,
    TRIAL_COLORS,
    find_subject_data,
    load_subject,
    output_dir,
)

BANDS = {
    "Delta (1-4)": (1, 4),
    "Theta (4-8)": (4, 8),
    "Alpha (8-13)": (8, 13),
    "Beta (13-30)": (13, 30),
    "Gamma (30-50)": (30, 50),
}
BAND_COLORS = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#b07aa1"]
TRIAL_TYPES = ["motor_imagery_left", "motor_imagery_right", "blink_intentional"]


def _trapz(y, x):
    # numpy 2.0 removed np.trapz; np.trapezoid replaces it.
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x)


def compute_bandpower(signal, fs, fmin, fmax):
    nperseg = min(int(fs * 2), len(signal))
    if nperseg < int(fs * 0.5):
        return np.nan
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = (freqs >= fmin) & (freqs <= fmax)
    return _trapz(psd[idx], freqs[idx])


def compute_relative_bandpower(signal, fs, fmin, fmax):
    nperseg = min(int(fs * 2), len(signal))
    if nperseg < int(fs * 0.5):
        return np.nan
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    band_idx = (freqs >= fmin) & (freqs <= fmax)
    total_idx = (freqs >= 1) & (freqs <= 50)
    total = _trapz(psd[total_idx], freqs[total_idx])
    if total == 0:
        return np.nan
    return _trapz(psd[band_idx], freqs[band_idx]) / total


# ---------------------------------------------------------------- per subject

def plot_psd(eeg, trials, fs, label, out_path):
    fig, axes = plt.subplots(len(TRIAL_TYPES), 4, figsize=(18, 10))
    fig.suptitle(f"{label} - Power Spectral Density by Trial Type",
                 fontsize=14, fontweight="bold")

    for row, ttype in enumerate(TRIAL_TYPES):
        subset = trials[trials["trial_type"] == ttype]
        for col_idx, (col, name) in enumerate(zip(CH_COLS, CH_NAMES)):
            ax = axes[row][col_idx]
            all_psds = []
            for _, trial in subset.iterrows():
                mask = (eeg["timestamp"] >= trial["start_time"]) & (
                    eeg["timestamp"] <= trial["end_time"])
                seg = eeg.loc[mask, col].values
                if len(seg) > int(fs):
                    nperseg = min(int(fs), len(seg))
                    freqs, psd = welch(seg, fs=fs, nperseg=nperseg)
                    all_psds.append(psd)
                    ax.semilogy(freqs, psd, alpha=0.3, linewidth=0.5,
                                color=TRIAL_COLORS[ttype])
            if all_psds:
                m = min(len(p) for p in all_psds)
                mean_psd = np.mean([p[:m] for p in all_psds], axis=0)
                ax.semilogy(freqs[:m], mean_psd, linewidth=2,
                            color=TRIAL_COLORS[ttype])
            ax.set_xlim(0, 50)
            if row == 0:
                ax.set_title(name)
            if col_idx == 0:
                ax.set_ylabel(f"{ttype.replace('_', ' ')}\nPSD (uV^2/Hz)")
            if row == len(TRIAL_TYPES) - 1:
                ax.set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_bandpower_absolute(bp, label, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{label} - Absolute Band Power by Trial Type",
                 fontsize=14, fontweight="bold")
    for ax_idx, ttype in enumerate(MI_TYPES):
        ax = axes[ax_idx]
        x = np.arange(len(CH_NAMES))
        width = 0.15
        for b_idx, bname in enumerate(BANDS):
            powers = [np.mean(bp[ttype][bname][ch]) if bp[ttype][bname][ch] else 0
                      for ch in CH_NAMES]
            stds = [np.std(bp[ttype][bname][ch]) if bp[ttype][bname][ch] else 0
                    for ch in CH_NAMES]
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
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_bandpower_relative(rbp, label, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{label} - Relative Band Power",
                 fontsize=14, fontweight="bold")
    for ax_idx, ttype in enumerate(MI_TYPES):
        ax = axes[ax_idx]
        bottoms = np.zeros(len(CH_NAMES))
        for b_idx, bname in enumerate(BANDS):
            vals = [np.mean(rbp[ttype][bname][ch]) if rbp[ttype][bname][ch] else 0
                    for ch in CH_NAMES]
            ax.bar(CH_NAMES, vals, bottom=bottoms, label=bname,
                   color=BAND_COLORS[b_idx], alpha=0.85)
            bottoms += np.array(vals)
        ax.set_ylabel("Relative Power")
        ax.set_title(ttype.replace("_", " ").title())
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_left_vs_right_bands(bp, label, out_path):
    fig, axes = plt.subplots(1, len(BANDS), figsize=(20, 5))
    fig.suptitle(f"{label} - Left vs Right MI Band Power",
                 fontsize=14, fontweight="bold")
    for b_idx, bname in enumerate(BANDS):
        ax = axes[b_idx]
        x = np.arange(len(CH_NAMES))
        width = 0.35
        for t_idx, ttype in enumerate(MI_TYPES):
            powers = [np.mean(bp[ttype][bname][ch]) if bp[ttype][bname][ch] else 0
                      for ch in CH_NAMES]
            stds = [np.std(bp[ttype][bname][ch]) if bp[ttype][bname][ch] else 0
                    for ch in CH_NAMES]
            offset = (t_idx - 0.5) * width
            color = "blue" if "left" in ttype else "red"
            label_t = "Left MI" if "left" in ttype else "Right MI"
            ax.bar(x + offset, powers, width, yerr=stds, capsize=3,
                   label=label_t, color=color, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(CH_NAMES)
        ax.set_title(bname)
        ax.set_ylabel("Power (uV^2)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def aggregate_subject_bandpower(eeg, trials, fs):
    """Return (abs, rel) nested dicts: {ttype: {band: {ch: [trial_values]}}}."""
    abs_bp = {t: {b: {c: [] for c in CH_NAMES} for b in BANDS} for t in MI_TYPES}
    rel_bp = {t: {b: {c: [] for c in CH_NAMES} for b in BANDS} for t in MI_TYPES}
    for ttype in MI_TYPES:
        subset = trials[trials["trial_type"] == ttype]
        for bname, (fmin, fmax) in BANDS.items():
            for col_idx, col in enumerate(CH_COLS):
                ch = CH_NAMES[col_idx]
                for _, trial in subset.iterrows():
                    mask = (eeg["timestamp"] >= trial["start_time"]) & (
                        eeg["timestamp"] <= trial["end_time"])
                    sig = eeg.loc[mask, col].values
                    a = compute_bandpower(sig, fs, fmin, fmax)
                    r = compute_relative_bandpower(sig, fs, fmin, fmax)
                    if not np.isnan(a):
                        abs_bp[ttype][bname][ch].append(a)
                    if not np.isnan(r):
                        rel_bp[ttype][bname][ch].append(r)
    return abs_bp, rel_bp


# ---------------------------------------------------------------- cross subject

def plot_cross_alpha_beta(all_abs, subjects, out_path):
    sub_ids = list(subjects.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Cross-Subject: Alpha & Beta Band Power (Motor Imagery)",
                 fontsize=14, fontweight="bold")
    for row, bname in enumerate(["Alpha (8-13)", "Beta (13-30)"]):
        for col, ttype in enumerate(MI_TYPES):
            ax = axes[row][col]
            x = np.arange(len(CH_NAMES))
            width = 0.8 / max(len(sub_ids), 1)
            for s_idx, sid in enumerate(sub_ids):
                powers, stds = [], []
                for ch in CH_NAMES:
                    vals = all_abs[sid][ttype][bname][ch]
                    powers.append(np.mean(vals) if vals else 0)
                    stds.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                offset = (s_idx - len(sub_ids) / 2 + 0.5) * width
                ax.bar(x + offset, powers, width, yerr=stds, capsize=2,
                       label=subjects[sid]["label"], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(CH_NAMES)
            ax.set_ylabel("Power (uV^2)")
            ax.set_title(f"{bname} - {ttype.replace('_', ' ').title()}")
            ax.legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cross_asymmetry(all_abs, subjects, out_path):
    sub_ids = list(subjects.keys())
    fig, axes = plt.subplots(1, len(BANDS), figsize=(22, 5))
    fig.suptitle("Cross-Subject: Left-Right Power Asymmetry per Band\n"
                 "(Positive = more power for Left MI, Negative = more for Right MI)",
                 fontsize=13, fontweight="bold")
    for b_idx, bname in enumerate(BANDS):
        ax = axes[b_idx]
        x = np.arange(len(CH_NAMES))
        width = 0.8 / max(len(sub_ids), 1)
        for s_idx, sid in enumerate(sub_ids):
            asym = []
            for ch in CH_NAMES:
                lv = all_abs[sid]["motor_imagery_left"][bname][ch]
                rv = all_abs[sid]["motor_imagery_right"][bname][ch]
                lm = np.mean(lv) if lv else 0
                rm = np.mean(rv) if rv else 0
                total = lm + rm
                asym.append((lm - rm) / total if total > 0 else 0)
            offset = (s_idx - len(sub_ids) / 2 + 0.5) * width
            ax.bar(x + offset, asym, width, label=subjects[sid]["label"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(CH_NAMES)
        ax.set_title(bname)
        ax.set_ylabel("Asymmetry Index")
        ax.axhline(0, color="black", linewidth=0.5)
        if b_idx == 0:
            ax.legend(fontsize=6, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cross_relative(all_rel, subjects, out_path):
    sub_ids = list(subjects.keys())
    fig, axes = plt.subplots(len(sub_ids), 2,
                             figsize=(14, 4 * max(len(sub_ids), 1)))
    fig.suptitle("Cross-Subject: Relative Band Power Profiles",
                 fontsize=14, fontweight="bold")
    if len(sub_ids) == 1:
        axes = np.array([axes])
    for s_idx, sid in enumerate(sub_ids):
        for t_idx, ttype in enumerate(MI_TYPES):
            ax = axes[s_idx][t_idx]
            bottoms = np.zeros(len(CH_NAMES))
            for b_idx, bname in enumerate(BANDS):
                vals = [np.mean(all_rel[sid][ttype][bname][ch])
                        if all_rel[sid][ttype][bname][ch] else 0
                        for ch in CH_NAMES]
                ax.bar(CH_NAMES, vals, bottom=bottoms, label=bname,
                       color=BAND_COLORS[b_idx], alpha=0.85)
                bottoms += np.array(vals)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Relative Power")
            ax.set_title(f"{subjects[sid]['label']} - "
                         f"{ttype.replace('_', ' ').title()}", fontsize=10)
            if s_idx == 0 and t_idx == 0:
                ax.legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cross_psd(subjects, out_path):
    sub_ids = list(subjects.keys())
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Cross-Subject: Grand Average PSD (All MI Trials Combined)",
                 fontsize=14, fontweight="bold")
    sub_colors = plt.cm.tab10(np.linspace(0, 1, max(len(sub_ids), 1)))
    for col_idx, (col, name) in enumerate(zip(CH_COLS, CH_NAMES)):
        ax = axes[col_idx]
        for s_idx, sid in enumerate(sub_ids):
            eeg, trials, fs = load_subject(subjects[sid])
            mi = trials[trials["trial_type"].isin(MI_TYPES)]
            psds = []
            for _, trial in mi.iterrows():
                mask = (eeg["timestamp"] >= trial["start_time"]) & (
                    eeg["timestamp"] <= trial["end_time"])
                seg = eeg.loc[mask, col].values
                if len(seg) > int(fs):
                    nperseg = min(int(fs), len(seg))
                    freqs, psd = welch(seg, fs=fs, nperseg=nperseg)
                    psds.append(psd)
            if psds:
                m = min(len(p) for p in psds)
                mean_psd = np.mean([p[:m] for p in psds], axis=0)
                ax.semilogy(freqs[:m], mean_psd, linewidth=2,
                            color=sub_colors[s_idx], label=subjects[sid]["label"])
        ax.set_xlim(0, 50)
        ax.set_title(name)
        ax.set_xlabel("Frequency (Hz)")
        if col_idx == 0:
            ax.set_ylabel("PSD (uV^2/Hz)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    subjects = find_subject_data()
    print(f"Found {len(subjects)} subjects: {list(subjects.keys())}")

    all_abs, all_rel = {}, {}
    for sub_id, info in subjects.items():
        print(f"\n[{sub_id}] {info['label']}")
        eeg, trials, fs = load_subject(info)
        out = output_dir(__file__, sub_id)

        plot_psd(eeg, trials, fs, info["label"], out / "plot_03_psd.png")
        print("  plot_03_psd.png")

        abs_bp, rel_bp = aggregate_subject_bandpower(eeg, trials, fs)
        all_abs[sub_id] = abs_bp
        all_rel[sub_id] = rel_bp

        plot_bandpower_absolute(abs_bp, info["label"],
                                out / "plot_06_bandpower_absolute.png")
        print("  plot_06_bandpower_absolute.png")

        plot_bandpower_relative(rel_bp, info["label"],
                                out / "plot_07_bandpower_relative.png")
        print("  plot_07_bandpower_relative.png")

        plot_left_vs_right_bands(abs_bp, info["label"],
                                 out / "plot_08_left_vs_right_bands.png")
        print("  plot_08_left_vs_right_bands.png")

    cross = output_dir(__file__, "cross_subject")
    print("\n[cross-subject]")
    plot_cross_alpha_beta(all_abs, subjects, cross / "cross_subject_alpha_beta.png")
    print("  cross_subject_alpha_beta.png")
    plot_cross_asymmetry(all_abs, subjects, cross / "cross_subject_asymmetry.png")
    print("  cross_subject_asymmetry.png")
    plot_cross_relative(all_rel, subjects, cross / "cross_subject_relative_bandpower.png")
    print("  cross_subject_relative_bandpower.png")
    plot_cross_psd(subjects, cross / "cross_subject_psd.png")
    print("  cross_subject_psd.png")


if __name__ == "__main__":
    main()
