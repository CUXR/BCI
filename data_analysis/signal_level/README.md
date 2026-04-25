# Signal-Level Analysis

Spectral characterization of the EEG: time-domain windowing → Welch PSD →
band power. No classification here; that's `../ml/`. Operates on the same
CSVs as `../raw/` but answers a different question — *what frequencies are
present, and how do they shift between trial types?*

## Per-subject plots (under `outputs/<subject_key>/`)

| File | What it shows |
|---|---|
| `plot_03_psd.png` | PSD per trial type per channel (3 × 4 grid). Individual-trial PSDs underlaid with the mean. Use this to spot mu/beta peaks and powerline contamination. |
| `plot_06_bandpower_absolute.png` | Absolute band power (delta/theta/alpha/beta/gamma) per channel, per MI side. Log y-axis. |
| `plot_07_bandpower_relative.png` | Same, but each band as a fraction of total 1–50 Hz power. Stacked bars. |
| `plot_08_left_vs_right_bands.png` | Left vs Right MI band power side-by-side per band. Use this as a quick lateralization sanity check. |

## Cross-subject plots (under `outputs/cross_subject/`)

| File | What it shows |
|---|---|
| `cross_subject_psd.png` | Grand-average PSD overlay per channel — one line per subject. |
| `cross_subject_alpha_beta.png` | Alpha & beta power per channel, grouped by subject, split by L/R MI. |
| `cross_subject_asymmetry.png` | `(L − R) / (L + R)` lateralization index per band per channel. |
| `cross_subject_relative_bandpower.png` | Per-subject relative-power profiles, L vs R MI. |

## Run

```bash
python data_analysis/signal_level/generate.py
```

Subjects come from `data_analysis/_common.py::find_subject_data` —
recursive over `data/` with parent-folder disambiguation. `outputs/` is
gitignored.
