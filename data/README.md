# Data

Raw EEG recordings live on Google Drive — they are deliberately **not committed to this repo** (see `.gitignore`). Download them once and drop them into this folder.

## Download

[Muse_Data — Google Drive](https://drive.google.com/drive/folders/1bHQ3Cm5NYfPvOb-EGZXeE6YcODsiLqth?usp=sharing)

Use Drive's "Download" button on the top-level `Muse_Data` folder to get
**one zip containing both phases** (e.g. `Muse_Data-YYYYMMDD....zip`). There
is no separate Phase 1 / Phase 2 download.

## Layout

After extracting **into this `data/` folder**, you should have **both
phases side-by-side**:

```
data/
  Muse_Data/
    Phase1/
      sub02/  sub03/  sub04/  sub05/  sub06/
      sub07/  sub08/  sub09/  sub10/  sub11/  sub12/
    Phase2/
      sub03/  sub05/
  README.md   ← this file
```

Each `subNN/` directory contains:

- `eeg_data_YYYYMMDD_HHMMSS.csv` — raw 4-channel EEG (TP9, AF7, AF8, TP10) at 256 Hz
- `trial_log_YYYYMMDD_HHMMSS.csv` — unified trial log
- `metadata.yaml` — participant + session info
- (Phase1 only) `plot_*.png` — per-subject diagnostic plots

## Quick extract

```bash
# from repo root — one zip, populates BOTH Phase1/ and Phase2/
unzip ~/Downloads/Muse_Data-*.zip -d data/
```

After this you will have `data/Muse_Data/Phase1/` **and**
`data/Muse_Data/Phase2/` populated from the same archive.

The analysis and training scripts walk `data/` recursively, so the nested
`Muse_Data/Phase{1,2}/sub*/` layout works out of the box — no flattening needed.
When the same subject appears in both phases (sub03, sub05), it is keyed as
`Phase1_sub03` / `Phase2_sub03` to keep results separate.

## Generated outputs

These files (when present) are pipeline outputs, not raw data, and are also
gitignored — they get regenerated when you rerun the scripts:

- `classification_results*.json` — motor imagery results
- `blink_detection_results.json` — blink detection results
