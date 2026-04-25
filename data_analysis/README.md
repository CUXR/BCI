# Data Analysis

Three independent areas, each owns its own scripts. Run any of them
after [downloading the data into `../data/`](../data/README.md).

| Folder | What it covers |
|---|---|
| [`raw/`](raw/) | Per-subject views of the unprocessed signal: continuous EEG with trial markers, averaged epochs, blink-block zoom. |
| [`signal_level/`](signal_level/) | PSD and band-power analyses, per-subject and cross-subject. |
| [`ml/`](ml/) | Classification results — see [`ml/results.md`](ml/results.md). |

`_common.py` holds the shared subject-discovery and CSV-loading helpers
so the per-area scripts don't duplicate them. Per-area `outputs/`
directories are gitignored — figures regenerate from the scripts.

## Running everything

```bash
python data_analysis/raw/generate.py
python data_analysis/signal_level/generate.py
```

ML is trained from `../all_ml_models/`; `ml/results.md` summarizes the
last run.
