# ML

Classification results for the two BCI tasks.

- **[`results.md`](results.md)** — full breakdown: pipelines, configs, classifiers, per-subject scores, channel quality, and key takeaways.

The training scripts themselves live under
[`../../all_ml_models/`](../../all_ml_models/) (motor imagery: bandpower /
timefreq / eegnet variants; blink detection). Re-running them
regenerates `data/classification_results*.json` and
`data/blink_detection_results.json`, which `results.md` summarizes.
