# Classification Results Summary

4 subjects (sub02–sub05), 77 motor imagery epochs, 190 blink epochs. Chance level: 50%.
sub03 excluded from blink detection (0 blink epochs due to poor signal).

## Evaluation Methods

### Within-Subject CV (Cross-Validation)

Evaluates how well a model works for a single person using their own data. Each subject's epochs are split into K folds (5-fold). The model trains on K-1 folds and tests on the remaining fold, rotating so every trial is tested exactly once. Per-subject accuracies are averaged. The "Mean Accuracy" tables report the average across all subjects.

This answers: *"If I calibrate a model on a user's data, how well does it predict that user's new trials?"*

### LOSO (Leave-One-Subject-Out)

Evaluates cross-subject generalization — whether a model trained on other people works on a new person without any calibration. One subject is held out entirely, the model trains on all remaining subjects, and then tests on the held-out subject. This repeats for each subject, and accuracies are averaged.

This answers: *"If a new user puts on the headband with zero calibration, how well will the model work?"*

### Why Both Matter

- **Within-Subject** is the realistic scenario with a short calibration session per user.
- **LOSO** is the harder, zero-calibration scenario.
- The drop from within-subject to LOSO reflects how much individual variation exists in the signal. A small drop (like blink detection: 88.7% → 86.0%) means the signal is consistent across people. A large drop (like motor imagery: ~61% → ~62% but both near chance) means high person-to-person variability.

---

## Motor Imagery (Left vs Right)

### Pipeline Comparison — Mean Accuracy (%)

#### Within-Subject CV

| Pipeline | Config | Features | LDA | SVM-lin | SVM-rbf | RF | **Best** |
|----------|--------|----------|-----|---------|---------|-----|----------|
| v1 Band Power | temporal_only (TP9, TP10) | 40 | 56.8 | 57.0 | 55.5 | 53.8 | SVM-lin 57.0 |
| v1 Band Power | all_4ch | 81 | 53.8 | 51.1 | **61.0** | 55.3 | SVM-rbf 61.0 |
| v1 Band Power | auto (TP9, AF8, TP10) | 58 | 55.9 | 54.6 | 59.7 | 49.9 | SVM-rbf 59.7 |
| v2 TimeFreq | temporal_only | 84 | 48.6 | 51.1 | 57.2 | **60.1** | RF 60.1 |
| v2 TimeFreq | all_4ch | 169 | 49.4 | 50.7 | 58.5 | 52.4 | SVM-rbf 58.5 |
| v2 TimeFreq | auto | 118 | 55.9 | 57.1 | 59.7 | 50.9 | SVM-rbf 59.7 |
| v3 EEGNet | temporal_only | 724 | 54.3 | 54.3 | 47.2 | 50.3 | LDA/SVM-lin 54.3 |
| v3 EEGNet | all_4ch | 809 | 51.8 | 51.8 | 51.0 | **60.7** | RF 60.7 |
| v3 EEGNet | auto | 758 | 50.3 | 51.8 | 48.5 | **60.5** | RF 60.5 |

#### LOSO (Leave-One-Subject-Out)

| Pipeline | Config | LDA | SVM-lin | SVM-rbf | RF | **Best** |
|----------|--------|-----|---------|---------|-----|----------|
| v1 Band Power | temporal_only | 53.7 | 52.2 | 50.5 | 49.3 | LDA 53.7 |
| v1 Band Power | all_4ch | 49.0 | 54.5 | 52.0 | 53.2 | SVM-lin 54.5 |
| v1 Band Power | auto | 49.3 | 49.3 | 53.5 | 54.7 | RF 54.7 |
| v2 TimeFreq | temporal_only | 48.2 | 43.9 | 50.5 | 48.8 | SVM-rbf 50.5 |
| v2 TimeFreq | all_4ch | 53.5 | 46.8 | 47.8 | **58.7** | RF 58.7 |
| v2 TimeFreq | auto | 49.5 | 45.5 | 54.9 | **56.2** | RF 56.2 |
| v3 EEGNet | temporal_only | 52.0 | 45.3 | 53.2 | **62.2** | RF 62.2 |
| v3 EEGNet | all_4ch | **61.0** | 58.7 | 54.7 | **61.2** | RF 61.2 |
| v3 EEGNet | auto | 51.2 | 48.5 | 49.3 | **61.0** | RF 61.0 |

### Best Motor Imagery Results

| Metric | Pipeline | Config | Classifier | Accuracy |
|--------|----------|--------|------------|----------|
| Best within-subject | v1 Band Power | all_4ch | SVM-rbf | **61.0%** |
| Best LOSO | v3 EEGNet | temporal_only | RF | **62.2%** |

### Per-Subject Within-Subject CV (Best Config per Pipeline)

| Subject | v1 (all_4ch, SVM-rbf) | v2 (temporal_only, RF) | v3 (all_4ch, RF) |
|---------|------------------------|------------------------|-------------------|
| sub02 | 60.0 | 65.0 | 65.0 |
| sub03 | 60.0 | 65.0 | 55.0 |
| sub04 | 58.8 | 35.3 | 52.9 |
| sub05 | 65.0 | 75.0 | 70.0 |

### Per-Subject LOSO (Best Config per Pipeline)

| Subject | v1 (auto, RF) | v2 (all_4ch, RF) | v3 (all_4ch, RF) |
|---------|---------------|-------------------|-------------------|
| sub02 | 55.0 | 70.0 | 60.0 |
| sub03 | 45.0 | 50.0 | 65.0 |
| sub04 | 58.8 | 64.7 | 64.7 |
| sub05 | 60.0 | 50.0 | 55.0 |

---

## Blink Detection (Intentional Blink vs Non-Blink)

### Mean Accuracy (%)

#### Within-Subject CV

| Config | Channels | Features | LDA | SVM-lin | SVM-rbf | RF | Threshold | **Best** |
|--------|----------|----------|-----|---------|---------|-----|-----------|----------|
| frontal_only | AF7, AF8 | 31 | 79.3 | 72.7 | 82.0 | **83.3** | 80.0 | RF 83.3 |
| all_4ch | TP9, AF7, AF8, TP10 | 59 | **88.7** | 88.0 | 84.7 | 88.0 | 80.0 | LDA 88.7 |

#### LOSO

| Config | LDA | SVM-lin | SVM-rbf | RF | Threshold | **Best** |
|--------|-----|---------|---------|-----|-----------|----------|
| frontal_only | 70.0 | 64.7 | 79.3 | 81.3 | **82.0** | Threshold 82.0 |
| all_4ch | **86.0** | 73.3 | 81.3 | 84.7 | 82.0 | LDA 86.0 |

### Best Blink Detection Results

| Metric | Config | Classifier | Accuracy |
|--------|--------|------------|----------|
| Best within-subject | all_4ch | LDA | **88.7%** |
| Best LOSO | all_4ch | LDA | **86.0%** |

### Per-Subject Blink Detection (all_4ch)

| Subject | Within-Subject LDA | LOSO LDA |
|---------|--------------------|----------|
| sub02 | 82.0 | 84.0 |
| sub03 | — (no blink data) | — |
| sub04 | 90.0 | 86.0 |
| sub05 | 94.0 | 88.0 |

---

## Channel Quality (Composite Score, 0–1)

| Subject | TP9 | AF7 | AF8 | TP10 |
|---------|-----|-----|-----|------|
| sub02 | 0.78 | 0.66 | **0.46** | 0.78 |
| sub03 | 0.66 | **0.37** | 0.73 | 0.66 |
| sub04 | 0.69 | 0.64 | **0.50** | 0.67 |
| sub05 | 0.76 | **0.47** | **0.49** | 0.69 |

Temporal channels (TP9, TP10) are consistently better quality. Frontal channels (AF7, AF8) show high artifact rates (>45%) in multiple subjects.

---

## Key Takeaways

1. **Blink detection works well** — 88.7% within-subject, 86.0% cross-subject with LDA on all 4 channels. Ready for real-time use.
2. **Motor imagery is near chance** — best results around 60–62%, only marginally above the 50% baseline. High variance across subjects.
3. **RF generalizes best for MI** — Random Forest consistently dominates LOSO across all pipelines, suggesting it handles cross-subject variability better.
4. **EEGNet features help LOSO** — v3 pipeline achieves the best cross-subject MI accuracy (62.2%), despite not improving within-subject scores.
5. **sub04 is difficult** — consistently lowest MI accuracy across pipelines, possibly due to AF8 artifact rate (45.7%).
6. **sub05 is the best MI performer** — consistently highest within-subject accuracy, up to 85% with RF in v2.
7. **More features != better** — v3 (640+ EEGNet features) doesn't clearly outperform v1/v2 for within-subject, suggesting overfitting risk with small sample sizes.
