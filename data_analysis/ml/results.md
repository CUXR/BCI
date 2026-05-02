# Classification Results Summary

11 subjects (Phase 1, sub02–sub12), 171 motor imagery epochs, 519 blink epochs (incl. negatives). Chance level: 50%. Refreshed 2026-04-25.

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
- The drop from within-subject to LOSO reflects how much individual variation exists in the signal.

---

## Motor Imagery (Left vs Right)

### Pipeline Comparison — Mean Accuracy (%)

#### Within-Subject CV

| Pipeline | Config | Channels | Features | LDA | SVM-lin | SVM-rbf | RF | **Best** |
|----------|--------|----------|----------|-----|---------|---------|-----|----------|
| v1 Band Power | temporal_only | TP9, TP10 | 40 | 56.7 | 55.6 | 51.6 | 49.9 | LDA 56.7 |
| v1 Band Power | all_4ch | TP9, AF7, AF8, TP10 | 81 | 46.8 | 50.8 | 48.6 | 47.4 | SVM_linear 50.8 |
| v1 Band Power | auto | TP9, TP10 | 40 | 56.7 | 55.6 | 51.6 | 49.9 | LDA 56.7 |
| v2 TimeFreq | temporal_only | TP9, TP10 | 84 | 55.7 | 55.1 | 54.1 | 54.8 | LDA 55.7 |
| v2 TimeFreq | all_4ch | TP9, AF7, AF8, TP10 | 169 | 48.5 | 52.8 | 48.6 | 52.1 | SVM_linear 52.8 |
| v2 TimeFreq | auto | TP9, TP10 | 84 | 55.7 | 55.1 | 54.1 | 54.8 | LDA 55.7 |
| v3 EEGNet | temporal_only | TP9, TP10 | 724 | 54.0 | 52.1 | 49.7 | 46.9 | LDA 54.0 |
| v3 EEGNet | all_4ch | TP9, AF7, AF8, TP10 | 809 | 50.4 | 50.5 | 50.3 | 47.2 | SVM_linear 50.5 |
| v3 EEGNet | auto | TP9, TP10 | 724 | 54.0 | 52.1 | 49.7 | 46.9 | LDA 54.0 |

#### LOSO (Leave-One-Subject-Out)

| Pipeline | Config | LDA | SVM-lin | SVM-rbf | RF | **Best** |
|----------|--------|-----|---------|---------|-----|----------|
| v1 Band Power | temporal_only | 55.5 | 53.7 | 57.3 | 56.2 | SVM_rbf 57.3 |
| v1 Band Power | all_4ch | 53.4 | 48.2 | 51.6 | 52.4 | LDA 53.4 |
| v1 Band Power | auto | 55.5 | 53.7 | 57.3 | 56.2 | SVM_rbf 57.3 |
| v2 TimeFreq | temporal_only | 46.9 | 52.0 | 54.5 | 55.3 | RF 55.3 |
| v2 TimeFreq | all_4ch | 52.5 | 51.4 | 54.9 | 56.3 | RF 56.3 |
| v2 TimeFreq | auto | 46.9 | 52.0 | 54.5 | 55.3 | RF 55.3 |
| v3 EEGNet | temporal_only | 46.9 | 45.9 | 57.6 | 45.6 | SVM_rbf 57.6 |
| v3 EEGNet | all_4ch | 46.9 | 46.5 | 52.4 | 59.4 | RF 59.4 |
| v3 EEGNet | auto | 46.9 | 45.9 | 57.6 | 45.6 | SVM_rbf 57.6 |

### Best Motor Imagery Results

| Metric | Pipeline | Config | Classifier | Accuracy |
|--------|----------|--------|------------|----------|
| Best within-subject | v1 | temporal_only | LDA | **56.7%** |
| Best LOSO | v3 | all_4ch | RF | **59.4%** |

### Per-Subject Within-Subject CV (Best Config per Pipeline)

| Subject | v1 (temporal_only, LDA) | v2 (temporal_only, LDA) | v3 (temporal_only, LDA) |
|---|---|---|---|
| sub02 | 45.0 | 55.0 | 50.0 |
| sub03 | 33.3 | 44.4 | 66.7 |
| sub04 | 47.1 | 35.3 | 41.2 |
| sub05 | 80.0 | 80.0 | 65.0 |
| sub06 | 57.9 | 63.2 | 52.6 |
| sub07 | 63.6 | 54.5 | 54.5 |
| sub08 | 72.2 | 66.7 | 44.4 |
| sub09 | 42.9 | 42.9 | 57.1 |
| sub10 | 50.0 | 25.0 | 43.8 |
| sub11 | — | — | — |
| sub12 | 75.0 | 90.0 | 65.0 |

### Per-Subject LOSO (Best Config per Pipeline)

| Subject | v1 (temporal_only, SVM_rbf) | v2 (all_4ch, RF) | v3 (all_4ch, RF) |
|---|---|---|---|
| sub02 | 60.0 | 50.0 | 70.0 |
| sub03 | 55.6 | 55.6 | 50.0 |
| sub04 | 58.8 | 64.7 | 58.8 |
| sub05 | 75.0 | 45.0 | 45.0 |
| sub06 | 57.9 | 52.6 | 47.4 |
| sub07 | 36.4 | 36.4 | 72.7 |
| sub08 | 77.8 | 61.1 | 55.6 |
| sub09 | 42.9 | 57.1 | 42.9 |
| sub10 | 56.2 | 56.2 | 56.2 |
| sub11 | 60.0 | 80.0 | 100.0 |
| sub12 | 50.0 | 60.0 | 55.0 |

---

## Blink Detection (Intentional Blink vs Non-Blink)

### Mean Accuracy (%)

#### Within-Subject CV

| Config | Channels | Features | LDA | SVM-lin | SVM-rbf | RF | Threshold | **Best** |
|--------|----------|----------|-----|---------|---------|-----|-----------|----------|
| frontal_only | AF7, AF8 | 31 | 86.1 | 84.3 | 87.4 | 86.5 | 81.2 | SVM_rbf 87.4 |
| all_4ch | TP9, AF7, AF8, TP10 | 59 | 91.3 | 89.6 | 87.6 | 90.2 | 81.8 | LDA 91.3 |

#### LOSO

| Config | LDA | SVM-lin | SVM-rbf | RF | Threshold | **Best** |
|--------|-----|---------|---------|-----|-----------|----------|
| frontal_only | 77.9 | 78.9 | 77.3 | 78.1 | 76.0 | SVM_linear 78.9 |
| all_4ch | 79.8 | 82.2 | 75.4 | 82.2 | 75.8 | SVM_linear 82.2 |

### Best Blink Detection Results

| Metric | Config | Classifier | Accuracy |
|--------|--------|------------|----------|
| Best within-subject | all_4ch | LDA | **91.3%** |
| Best LOSO | all_4ch | SVM_linear | **82.2%** |

### Per-Subject Blink Detection (all_4ch, LDA)

| Subject | Within-Subject LDA | LOSO LDA |
|---------|--------------------|----------|
| sub02 | 92.0 | 62.0 |
| sub03 | 92.0 | 90.0 |
| sub04 | 88.0 | 92.0 |
| sub05 | 92.0 | 80.0 |
| sub06 | 93.2 | 86.4 |
| sub07 | 83.7 | 75.5 |
| sub08 | 92.0 | 66.0 |
| sub09 | 100.0 | 96.0 |
| sub10 | — | — |
| sub11 | 97.7 | 100.0 |
| sub12 | 82.0 | 50.0 |

---

## Channel Quality (Composite Score, 0–1)

| Subject | TP9 | AF7 | AF8 | TP10 |
|---------|-----|-----|-----|------|
| sub02 | 0.81 | 0.45 | 0.44 | 0.74 |
| sub03 | 0.73 | 0.44 | 0.44 | 0.65 |
| sub04 | 0.69 | 0.64 | 0.50 | 0.67 |
| sub05 | 0.76 | 0.47 | 0.49 | 0.69 |
| sub06 | 0.62 | 0.43 | 0.41 | 0.60 |
| sub07 | 0.72 | 0.33 | 0.35 | 0.72 |
| sub08 | 0.78 | 0.34 | 0.35 | 0.79 |
| sub09 | 0.83 | 0.70 | 0.59 | 0.82 |
| sub10 | 0.79 | 0.66 | 0.43 | 0.68 |
| sub11 | 0.73 | 0.37 | 0.35 | 0.76 |
| sub12 | 0.38 | 0.34 | 0.35 | 0.69 |

