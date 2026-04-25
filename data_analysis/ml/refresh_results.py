"""Re-train v1/v2/v3 MI + blink classifiers on the current Phase 1 subject
pool and regenerate data_analysis/ml/results.md.

Pipelines & feature sets are unchanged from all_ml_models/. Only the data
sample changes (4 subjects → 11 Phase 1 subjects).

Outputs:
  data_analysis/ml/results.md
  data_analysis/ml/results_data.json
"""

import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "all_ml_models"))
sys.path.insert(0, str(REPO / "all_ml_models" / "training" / "bandpower"))
sys.path.insert(0, str(REPO / "all_ml_models" / "training" / "timefreq"))
sys.path.insert(0, str(REPO / "all_ml_models" / "training" / "eegnet"))

import train_motor_imagery as v1
import train_motor_imagery_v2_timefreq as v2
import train_motor_imagery_v3_eegnet as v3
import train_blink_detector as blink_mod
from eeg_filters import EEGFilter

PHASE1_DIR = REPO / "data" / "Muse_Data" / "Phase1"
OUT_DIR = REPO / "data_analysis" / "ml"

CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
TEMPORAL_CH = [0, 3]
FRONTAL_CH = [1, 2]
ALL_CH = [0, 1, 2, 3]
MI_CONFIGS = {"temporal_only": TEMPORAL_CH, "all_4ch": ALL_CH, "auto": None}
BLINK_CONFIGS = {"frontal_only": FRONTAL_CH, "all_4ch": ALL_CH}


def load_phase1():
    subjects_info = v1.find_subject_data(PHASE1_DIR)
    subjects = {}
    for sub_id, info in subjects_info.items():
        eeg, trials, fs = v1.load_subject(info)
        subjects[sub_id] = (eeg, trials, fs, info)
    return subjects


def channel_quality_table(subjects):
    """Per-subject composite channel quality (keyed by channel name)."""
    fs = list(subjects.values())[0][2]
    rows = {}
    for sub_id, (eeg, trials, _fs, _info) in subjects.items():
        q = v1.compute_channel_quality(eeg, trials, fs, ch_indices=ALL_CH)
        rows[sub_id] = {name: float(q[name]["composite"]) for name in CH_NAMES}
    return rows


def auto_channels(quality_dict):
    """Average per-subject composite scores then run v1's selector."""
    mean_q = {}
    for name in CH_NAMES:
        scores = [s[name] for s in quality_dict.values()]
        mean_q[name] = {"composite": float(np.mean(scores))}
    return v1.select_channels_by_quality(mean_q)


def _pad(epochs):
    if not epochs:
        return epochs
    n = max(e.shape[0] for e in epochs)
    return [np.pad(e, ((0, n - e.shape[0]), (0, 0)), mode="edge") if e.shape[0] < n else e
            for e in epochs]


def build_mi(subjects, ch_indices, version):
    fs = list(subjects.values())[0][2]
    flt = EEGFilter(sampling_rate=int(fs), notch_freq=60.0)
    eps, ys, subs = [], [], []
    for sid, (eeg, trials, fs_s, _i) in subjects.items():
        e, y, _ = v1.extract_mi_epochs(eeg, trials, fs_s, ch_indices, flt)
        eps.extend(e); ys.extend(y); subs.extend([sid] * len(e))
    if not eps:
        return None, None, None
    padded = _pad(eps)
    y = np.array(ys)
    if version == "v1":
        Xf = v1.extract_features_all(padded, fs, ch_indices)
        Xc, _ = v1.compute_csp(padded, ys)
        X = np.hstack([Xf, Xc])
    elif version == "v2":
        Xf = v2.extract_features_all(padded, fs, ch_indices)
        Xt = v2.extract_timefreq_all(padded, fs, ch_indices)
        Xc, _ = v2.compute_csp(padded, ys)
        X = np.hstack([Xf, Xt, Xc])
    elif version == "v3":
        Xf = v3.extract_features_all(padded, fs, ch_indices)
        Xt = v3.extract_timefreq_all(padded, fs, ch_indices)
        Xc, _ = v3.compute_csp(padded, ys)
        Xn = v3.extract_eegnet_features_all(padded, fs, ch_indices)
        X = np.hstack([Xf, Xt, Xc, Xn])
    return X, y, subs


def eval_mi(subjects, ch_lookup):
    classifiers = v3.get_classifiers()
    out = {}
    for version in ("v1", "v2", "v3"):
        out[version] = {}
        for cfg_name, ch_indices in ch_lookup.items():
            X, y, subs = build_mi(subjects, ch_indices, version)
            if X is None:
                continue
            ws = v3.evaluate_within_subject(X, y, subs, classifiers)
            loso = v3.evaluate_loso(X, y, subs, classifiers)
            out[version][cfg_name] = {
                "channels": [CH_NAMES[i] for i in ch_indices],
                "n_features": int(X.shape[1]),
                "n_epochs": int(X.shape[0]),
                "ws_means": {clf: float(np.nanmean([ws[s][clf] for s in ws])) for clf in classifiers},
                "loso_means": {clf: float(np.nanmean([loso[s][clf] for s in loso])) for clf in classifiers},
                "ws_per_subject": {s: {clf: (None if np.isnan(ws[s][clf]) else float(ws[s][clf]))
                                       for clf in classifiers} for s in ws},
                "loso_per_subject": {s: {clf: (None if np.isnan(loso[s][clf]) else float(loso[s][clf]))
                                         for clf in classifiers} for s in loso},
            }
            print(f"  MI {version}/{cfg_name}: WS LDA={out[version][cfg_name]['ws_means']['LDA']*100:.1f}%, "
                  f"LOSO best={max(out[version][cfg_name]['loso_means'].values())*100:.1f}%")
    return out


def eval_blink(subjects):
    fs = list(subjects.values())[0][2]
    flt = EEGFilter(sampling_rate=int(fs), notch_freq=60.0)
    target_n = int(blink_mod.BLINK_EPOCH_SECONDS * fs)
    out = {}
    classifiers = blink_mod.get_classifiers()
    for cfg_name, ch_indices in BLINK_CONFIGS.items():
        eps, ys, subs = [], [], []
        for sid, (eeg, trials, fs_s, _i) in subjects.items():
            pos = blink_mod.extract_blink_epochs(eeg, trials, fs_s, ch_indices, flt)
            neg = blink_mod.extract_nonblink_epochs(eeg, trials, fs_s, ch_indices, flt, target_samples=target_n)
            eps.extend(pos); ys.extend([1] * len(pos)); subs.extend([sid] * len(pos))
            eps.extend(neg); ys.extend([0] * len(neg)); subs.extend([sid] * len(neg))
        padded = _pad(eps)
        X = blink_mod.extract_features_all(padded, fs, ch_indices)
        y = np.array(ys)
        n_per_ch = 14
        ptp_idx = [i * n_per_ch for i in range(len(ch_indices))]
        ws = blink_mod.evaluate_within_subject(X, y, subs, classifiers, ptp_col_indices=ptp_idx)
        loso = blink_mod.evaluate_loso(X, y, subs, classifiers, ptp_col_indices=ptp_idx)
        all_clfs = list(classifiers.keys()) + ["Threshold"]
        out[cfg_name] = {
            "channels": [CH_NAMES[i] for i in ch_indices],
            "n_features": int(X.shape[1]),
            "n_epochs": int(X.shape[0]),
            "ws_means": {c: float(np.nanmean([ws[s].get(c, np.nan) for s in ws])) for c in all_clfs},
            "loso_means": {c: float(np.nanmean([loso[s].get(c, np.nan) for s in loso])) for c in all_clfs},
            "ws_per_subject": {s: {c: (None if np.isnan(ws[s].get(c, np.nan)) else float(ws[s][c]))
                                   for c in all_clfs} for s in ws},
            "loso_per_subject": {s: {c: (None if np.isnan(loso[s].get(c, np.nan)) else float(loso[s][c]))
                                     for c in all_clfs} for s in loso},
        }
        print(f"  Blink {cfg_name}: WS LDA={out[cfg_name]['ws_means']['LDA']*100:.1f}%, "
              f"LOSO LDA={out[cfg_name]['loso_means']['LDA']*100:.1f}%")
    return out


def fmt(v):
    return "—" if v is None else f"{v*100:.1f}"


def best_pair(d):
    """Return (name, value) for max accuracy in a dict (skipping NaN-likes)."""
    items = [(k, v) for k, v in d.items() if v is not None and not (isinstance(v, float) and np.isnan(v))]
    name, val = max(items, key=lambda kv: kv[1])
    return name, val


def render_md(mi_results, blink_results, channel_q, n_subjects, mi_n_epochs, blink_n_epochs):
    L = []
    L.append("# Classification Results Summary")
    L.append("")
    L.append(f"{n_subjects} subjects (Phase 1, sub02–sub12), "
             f"{mi_n_epochs} motor imagery epochs, {blink_n_epochs} blink epochs (incl. negatives). "
             f"Chance level: 50%. Refreshed {datetime.now():%Y-%m-%d}.")
    L.append("")
    L.append("## Evaluation Methods")
    L.append("")
    L.append("### Within-Subject CV (Cross-Validation)")
    L.append("")
    L.append("Evaluates how well a model works for a single person using their own data. "
             "Each subject's epochs are split into K folds (5-fold). The model trains on K-1 folds "
             "and tests on the remaining fold, rotating so every trial is tested exactly once. "
             "Per-subject accuracies are averaged. The \"Mean Accuracy\" tables report the average "
             "across all subjects.")
    L.append("")
    L.append("This answers: *\"If I calibrate a model on a user's data, how well does it predict that user's new trials?\"*")
    L.append("")
    L.append("### LOSO (Leave-One-Subject-Out)")
    L.append("")
    L.append("Evaluates cross-subject generalization — whether a model trained on other people works "
             "on a new person without any calibration. One subject is held out entirely, the model "
             "trains on all remaining subjects, and then tests on the held-out subject. This repeats "
             "for each subject, and accuracies are averaged.")
    L.append("")
    L.append("This answers: *\"If a new user puts on the headband with zero calibration, how well will the model work?\"*")
    L.append("")
    L.append("### Why Both Matter")
    L.append("")
    L.append("- **Within-Subject** is the realistic scenario with a short calibration session per user.")
    L.append("- **LOSO** is the harder, zero-calibration scenario.")
    L.append("- The drop from within-subject to LOSO reflects how much individual variation exists in the signal.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Motor Imagery (Left vs Right)")
    L.append("")
    L.append("### Pipeline Comparison — Mean Accuracy (%)")
    L.append("")
    L.append("#### Within-Subject CV")
    L.append("")
    L.append("| Pipeline | Config | Channels | Features | LDA | SVM-lin | SVM-rbf | RF | **Best** |")
    L.append("|----------|--------|----------|----------|-----|---------|---------|-----|----------|")
    for vlabel, vkey in (("v1 Band Power", "v1"), ("v2 TimeFreq", "v2"), ("v3 EEGNet", "v3")):
        for cfg_name, cfg in mi_results[vkey].items():
            best_name, best_val = best_pair(cfg["ws_means"])
            ch = ", ".join(cfg["channels"])
            L.append(f"| {vlabel} | {cfg_name} | {ch} | {cfg['n_features']} | "
                     f"{fmt(cfg['ws_means']['LDA'])} | {fmt(cfg['ws_means']['SVM_linear'])} | "
                     f"{fmt(cfg['ws_means']['SVM_rbf'])} | {fmt(cfg['ws_means']['RF'])} | "
                     f"{best_name} {best_val*100:.1f} |")
    L.append("")
    L.append("#### LOSO (Leave-One-Subject-Out)")
    L.append("")
    L.append("| Pipeline | Config | LDA | SVM-lin | SVM-rbf | RF | **Best** |")
    L.append("|----------|--------|-----|---------|---------|-----|----------|")
    for vlabel, vkey in (("v1 Band Power", "v1"), ("v2 TimeFreq", "v2"), ("v3 EEGNet", "v3")):
        for cfg_name, cfg in mi_results[vkey].items():
            best_name, best_val = best_pair(cfg["loso_means"])
            L.append(f"| {vlabel} | {cfg_name} | {fmt(cfg['loso_means']['LDA'])} | "
                     f"{fmt(cfg['loso_means']['SVM_linear'])} | {fmt(cfg['loso_means']['SVM_rbf'])} | "
                     f"{fmt(cfg['loso_means']['RF'])} | {best_name} {best_val*100:.1f} |")
    L.append("")
    # Best MI overall
    best_ws = max(
        ((v, c, *best_pair(mi_results[v][c]["ws_means"])) for v in mi_results for c in mi_results[v]),
        key=lambda x: x[3])
    best_lo = max(
        ((v, c, *best_pair(mi_results[v][c]["loso_means"])) for v in mi_results for c in mi_results[v]),
        key=lambda x: x[3])
    L.append("### Best Motor Imagery Results")
    L.append("")
    L.append("| Metric | Pipeline | Config | Classifier | Accuracy |")
    L.append("|--------|----------|--------|------------|----------|")
    L.append(f"| Best within-subject | {best_ws[0]} | {best_ws[1]} | {best_ws[2]} | **{best_ws[3]*100:.1f}%** |")
    L.append(f"| Best LOSO | {best_lo[0]} | {best_lo[1]} | {best_lo[2]} | **{best_lo[3]*100:.1f}%** |")
    L.append("")
    # Per-subject WS — use each pipeline's best WS config + classifier
    L.append("### Per-Subject Within-Subject CV (Best Config per Pipeline)")
    L.append("")
    pipeline_best_ws = {}
    for vkey in ("v1", "v2", "v3"):
        cfg_name, _ = max(
            ((c, max(mi_results[vkey][c]["ws_means"].values())) for c in mi_results[vkey]),
            key=lambda x: x[1])
        clf_name, _ = best_pair(mi_results[vkey][cfg_name]["ws_means"])
        pipeline_best_ws[vkey] = (cfg_name, clf_name)
    header = "| Subject "
    for vkey in ("v1", "v2", "v3"):
        c, k = pipeline_best_ws[vkey]
        header += f"| {vkey} ({c}, {k}) "
    header += "|"
    L.append(header)
    L.append("|" + "|".join(["---"] * 4) + "|")
    all_subs = sorted({s for vkey in mi_results for c in mi_results[vkey] for s in mi_results[vkey][c]["ws_per_subject"]})
    for s in all_subs:
        row = f"| {s} "
        for vkey in ("v1", "v2", "v3"):
            cfg_name, clf_name = pipeline_best_ws[vkey]
            v = mi_results[vkey][cfg_name]["ws_per_subject"].get(s, {}).get(clf_name)
            row += f"| {fmt(v)} "
        row += "|"
        L.append(row)
    L.append("")
    L.append("### Per-Subject LOSO (Best Config per Pipeline)")
    L.append("")
    pipeline_best_lo = {}
    for vkey in ("v1", "v2", "v3"):
        cfg_name, _ = max(
            ((c, max(mi_results[vkey][c]["loso_means"].values())) for c in mi_results[vkey]),
            key=lambda x: x[1])
        clf_name, _ = best_pair(mi_results[vkey][cfg_name]["loso_means"])
        pipeline_best_lo[vkey] = (cfg_name, clf_name)
    header = "| Subject "
    for vkey in ("v1", "v2", "v3"):
        c, k = pipeline_best_lo[vkey]
        header += f"| {vkey} ({c}, {k}) "
    header += "|"
    L.append(header)
    L.append("|" + "|".join(["---"] * 4) + "|")
    for s in all_subs:
        row = f"| {s} "
        for vkey in ("v1", "v2", "v3"):
            cfg_name, clf_name = pipeline_best_lo[vkey]
            v = mi_results[vkey][cfg_name]["loso_per_subject"].get(s, {}).get(clf_name)
            row += f"| {fmt(v)} "
        row += "|"
        L.append(row)
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Blink Detection (Intentional Blink vs Non-Blink)")
    L.append("")
    L.append("### Mean Accuracy (%)")
    L.append("")
    L.append("#### Within-Subject CV")
    L.append("")
    L.append("| Config | Channels | Features | LDA | SVM-lin | SVM-rbf | RF | Threshold | **Best** |")
    L.append("|--------|----------|----------|-----|---------|---------|-----|-----------|----------|")
    for cfg_name, cfg in blink_results.items():
        m = cfg["ws_means"]
        bn, bv = best_pair(m)
        L.append(f"| {cfg_name} | {', '.join(cfg['channels'])} | {cfg['n_features']} | "
                 f"{fmt(m['LDA'])} | {fmt(m['SVM_linear'])} | {fmt(m['SVM_rbf'])} | "
                 f"{fmt(m['RF'])} | {fmt(m['Threshold'])} | {bn} {bv*100:.1f} |")
    L.append("")
    L.append("#### LOSO")
    L.append("")
    L.append("| Config | LDA | SVM-lin | SVM-rbf | RF | Threshold | **Best** |")
    L.append("|--------|-----|---------|---------|-----|-----------|----------|")
    for cfg_name, cfg in blink_results.items():
        m = cfg["loso_means"]
        bn, bv = best_pair(m)
        L.append(f"| {cfg_name} | {fmt(m['LDA'])} | {fmt(m['SVM_linear'])} | {fmt(m['SVM_rbf'])} | "
                 f"{fmt(m['RF'])} | {fmt(m['Threshold'])} | {bn} {bv*100:.1f} |")
    L.append("")
    # Best blink overall
    bw = max(((c, *best_pair(blink_results[c]["ws_means"])) for c in blink_results), key=lambda x: x[2])
    bl = max(((c, *best_pair(blink_results[c]["loso_means"])) for c in blink_results), key=lambda x: x[2])
    L.append("### Best Blink Detection Results")
    L.append("")
    L.append("| Metric | Config | Classifier | Accuracy |")
    L.append("|--------|--------|------------|----------|")
    L.append(f"| Best within-subject | {bw[0]} | {bw[1]} | **{bw[2]*100:.1f}%** |")
    L.append(f"| Best LOSO | {bl[0]} | {bl[1]} | **{bl[2]*100:.1f}%** |")
    L.append("")
    L.append("### Per-Subject Blink Detection (all_4ch, LDA)")
    L.append("")
    L.append("| Subject | Within-Subject LDA | LOSO LDA |")
    L.append("|---------|--------------------|----------|")
    for s in all_subs:
        ws_v = blink_results["all_4ch"]["ws_per_subject"].get(s, {}).get("LDA")
        lo_v = blink_results["all_4ch"]["loso_per_subject"].get(s, {}).get("LDA")
        L.append(f"| {s} | {fmt(ws_v)} | {fmt(lo_v)} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Channel Quality (Composite Score, 0–1)")
    L.append("")
    L.append("| Subject | TP9 | AF7 | AF8 | TP10 |")
    L.append("|---------|-----|-----|-----|------|")
    for s in sorted(channel_q):
        q = channel_q[s]
        L.append(f"| {s} | {q['TP9']:.2f} | {q['AF7']:.2f} | {q['AF8']:.2f} | {q['TP10']:.2f} |")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading Phase 1 from {PHASE1_DIR}")
    subjects = load_phase1()
    print(f"  {len(subjects)} subjects: {list(subjects.keys())}")

    print("\nChannel quality:")
    cq = channel_quality_table(subjects)
    for s, q in cq.items():
        print(f"  {s}: " + " ".join(f"{k}={v:.2f}" for k, v in q.items()))

    auto_ch = auto_channels(cq)
    print(f"\nAuto-selected channels (mean composite): {[CH_NAMES[i] for i in auto_ch]}")
    mi_lookup = {"temporal_only": TEMPORAL_CH, "all_4ch": ALL_CH, "auto": auto_ch}

    print("\n=== Motor imagery ===")
    mi = eval_mi(subjects, mi_lookup)
    mi_n_epochs = next(iter(next(iter(mi.values())).values()))["n_epochs"]

    print("\n=== Blink ===")
    blink = eval_blink(subjects)
    blink_n_epochs = next(iter(blink.values()))["n_epochs"]

    print("\nWriting results.md and JSON")
    md = render_md(mi, blink, cq, len(subjects), mi_n_epochs, blink_n_epochs)
    (OUT_DIR / "results.md").write_text(md)
    json_payload = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "Phase1",
        "subjects": list(subjects.keys()),
        "channel_quality": cq,
        "auto_channels": [CH_NAMES[i] for i in auto_ch],
        "mi": mi,
        "blink": blink,
    }
    (OUT_DIR / "results_data.json").write_text(json.dumps(json_payload, indent=2))
    print(f"Wrote {OUT_DIR / 'results.md'}")
    print(f"Wrote {OUT_DIR / 'results_data.json'}")


if __name__ == "__main__":
    main()
