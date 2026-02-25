"""Motor Imagery Classification Pipeline — v3: Time-Frequency + EEGNet Features.

Builds on v2 by adding an EEGNet-inspired fixed (non-trained) feature extractor.
The EEGNet architecture mirrors classical EEG processing:
  temporal filtering (sinc bandpass) → spatial filtering (L-R contrasts)
  → temporal summarization → feature vector

All weights are analytically initialized and frozen — no gradient-based training.

New features vs v2:
  - EEGNet forward-pass features from biologically-informed architecture

Requires: torch>=2.0.0

Run:
  source /opt/anaconda3/etc/profile.d/conda.sh && conda activate muse
  python src/training/eegnet/train_motor_imagery_v3_eegnet.py
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew, linregress
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
import mne
mne.set_log_level("ERROR")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_filters import EEGFilter

# ---------- constants ----------
DATA_DIR = PROJECT_ROOT / "data"
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]
CH_COLS = ["EEG_0", "EEG_1", "EEG_2", "EEG_3"]
MI_TYPES = ["motor_imagery_left", "motor_imagery_right"]

BANDS = {
    "mu": (8, 12),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
}

TIMEFREQ_BANDS = {
    "mu": (8, 12),
    "beta": (13, 30),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
}

CHANNEL_CONFIGS = {
    "temporal_only": [0, 3],
    "all_4ch": [0, 1, 2, 3],
}

MIN_EPOCH_SECONDS = 2.0
MAX_EPOCH_SECONDS = 5.0
ARTIFACT_THRESHOLD_UV = 200.0
EEGNET_FIXED_LENGTH = int(5.0 * 256)  # 1280 samples at 256 Hz


# ============================================================
# Data loading (same as v1)
# ============================================================

def find_subject_data(data_dir):
    subjects = {}
    for sub_dir in sorted(data_dir.glob("sub*")):
        eeg_files = list(sub_dir.glob("eeg_data_*.csv"))
        trial_files = list(sub_dir.glob("trial_log_*.csv"))
        if eeg_files and trial_files:
            meta_file = sub_dir / "metadata.yaml"
            name = sub_dir.name
            if meta_file.exists():
                for line in meta_file.read_text().splitlines():
                    if line.startswith("participant:"):
                        name = f"{sub_dir.name} ({line.split(':')[1].strip()})"
            subjects[sub_dir.name] = {
                "dir": sub_dir, "eeg": eeg_files[0],
                "trials": trial_files[0], "label": name,
            }
    return subjects


def load_subject(info):
    eeg = pd.read_csv(info["eeg"])
    trials = pd.read_csv(info["trials"])
    dt = np.diff(eeg["timestamp"].values[:1000])
    fs = 1.0 / np.median(dt)
    return eeg, trials, fs


# ============================================================
# Channel quality (same as v1)
# ============================================================

def compute_channel_quality(eeg, trials, fs, ch_indices=None):
    if ch_indices is None:
        ch_indices = list(range(4))
    mi_trials = trials[trials["trial_type"].isin(MI_TYPES)]
    quality = {}
    for ch_idx in ch_indices:
        col = CH_COLS[ch_idx]
        name = CH_NAMES[ch_idx]
        segments = []
        for _, trial in mi_trials.iterrows():
            mask = (eeg["timestamp"] >= trial["start_time"]) & (
                eeg["timestamp"] <= trial["end_time"])
            seg = eeg.loc[mask, col].values
            if len(seg) > int(fs):
                segments.append(seg)
        if not segments:
            quality[name] = {"rms": np.nan, "kurtosis": np.nan,
                             "spectral_flatness": np.nan, "mu_snr": np.nan,
                             "artifact_rate": np.nan, "composite": 0.0}
            continue
        all_data = np.concatenate(segments)
        rms = np.sqrt(np.mean(all_data ** 2))
        kurt = kurtosis(all_data)
        psds = []
        for seg in segments:
            nperseg = min(int(fs * 2), len(seg))
            if nperseg >= int(fs * 0.5):
                freqs, psd = welch(seg, fs=fs, nperseg=nperseg)
                band_mask = (freqs >= 1) & (freqs <= 40)
                if np.any(band_mask):
                    psds.append(psd[band_mask])
        if psds:
            min_len = min(len(p) for p in psds)
            mean_psd = np.mean([p[:min_len] for p in psds], axis=0)
            log_psd = np.log(mean_psd + 1e-12)
            spectral_flatness = np.exp(np.mean(log_psd)) / (np.mean(mean_psd) + 1e-12)
        else:
            spectral_flatness = 1.0
        mu_snrs = []
        for seg in segments:
            nperseg = min(int(fs * 2), len(seg))
            if nperseg >= int(fs * 0.5):
                freqs, psd = welch(seg, fs=fs, nperseg=nperseg)
                mu_mask = (freqs >= 8) & (freqs <= 12)
                total_mask = (freqs >= 1) & (freqs <= 40)
                total_power = np.trapz(psd[total_mask], freqs[total_mask])
                if total_power > 0:
                    mu_power = np.trapz(psd[mu_mask], freqs[mu_mask])
                    mu_snrs.append(mu_power / total_power)
        mu_snr = np.mean(mu_snrs) if mu_snrs else 0.0
        artifact_rate = np.mean(np.abs(all_data) > ARTIFACT_THRESHOLD_UV)
        rms_score = np.clip(1.0 - rms / 500.0, 0, 1)
        kurt_score = np.clip(1.0 - abs(kurt) / 20.0, 0, 1)
        flat_score = np.clip(1.0 - spectral_flatness, 0, 1)
        artifact_score = 1.0 - artifact_rate
        mu_score = np.clip(mu_snr * 5.0, 0, 1)
        composite = (rms_score * 0.25 + kurt_score * 0.15 + flat_score * 0.2 +
                     artifact_score * 0.2 + mu_score * 0.2)
        quality[name] = {
            "rms": rms, "kurtosis": kurt,
            "spectral_flatness": spectral_flatness,
            "mu_snr": mu_snr, "artifact_rate": artifact_rate,
            "composite": composite,
        }
    return quality


def select_channels_by_quality(quality, min_channels=2):
    scores = {ch: q["composite"] for ch, q in quality.items()
              if not np.isnan(q["composite"])}
    if len(scores) <= min_channels:
        return list(range(4))
    median_score = np.median(list(scores.values()))
    selected = [i for i, ch in enumerate(CH_NAMES)
                if ch in scores and scores[ch] >= median_score]
    if len(selected) < min_channels:
        sorted_chs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [CH_NAMES.index(ch) for ch, _ in sorted_chs[:min_channels]]
    return sorted(selected)


# ============================================================
# Epoch extraction (same as v1)
# ============================================================

def extract_mi_epochs(eeg, trials, fs, ch_indices, eeg_filter):
    mi_trials = trials[trials["trial_type"].isin(MI_TYPES)]
    epochs, labels, trial_ids = [], [], []
    min_samples = int(MIN_EPOCH_SECONDS * fs)
    max_samples = int(MAX_EPOCH_SECONDS * fs)

    for _, trial in mi_trials.iterrows():
        mask = (eeg["timestamp"] >= trial["start_time"]) & (
            eeg["timestamp"] <= trial["end_time"])
        segment = eeg.loc[mask, [CH_COLS[i] for i in ch_indices]].values
        if len(segment) < min_samples:
            continue
        if len(segment) > max_samples:
            segment = segment[:max_samples]
        processed = np.zeros_like(segment, dtype=np.float64)
        n_clean_channels = 0
        for col_i in range(segment.shape[1]):
            ch_data = segment[:, col_i].astype(np.float64)
            ch_data = eeg_filter.apply_moving_artifact_removal(ch_data, threshold=3.0)
            ch_data = eeg_filter.apply_comprehensive_filter(ch_data)
            processed[:, col_i] = ch_data
            if np.std(ch_data) < 500:
                n_clean_channels += 1
        if n_clean_channels < min(2, segment.shape[1]):
            continue
        label = 0 if trial["trial_type"] == "motor_imagery_left" else 1
        epochs.append(processed)
        labels.append(label)
        trial_ids.append(trial["trial_id"])

    return epochs, labels, trial_ids


# ============================================================
# Feature extraction — v1 spectral/time-domain
# ============================================================

def _bandpower(signal, fs, fmin, fmax):
    nperseg = min(int(fs * 2), len(signal))
    if nperseg < int(fs * 0.5):
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return 0.0
    return np.trapz(psd[idx], freqs[idx])


def _relative_bandpower(signal, fs, fmin, fmax):
    nperseg = min(int(fs * 2), len(signal))
    if nperseg < int(fs * 0.5):
        return 0.0
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    band_idx = (freqs >= fmin) & (freqs <= fmax)
    total_idx = (freqs >= 1) & (freqs <= 40)
    total_power = np.trapz(psd[total_idx], freqs[total_idx])
    if total_power == 0:
        return 0.0
    return np.trapz(psd[band_idx], freqs[band_idx]) / total_power


def _hjorth_parameters(signal):
    activity = np.var(signal)
    d1 = np.diff(signal)
    d2 = np.diff(d1)
    var_d1 = np.var(d1)
    var_d2 = np.var(d2)
    mobility = np.sqrt(var_d1 / (activity + 1e-12))
    mob_d1 = np.sqrt(var_d2 / (var_d1 + 1e-12))
    complexity = mob_d1 / (mobility + 1e-12)
    return activity, mobility, complexity


def _zero_crossing_rate(signal):
    return np.mean(np.diff(np.sign(signal)) != 0)


def extract_features_single_epoch(epoch, fs, ch_indices):
    n_ch = epoch.shape[1]
    features = []
    per_ch_powers = {band_name: [] for band_name in BANDS}

    for ch_i in range(n_ch):
        sig = epoch[:, ch_i]
        for band_name, (fmin, fmax) in BANDS.items():
            abs_bp = _bandpower(sig, fs, fmin, fmax)
            rel_bp = _relative_bandpower(sig, fs, fmin, fmax)
            features.extend([abs_bp, rel_bp])
            per_ch_powers[band_name].append(abs_bp)
        activity, mobility, complexity = _hjorth_parameters(sig)
        features.extend([activity, mobility, complexity])
        zcr = _zero_crossing_rate(sig)
        kurt = kurtosis(sig)
        sk = skew(sig)
        features.extend([zcr, kurt, sk])
        alpha_power = _bandpower(sig, fs, 8, 13)
        beta_power = _bandpower(sig, fs, 13, 30)
        ab_ratio = alpha_power / (beta_power + 1e-12)
        features.append(ab_ratio)

    pairs = []
    if 0 in ch_indices and 3 in ch_indices:
        l_idx = ch_indices.index(0)
        r_idx = ch_indices.index(3)
        pairs.append((l_idx, r_idx, "TP9_TP10"))
    if 1 in ch_indices and 2 in ch_indices:
        l_idx = ch_indices.index(1)
        r_idx = ch_indices.index(2)
        pairs.append((l_idx, r_idx, "AF7_AF8"))

    for l_idx, r_idx, pair_name in pairs:
        for band_name in BANDS:
            l_power = per_ch_powers[band_name][l_idx]
            r_power = per_ch_powers[band_name][r_idx]
            denom = r_power + l_power
            asym = (r_power - l_power) / denom if denom > 0 else 0.0
            features.append(asym)

    return np.array(features, dtype=np.float64)


def extract_features_all(epochs, fs, ch_indices):
    if not epochs:
        return np.empty((0, 0))
    X = np.array([extract_features_single_epoch(ep, fs, ch_indices)
                  for ep in epochs])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ============================================================
# v2: Time-frequency feature extraction
# ============================================================

def extract_timefreq_features(epoch, fs, ch_indices):
    """Extract sliding-window time-frequency features from a single epoch.

    Uses a 1s window with 0.5s step to compute band power over time,
    then derives temporal statistics (mean, std, slope, range) that
    capture ERD/ERS dynamics.
    """
    n_samples, n_ch = epoch.shape
    win_samples = int(1.0 * fs)
    step_samples = int(0.5 * fs)

    starts = list(range(0, n_samples - win_samples + 1, step_samples))
    if len(starts) < 2:
        starts = [0]
        win_samples = n_samples

    n_windows = len(starts)
    band_names = list(TIMEFREQ_BANDS.keys())
    n_bands = len(band_names)

    bp_array = np.zeros((n_windows, n_ch, n_bands))
    for w_i, start in enumerate(starts):
        window = epoch[start:start + win_samples]
        for ch_i in range(n_ch):
            sig = window[:, ch_i]
            for b_i, band_name in enumerate(band_names):
                fmin, fmax = TIMEFREQ_BANDS[band_name]
                bp_array[w_i, ch_i, b_i] = _bandpower(sig, fs, fmin, fmax)

    features = []

    for ch_i in range(n_ch):
        for b_i in range(n_bands):
            series = bp_array[:, ch_i, b_i]
            features.append(np.mean(series))
            features.append(np.std(series))
            features.append(np.ptp(series))
            if len(series) >= 2:
                x = np.arange(len(series))
                slope = linregress(x, series).slope
            else:
                slope = 0.0
            features.append(slope)

    pairs = []
    if 0 in ch_indices and 3 in ch_indices:
        l_idx = ch_indices.index(0)
        r_idx = ch_indices.index(3)
        pairs.append((l_idx, r_idx))
    if 1 in ch_indices and 2 in ch_indices:
        l_idx = ch_indices.index(1)
        r_idx = ch_indices.index(2)
        pairs.append((l_idx, r_idx))

    for l_idx, r_idx in pairs:
        for b_i in range(n_bands):
            l_series = bp_array[:, l_idx, b_i]
            r_series = bp_array[:, r_idx, b_i]
            denom = r_series + l_series + 1e-12
            asym_series = (r_series - l_series) / denom
            features.append(np.mean(asym_series))
            features.append(np.std(asym_series))
            if len(asym_series) >= 2:
                x = np.arange(len(asym_series))
                slope = linregress(x, asym_series).slope
            else:
                slope = 0.0
            features.append(slope)

    return np.array(features, dtype=np.float64)


def extract_timefreq_all(epochs, fs, ch_indices):
    if not epochs:
        return np.empty((0, 0))
    X = np.array([extract_timefreq_features(ep, fs, ch_indices)
                  for ep in epochs])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ============================================================
# v3: EEGNet-inspired fixed feature extractor
# ============================================================

def _sinc_bandpass(fmin, fmax, fs, kernel_size):
    """Construct a sinc bandpass filter with Hamming window.

    Returns a 1D numpy array of length kernel_size.
    """
    n = np.arange(kernel_size) - (kernel_size - 1) / 2
    t = n / fs
    # sinc bandpass = highpass(fmin) subtracted from lowpass(fmax)
    h = 2 * fmax * np.sinc(2 * fmax * t) - 2 * fmin * np.sinc(2 * fmin * t)
    h *= np.hamming(kernel_size)
    h /= np.sum(np.abs(h)) + 1e-12
    return h


class EEGNetFeatureExtractor(nn.Module):
    """EEGNet-inspired fixed feature extractor (no training).

    Architecture (adapted for 4-channel Muse 2):
      Block 1 — Temporal Conv: F1=8 sinc bandpass filters, kernel (1, 128)
      Block 1 — Depthwise Spatial Conv: D=2 per temporal filter, kernel (C, 1)
      Block 2 — Separable Conv: depthwise (1, 16) + pointwise (1, 1)
      Average pooling → flatten → feature vector

    All weights are analytically initialized and frozen.

    Args:
        n_channels: number of EEG channels
        n_samples: fixed input length in samples (epochs padded to this)
        fs: sampling rate in Hz
        ch_indices: list of global channel indices (for spatial filter init)
    """

    # Center frequencies for 8 sinc bandpass filters
    FILTER_BANDS = [
        (4, 8),    # theta
        (6, 10),   # theta-alpha transition
        (8, 12),   # mu / alpha
        (10, 14),  # upper alpha / low beta transition
        (13, 20),  # low beta
        (15, 25),  # mid beta
        (20, 30),  # high beta
        (8, 30),   # broad alpha-beta
    ]

    def __init__(self, n_channels, n_samples, fs=256.0, ch_indices=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.fs = fs
        self.ch_indices = ch_indices or list(range(n_channels))

        F1 = 8   # temporal filters
        D = 2    # spatial filters per temporal filter
        F2 = F1 * D  # = 16

        # Block 1: Temporal convolution (sinc bandpass filters)
        temporal_kernel = 128  # 0.5s at 256 Hz
        self.temporal_conv = nn.Conv2d(
            1, F1, (1, temporal_kernel), padding=(0, temporal_kernel // 2),
            bias=False)
        self._init_sinc_filters(temporal_kernel)

        # Block 1: Batch norm after temporal conv
        self.bn1 = nn.BatchNorm2d(F1, affine=False)

        # Block 1: Depthwise spatial convolution
        self.spatial_conv = nn.Conv2d(
            F1, F2, (n_channels, 1), groups=F1, bias=False)
        self._init_spatial_filters()

        self.bn2 = nn.BatchNorm2d(F2, affine=False)

        # Block 2: Separable conv = depthwise temporal + pointwise
        sep_kernel = 16
        self.sep_depthwise = nn.Conv2d(
            F2, F2, (1, sep_kernel), groups=F2,
            padding=(0, sep_kernel // 2), bias=False)
        self._init_sep_depthwise(sep_kernel)

        self.sep_pointwise = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self._init_sep_pointwise()

        self.bn3 = nn.BatchNorm2d(F2, affine=False)

        # Average pooling
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.avg_pool2 = nn.AvgPool2d((1, 8))

        self.elu = nn.ELU()

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Compute output size
        self._output_size = self._compute_output_size()

    def _init_sinc_filters(self, kernel_size):
        """Initialize temporal conv with sinc bandpass filters."""
        weight = self.temporal_conv.weight.data  # (F1, 1, 1, kernel_size)
        for i, (fmin, fmax) in enumerate(self.FILTER_BANDS):
            h = _sinc_bandpass(fmin, fmax, self.fs, kernel_size)
            weight[i, 0, 0, :] = torch.from_numpy(h).float()

    def _init_spatial_filters(self):
        """Initialize depthwise spatial conv with L-R contrast filters.

        For each temporal filter, D=2 spatial filters:
          [0]: L-R temporal contrast [TP9 - TP10] = [1, 0, 0, -1]
          [1]: L-R frontal contrast  [AF7 - AF8]  = [0, 1, -1, 0]

        Adapted for arbitrary channel subsets.
        """
        weight = self.spatial_conv.weight.data  # (F2, 1, n_ch, 1)
        weight.zero_()

        # Map global channel indices to local positions
        ch_map = {g: l for l, g in enumerate(self.ch_indices)}

        for f in range(8):  # F1 = 8 temporal filters
            # First spatial filter: temporal L-R (TP9 - TP10)
            idx_0 = f * 2
            if 0 in ch_map and 3 in ch_map:
                weight[idx_0, 0, ch_map[0], 0] = 1.0   # TP9
                weight[idx_0, 0, ch_map[3], 0] = -1.0   # TP10
            elif 0 in ch_map:
                weight[idx_0, 0, ch_map[0], 0] = 1.0
            elif 3 in ch_map:
                weight[idx_0, 0, ch_map[3], 0] = 1.0
            else:
                # Fallback: uniform average
                weight[idx_0, 0, :, 0] = 1.0 / self.n_channels

            # Second spatial filter: frontal L-R (AF7 - AF8)
            idx_1 = f * 2 + 1
            if 1 in ch_map and 2 in ch_map:
                weight[idx_1, 0, ch_map[1], 0] = 1.0    # AF7
                weight[idx_1, 0, ch_map[2], 0] = -1.0    # AF8
            elif 1 in ch_map:
                weight[idx_1, 0, ch_map[1], 0] = 1.0
            elif 2 in ch_map:
                weight[idx_1, 0, ch_map[2], 0] = 1.0
            else:
                # Fallback: uniform average
                weight[idx_1, 0, :, 0] = 1.0 / self.n_channels

    def _init_sep_depthwise(self, kernel_size):
        """Initialize separable depthwise conv with averaging kernel."""
        weight = self.sep_depthwise.weight.data  # (F2, 1, 1, kernel_size)
        weight.fill_(1.0 / kernel_size)

    def _init_sep_pointwise(self):
        """Initialize pointwise conv as identity."""
        weight = self.sep_pointwise.weight.data  # (F2, F2, 1, 1)
        nn.init.eye_(weight.squeeze())
        weight.copy_(weight.view(weight.shape))

    def _compute_output_size(self):
        """Run a dummy forward pass to determine output feature size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_samples)
            out = self._forward_impl(dummy)
            return out.shape[1]

    def _forward_impl(self, x):
        """Forward pass implementation.

        Args:
            x: (batch, 1, n_channels, n_samples)
        Returns:
            (batch, n_features) flattened feature vector
        """
        # Block 1: temporal filtering
        x = self.temporal_conv(x)
        x = self.bn1(x)

        # Block 1: spatial filtering
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)

        # Block 2: separable conv
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    @property
    def output_size(self):
        return self._output_size


def extract_eegnet_features_all(epochs, fs, ch_indices):
    """Extract EEGNet features from all epochs.

    Each epoch is padded/truncated to EEGNET_FIXED_LENGTH, reshaped to
    (1, 1, n_channels, n_samples), and passed through the frozen EEGNet.

    Returns:
        X_eegnet: (n_epochs, n_eegnet_features) array
    """
    if not epochs:
        return np.empty((0, 0))

    n_ch = epochs[0].shape[1]
    target_len = EEGNET_FIXED_LENGTH

    # Build the model once
    model = EEGNetFeatureExtractor(
        n_channels=n_ch, n_samples=target_len, fs=fs, ch_indices=ch_indices)
    model.eval()

    features_list = []

    with torch.no_grad():
        for ep in epochs:
            # Pad or truncate to fixed length
            n_samples = ep.shape[0]
            if n_samples < target_len:
                pad_width = ((0, target_len - n_samples), (0, 0))
                ep_fixed = np.pad(ep, pad_width, mode="edge")
            else:
                ep_fixed = ep[:target_len]

            # Reshape to (1, 1, n_channels, n_samples) — batch, 1 "image channel",
            # EEG channels, time
            x = torch.from_numpy(ep_fixed.T[np.newaxis, np.newaxis, :, :]).float()
            feat = model(x).numpy().flatten()
            features_list.append(feat)

    X = np.array(features_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ============================================================
# CSP (same as v1)
# ============================================================

def compute_csp(epochs, labels, n_components=None):
    labels = np.array(labels)
    n_ch = epochs[0].shape[1]
    if n_components is None:
        n_components = min(n_ch, max(1, n_ch - 1))
    covs = {0: [], 1: []}
    for ep, lab in zip(epochs, labels):
        C = np.cov(ep.T)
        if C.ndim == 0:
            C = np.array([[C]])
        C /= np.trace(C) + 1e-12
        covs[lab].append(C)
    if not covs[0] or not covs[1]:
        return np.zeros((len(epochs), 1)), np.eye(n_ch)[:, :1]
    C0 = np.mean(covs[0], axis=0)
    C1 = np.mean(covs[1], axis=0)
    C_composite = C0 + C1
    C_composite += np.eye(n_ch) * 1e-6
    try:
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(C0, C_composite)
    except np.linalg.LinAlgError:
        return np.zeros((len(epochs), 1)), np.eye(n_ch)[:, :1]
    sorted_idx = np.argsort(eigenvalues)
    if n_components >= n_ch:
        selected_idx = sorted_idx
    else:
        n_half = n_components // 2
        n_other = n_components - n_half
        selected_idx = np.concatenate([sorted_idx[:n_half], sorted_idx[-n_other:]])
    W = eigenvectors[:, selected_idx]
    csp_features = np.zeros((len(epochs), len(selected_idx)))
    for i, ep in enumerate(epochs):
        projected = ep @ W
        variances = np.var(projected, axis=0)
        csp_features[i] = np.log(variances + 1e-12)
    return csp_features, W


# ============================================================
# Classifiers (same as v1)
# ============================================================

def get_classifiers():
    return {
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "SVM_linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0)),
        ]),
        "SVM_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]),
        "RF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=3,
                random_state=42)),
        ]),
    }


# ============================================================
# Evaluation (same as v1)
# ============================================================

def evaluate_within_subject(X, y, subject_ids, classifiers):
    unique_subs = sorted(set(subject_ids))
    results = {}
    for sub in unique_subs:
        mask = np.array([s == sub for s in subject_ids])
        X_sub, y_sub = X[mask], y[mask]
        if len(np.unique(y_sub)) < 2 or len(y_sub) < 5:
            results[sub] = {name: np.nan for name in classifiers}
            continue
        n_folds = min(5, min(np.bincount(y_sub)))
        if n_folds < 2:
            results[sub] = {name: np.nan for name in classifiers}
            continue
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        results[sub] = {}
        for clf_name, clf in classifiers.items():
            try:
                y_pred = cross_val_predict(clf, X_sub, y_sub, cv=cv)
                acc = accuracy_score(y_sub, y_pred)
            except Exception:
                acc = np.nan
            results[sub][clf_name] = acc
    return results


def evaluate_loso(X, y, subject_ids, classifiers):
    unique_subs = sorted(set(subject_ids))
    results = {}
    for test_sub in unique_subs:
        test_mask = np.array([s == test_sub for s in subject_ids])
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            results[test_sub] = {name: np.nan for name in classifiers}
            continue
        results[test_sub] = {}
        for clf_name, clf in classifiers.items():
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
            except Exception:
                acc = np.nan
            results[test_sub][clf_name] = acc
    return results


# ============================================================
# Main pipeline (v3: spectral + time-frequency + EEGNet + CSP)
# ============================================================

def run_pipeline():
    print("=" * 70)
    print("Motor Imagery Classification Pipeline — v3 (Time-Freq + EEGNet)")
    print("=" * 70)

    subjects = find_subject_data(DATA_DIR)
    if not subjects:
        print("ERROR: No subject data found in", DATA_DIR)
        return

    print(f"\nFound {len(subjects)} subjects: {list(subjects.keys())}")

    subject_data = {}
    for sub_id, info in subjects.items():
        eeg, trials, fs = load_subject(info)
        n_mi = len(trials[trials["trial_type"].isin(MI_TYPES)])
        print(f"  {info['label']}: {n_mi} MI trials, Fs={fs:.0f} Hz")
        subject_data[sub_id] = (eeg, trials, fs, info)

    fs = list(subject_data.values())[0][2]

    # Channel quality assessment
    print(f"\n{'=' * 70}")
    print("Channel Quality Assessment")
    print("=" * 70)

    all_quality = {}
    for sub_id, (eeg, trials, fs_sub, info) in subject_data.items():
        quality = compute_channel_quality(eeg, trials, fs_sub)
        all_quality[sub_id] = quality

    print(f"\n{'Subject':<10}", end="")
    for ch in CH_NAMES:
        print(f"  {ch:>12}", end="")
    print()
    print("-" * 62)
    for sub_id in sorted(all_quality.keys()):
        q = all_quality[sub_id]
        print(f"{sub_id:<10}", end="")
        for ch in CH_NAMES:
            print(f"  {q[ch]['composite']:>12.3f}", end="")
        print()

    for metric in ["rms", "kurtosis", "spectral_flatness", "mu_snr", "artifact_rate"]:
        print(f"\n  {metric}:")
        for sub_id in sorted(all_quality.keys()):
            q = all_quality[sub_id]
            print(f"    {sub_id:<8}", end="")
            for ch in CH_NAMES:
                val = q[ch][metric]
                if metric == "rms":
                    print(f"  {val:>10.1f}", end="")
                else:
                    print(f"  {val:>10.4f}", end="")
            print()

    # Build channel configs including auto
    configs = dict(CHANNEL_CONFIGS)
    auto_channels = set()
    for sub_id, quality in all_quality.items():
        selected = select_channels_by_quality(quality)
        auto_channels.update(selected)
    configs["auto"] = sorted(auto_channels)

    print(f"\nChannel configurations:")
    for config_name, ch_idx in configs.items():
        ch_names = [CH_NAMES[i] for i in ch_idx]
        print(f"  {config_name}: {ch_names}")

    # --- Classification per channel config ---
    all_results = {}

    for config_name, ch_indices in configs.items():
        print(f"\n{'=' * 70}")
        print(f"Config: {config_name} — channels: {[CH_NAMES[i] for i in ch_indices]}")
        print("=" * 70)

        eeg_filter = EEGFilter(sampling_rate=int(fs), notch_freq=60.0)

        all_epochs, all_labels, all_sub_ids = [], [], []

        for sub_id, (eeg, trials, fs_sub, info) in subject_data.items():
            epochs, labels, trial_ids = extract_mi_epochs(
                eeg, trials, fs_sub, ch_indices, eeg_filter)
            print(f"  {sub_id}: {len(epochs)} valid epochs "
                  f"(L={labels.count(0)}, R={labels.count(1)})")
            all_epochs.extend(epochs)
            all_labels.extend(labels)
            all_sub_ids.extend([sub_id] * len(epochs))

        if not all_epochs:
            print("  No valid epochs — skipping config")
            continue

        # Pad epochs to same length for CSP
        max_len = max(ep.shape[0] for ep in all_epochs)
        padded_epochs = []
        for ep in all_epochs:
            if ep.shape[0] < max_len:
                pad_width = ((0, max_len - ep.shape[0]), (0, 0))
                ep = np.pad(ep, pad_width, mode="edge")
            padded_epochs.append(ep)

        y = np.array(all_labels)

        # --- Feature extraction ---
        X_feat = extract_features_all(padded_epochs, fs, ch_indices)
        X_timefreq = extract_timefreq_all(padded_epochs, fs, ch_indices)
        csp_features, W_csp = compute_csp(padded_epochs, all_labels)

        print(f"\n  Extracting EEGNet features (fixed, no training)...")
        X_eegnet = extract_eegnet_features_all(padded_epochs, fs, ch_indices)

        X = np.hstack([X_feat, X_timefreq, csp_features, X_eegnet])

        print(f"\n  Feature matrix: {X.shape[0]} epochs x {X.shape[1]} features")
        print(f"    Spectral/time features: {X_feat.shape[1]}")
        print(f"    Time-frequency features: {X_timefreq.shape[1]}  [v2]")
        print(f"    CSP features: {csp_features.shape[1]}")
        print(f"    EEGNet features: {X_eegnet.shape[1]}  [NEW in v3]")
        print(f"    Class balance: L={np.sum(y == 0)}, R={np.sum(y == 1)}")

        classifiers = get_classifiers()

        # Within-subject CV
        print(f"\n  --- Within-Subject 5-Fold CV ---")
        ws_results = evaluate_within_subject(X, y, all_sub_ids, classifiers)
        for sub_id in sorted(ws_results.keys()):
            accs = ws_results[sub_id]
            parts = [f"{name}: {acc:.1%}" if not np.isnan(acc) else f"{name}: N/A"
                     for name, acc in accs.items()]
            print(f"    {sub_id}: {', '.join(parts)}")

        ws_means = {}
        for clf_name in classifiers:
            vals = [ws_results[s][clf_name] for s in ws_results
                    if not np.isnan(ws_results[s][clf_name])]
            ws_means[clf_name] = np.mean(vals) if vals else np.nan
        print(f"    {'Mean':<6}: ", end="")
        print(", ".join(f"{name}: {acc:.1%}" if not np.isnan(acc) else f"{name}: N/A"
                        for name, acc in ws_means.items()))

        # LOSO CV
        print(f"\n  --- Leave-One-Subject-Out CV ---")
        loso_results = evaluate_loso(X, y, all_sub_ids, classifiers)
        for sub_id in sorted(loso_results.keys()):
            accs = loso_results[sub_id]
            parts = [f"{name}: {acc:.1%}" if not np.isnan(acc) else f"{name}: N/A"
                     for name, acc in accs.items()]
            print(f"    {sub_id} (test): {', '.join(parts)}")

        loso_means = {}
        for clf_name in classifiers:
            vals = [loso_results[s][clf_name] for s in loso_results
                    if not np.isnan(loso_results[s][clf_name])]
            loso_means[clf_name] = np.mean(vals) if vals else np.nan
        print(f"    {'Mean':<6}: ", end="")
        print(", ".join(f"{name}: {acc:.1%}" if not np.isnan(acc) else f"{name}: N/A"
                        for name, acc in loso_means.items()))

        all_results[config_name] = {
            "channels": [CH_NAMES[i] for i in ch_indices],
            "n_epochs": len(all_epochs),
            "n_features": X.shape[1],
            "n_spectral_features": X_feat.shape[1],
            "n_timefreq_features": X_timefreq.shape[1],
            "n_csp_features": csp_features.shape[1],
            "n_eegnet_features": X_eegnet.shape[1],
            "within_subject": {
                sub: {clf: float(f"{acc:.4f}") if not np.isnan(acc) else None
                      for clf, acc in accs.items()}
                for sub, accs in ws_results.items()
            },
            "within_subject_mean": {
                clf: float(f"{acc:.4f}") if not np.isnan(acc) else None
                for clf, acc in ws_means.items()
            },
            "loso": {
                sub: {clf: float(f"{acc:.4f}") if not np.isnan(acc) else None
                      for clf, acc in accs.items()}
                for sub, accs in loso_results.items()
            },
            "loso_mean": {
                clf: float(f"{acc:.4f}") if not np.isnan(acc) else None
                for clf, acc in loso_means.items()
            },
        }

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY — Best Configurations (v3: Time-Freq + EEGNet)")
    print("=" * 70)

    best_ws_acc = 0
    best_ws = ("", "")
    for config, res in all_results.items():
        for clf, acc in res["within_subject_mean"].items():
            if acc is not None and acc > best_ws_acc:
                best_ws_acc = acc
                best_ws = (config, clf)

    print(f"\n  Best within-subject: {best_ws[1]} with {best_ws[0]} "
          f"channels — {best_ws_acc:.1%}")

    best_loso_acc = 0
    best_loso = ("", "")
    for config, res in all_results.items():
        for clf, acc in res["loso_mean"].items():
            if acc is not None and acc > best_loso_acc:
                best_loso_acc = acc
                best_loso = (config, clf)

    print(f"  Best LOSO:           {best_loso[1]} with {best_loso[0]} "
          f"channels — {best_loso_acc:.1%}")

    # Save results
    results_file = DATA_DIR / "classification_results_v3.json"
    output = {
        "pipeline_version": "v3_eegnet",
        "channel_quality": {
            sub: {ch: {k: float(f"{v:.6f}") if isinstance(v, float) else v
                       for k, v in metrics.items()}
                  for ch, metrics in quality.items()}
            for sub, quality in all_quality.items()
        },
        "configs": all_results,
        "best_within_subject": {
            "config": best_ws[0], "classifier": best_ws[1],
            "accuracy": float(f"{best_ws_acc:.4f}"),
        },
        "best_loso": {
            "config": best_loso[0], "classifier": best_loso[1],
            "accuracy": float(f"{best_loso_acc:.4f}"),
        },
    }
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {results_file}")


if __name__ == "__main__":
    run_pipeline()
