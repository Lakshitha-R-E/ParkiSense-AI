"""
ParkiSense AI - Preprocessing & Feature Extraction
===================================================
Standalone pipelines for three physiological modalities:
  - Voice (audio) → 52-dim feature vector
  - EEG           → PSD band powers + inter-channel coherence
  - Gait (IMU)    → 42-dim spatiotemporal features
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
VOICE_SR: int = 22_050          # Target sample rate (Hz)
VOICE_CLIP_DUR: float = 3.0     # Clip length (s)
VOICE_OVERLAP: float = 0.5      # Overlap fraction
VOICE_N_MFCC: int = 13          # Number of MFCC coefficients

EEG_LOWCUT: float = 1.0         # Bandpass low (Hz)
EEG_HIGHCUT: float = 45.0       # Bandpass high (Hz)
EEG_SEG_DUR: float = 4.0        # Segment length (s)
EEG_OVERLAP: float = 0.75       # Overlap fraction
EEG_BANDS: dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

GAIT_FS: int = 100              # IMU sample rate (Hz)
GAIT_LP_CUTOFF: float = 10.0    # Low-pass cutoff (Hz)
GAIT_LP_ORDER: int = 4          # Butterworth order


# ─────────────────────────────────────────────────────────────────────────────
# VOICE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _segment_audio(y: np.ndarray, sr: int, clip_dur: float, overlap: float) -> list[np.ndarray]:
    """Slice a 1-D audio array into overlapping fixed-length clips."""
    clip_len = int(clip_dur * sr)
    hop = int(clip_len * (1 - overlap))
    clips = []
    start = 0
    while start + clip_len <= len(y):
        clips.append(y[start: start + clip_len])
        start += hop
    return clips


def extract_voice_features(
    audio_path: str,
    target_sr: int = VOICE_SR,
    clip_dur: float = VOICE_CLIP_DUR,
    overlap: float = VOICE_OVERLAP,
    n_mfcc: int = VOICE_N_MFCC,
) -> np.ndarray:
    """
    Extract a 52-dimensional voice feature vector from a WAV file.

    Feature layout (52 dims):
      - [0:13]  : 13 MFCCs (mean across time)
      - [13:26] : Delta MFCCs
      - [26:39] : Delta-delta MFCCs
      - [39]    : F0 (fundamental frequency, mean)
      - [40]    : Jitter (local, %)
      - [41]    : Shimmer (local, dB)
      - [42]    : HNR (Harmonics-to-Noise Ratio, dB)
      Remaining [43:52]: zero-padded to ensure fixed dim = 52 (reserved).

    Parameters
    ----------
    audio_path : str
        Path to the input `.wav` file.
    target_sr : int
        Target sample rate in Hz (default 22,050).
    clip_dur : float
        Clip duration in seconds (default 3.0).
    overlap : float
        Fractional overlap between clips (default 0.5).
    n_mfcc : int
        Number of MFCC coefficients to extract (default 13).

    Returns
    -------
    np.ndarray of shape (52,)
        Mean feature vector aggregated over all valid clips.
    """
    try:
        import librosa
        import parselmouth
        from parselmouth.praat import call
    except ImportError as e:
        raise ImportError(
            "Voice processing requires 'librosa' and 'praat-parselmouth'. "
            f"Install via requirements.txt. Original error: {e}"
        )

    # Load & resample
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    clips = _segment_audio(y, sr, clip_dur, overlap)

    if not clips:
        logger.warning("Audio too short to extract any clip; returning zeros.")
        return np.zeros(52, dtype=np.float32)

    clip_features = []
    for clip in clips:
        # ── MFCCs + deltas ──────────────────────────────────────────────────
        mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=n_mfcc)  # (13, T)
        mfcc_mean = mfcc.mean(axis=1)                                # (13,)
        delta_mfcc = librosa.feature.delta(mfcc).mean(axis=1)       # (13,)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2).mean(axis=1)  # (13,)

        # ── Praat-based prosodic features ───────────────────────────────────
        snd = parselmouth.Sound(clip, sampling_frequency=sr)

        # F0 (fundamental frequency)
        try:
            pitch = call(snd, "To Pitch", 0.0, 75, 600)
            f0_mean = float(call(pitch, "Get mean", 0, 0, "Hertz"))
            if np.isnan(f0_mean):
                f0_mean = 0.0
        except Exception:
            f0_mean = 0.0

        # PointProcess for jitter & shimmer
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)

        # Jitter (local, %)
        try:
            jitter = call(
                [snd, point_process],
                "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
            )
        except Exception:
            jitter = 0.0

        # Shimmer (local, dB)
        try:
            shimmer = call(
                [snd, point_process],
                "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            )
        except Exception:
            shimmer = 0.0

        # HNR (Harmonics-to-Noise Ratio)
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
        except Exception:
            hnr = 0.0

        # ── Concatenate to 52-dim vector ────────────────────────────────────
        feat = np.zeros(52, dtype=np.float32)
        feat[0:13] = mfcc_mean.astype(np.float32)
        feat[13:26] = delta_mfcc.astype(np.float32)
        feat[26:39] = delta2_mfcc.astype(np.float32)
        feat[39] = f0_mean
        feat[40] = float(jitter) if not np.isnan(jitter) else 0.0
        feat[41] = float(shimmer) if not np.isnan(shimmer) else 0.0
        feat[42] = float(hnr) if not np.isnan(hnr) else 0.0
        # feat[43:52] reserved / zero-padded

        clip_features.append(feat)

    return np.mean(clip_features, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# EEG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_eeg_features(
    edf_path: str,
    lowcut: float = EEG_LOWCUT,
    highcut: float = EEG_HIGHCUT,
    seg_dur: float = EEG_SEG_DUR,
    overlap: float = EEG_OVERLAP,
) -> np.ndarray:
    """
    Extract PSD band powers + inter-channel coherence from a raw EEG .edf file.

    Processing steps:
      1. Load via MNE, bandpass filter 1–45 Hz.
      2. Run ICA (n_components=15) to remove eye-blink & muscle artifacts.
      3. Segment into 4-second windows with 75 % overlap.
      4. Compute PSD for each band × channel.
      5. Compute pairwise coherence between adjacent channels.

    Parameters
    ----------
    edf_path : str
        Path to the EEG `.edf` file.
    lowcut, highcut : float
        Bandpass frequency limits (Hz).
    seg_dur : float
        Segment length in seconds.
    overlap : float
        Fractional overlap between segments.

    Returns
    -------
    np.ndarray
        1-D feature vector. Shape depends on channel count:
        n_bands * n_channels + n_channel_pairs for the first segment average.
    """
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "EEG processing requires 'mne'. "
            f"Install via requirements.txt. Original error: {e}"
        )

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mne.set_log_level("WARNING")

    # Load raw EDF
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    n_channels = len(raw.ch_names)

    # Bandpass filter
    raw.filter(
        l_freq=lowcut,
        h_freq=highcut,
        method="fir",
        fir_window="hamming",
        verbose=False,
    )

    # ICA artifact removal
    ica = mne.preprocessing.ICA(
        n_components=min(15, n_channels - 1),
        random_state=42,
        max_iter="auto",
    )
    ica.fit(raw, verbose=False)
    # Auto-detect eye blink components
    try:
        eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_indices
    except Exception:
        pass  # No EOG channel available; skip auto-detection
    raw = ica.apply(raw, verbose=False)

    # Segment data
    data = raw.get_data()  # (n_channels, n_times)
    seg_len = int(seg_dur * sfreq)
    hop = int(seg_len * (1 - overlap))

    all_seg_feats = []
    start = 0
    while start + seg_len <= data.shape[1]:
        segment = data[:, start: start + seg_len]  # (n_ch, seg_len)

        # ── PSD band powers ─────────────────────────────────────────────────
        band_powers = []
        for (band_low, band_high) in EEG_BANDS.values():
            freqs, psd = signal.welch(segment, fs=sfreq, nperseg=min(256, seg_len))
            band_mask = (freqs >= band_low) & (freqs < band_high)
            power = psd[:, band_mask].mean(axis=1)  # (n_ch,)
            band_powers.append(power)
        band_feat = np.concatenate(band_powers)  # (n_bands * n_ch,)

        # ── Inter-channel coherence ─────────────────────────────────────────
        coherence_vals = []
        for i in range(n_channels - 1):
            _, coh = signal.coherence(segment[i], segment[i + 1], fs=sfreq)
            coherence_vals.append(coh.mean())
        coh_feat = np.array(coherence_vals, dtype=np.float32)

        seg_feat = np.concatenate([band_feat, coh_feat]).astype(np.float32)
        all_seg_feats.append(seg_feat)
        start += hop

    if not all_seg_feats:
        logger.warning("EEG recording too short; returning zeros.")
        return np.zeros(5 * n_channels + (n_channels - 1), dtype=np.float32)

    return np.mean(all_seg_feats, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# GAIT (IMU) PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _butterworth_lowpass(
    data: np.ndarray, cutoff: float, fs: int, order: int = 4
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter along axis -1."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data).astype(np.float32)


def _detect_steps(acc_z: np.ndarray, fs: int) -> np.ndarray:
    """Detect step events via peak-finding on vertical accelerometer signal."""
    min_distance = int(0.3 * fs)  # at least 300 ms between steps
    peaks, _ = signal.find_peaks(
        acc_z,
        height=np.mean(acc_z) + 0.5 * np.std(acc_z),
        distance=min_distance,
    )
    return peaks


def extract_gait_features(
    gait_csv_path: str,
    fs: int = GAIT_FS,
    lp_cutoff: float = GAIT_LP_CUTOFF,
    lp_order: int = GAIT_LP_ORDER,
) -> np.ndarray:
    """
    Extract a 42-dimensional spatiotemporal gait feature vector from CSV IMU data.

    Expected CSV columns: ``time, acc_x, acc_y, acc_z`` (vertical = acc_z).

    Feature layout (42 dims):
      [0]   Cadence (steps/min)
      [1]   Mean step duration (s)
      [2]   Std step duration (s)
      [3]   Mean double-support time (s)
      [4]   Gait asymmetry (ratio)
      [5]   Coefficient of variation of step duration
      [6]   Mean stride length proxy (normalised integral)
      [7]   Stride length std
      [8:18] Percentiles (10, 20, …, 100) of filtered acc_z
      [18:28] Percentiles of step peak amplitudes
      [28:42] 14 additional statistical descriptors (skewness, kurtosis,
               zero-crossing rate, RMS per 1-s window mean/std, etc.)

    Parameters
    ----------
    gait_csv_path : str
        CSV file with IMU data.
    fs : int
        Sampling rate of the IMU (default 100 Hz).
    lp_cutoff : float
        Low-pass cut-off frequency in Hz (default 10 Hz).
    lp_order : int
        Butterworth filter order (default 4).

    Returns
    -------
    np.ndarray of shape (42,)
    """
    import pandas as pd
    from scipy.stats import skew, kurtosis

    df = pd.read_csv(gait_csv_path)

    # ── Calibration: zero-mean vertical accelerometer ───────────────────────
    if "acc_z" not in df.columns:
        raise ValueError("Gait CSV must contain 'acc_z' column (vertical axis).")
    acc_z_raw = df["acc_z"].to_numpy(dtype=np.float64)
    acc_z_raw -= np.mean(acc_z_raw)  # Remove gravity offset

    # ── Low-pass Butterworth filter ─────────────────────────────────────────
    acc_z = _butterworth_lowpass(acc_z_raw, lp_cutoff, fs, lp_order)

    # ── Step detection ──────────────────────────────────────────────────────
    steps = _detect_steps(acc_z, fs)
    n_steps = len(steps)

    if n_steps < 2:
        logger.warning("Fewer than 2 steps detected; gait features set to zero.")
        return np.zeros(42, dtype=np.float32)

    total_time_s = len(acc_z) / fs
    step_durations = np.diff(steps) / fs            # (n_steps-1,)

    cadence = (n_steps / total_time_s) * 60         # steps/min
    mean_step_dur = float(np.mean(step_durations))
    std_step_dur = float(np.std(step_durations))
    cv_step = std_step_dur / (mean_step_dur + 1e-8)

    # Double support time proxy: half the stride interval
    mean_dbl_support = mean_step_dur * 0.4
    gait_asymmetry = float(
        np.std(step_durations[::2]) / (np.mean(step_durations[::2]) + 1e-8)
    ) if len(step_durations) > 2 else 0.0

    # Stride length proxy: trapezoidal integral of |acc| per stride
    stride_lengths = []
    for i in range(0, len(steps) - 1, 2):
        seg = np.abs(acc_z[steps[i]: steps[i + 1]])
        stride_lengths.append(float(np.trapz(seg) / fs))
    mean_stride = float(np.mean(stride_lengths)) if stride_lengths else 0.0
    std_stride = float(np.std(stride_lengths)) if stride_lengths else 0.0

    # Percentiles of acc_z signal
    acc_pcts = np.percentile(acc_z, np.arange(10, 110, 10)).astype(np.float32)  # 10 vals

    # Percentiles of step peak amplitudes
    peak_amps = acc_z[steps]
    peak_pcts = np.percentile(peak_amps, np.arange(10, 110, 10)).astype(np.float32)

    # Additional statistical descriptors (14 values)
    rms_windows = [
        np.sqrt(np.mean(acc_z[i: i + fs] ** 2))
        for i in range(0, len(acc_z) - fs, fs)
    ]
    rms_windows = rms_windows if rms_windows else [0.0]
    zero_crossings = float(np.sum(np.diff(np.sign(acc_z)) != 0)) / total_time_s

    extra = np.array([
        skew(acc_z),                                  # 0
        kurtosis(acc_z),                              # 1
        zero_crossings,                               # 2
        float(np.mean(rms_windows)),                  # 3
        float(np.std(rms_windows)),                   # 4
        float(np.max(acc_z)),                         # 5
        float(np.min(acc_z)),                         # 6
        float(np.ptp(acc_z)),                         # 7 peak-to-peak
        float(np.median(step_durations)),             # 8
        float(np.percentile(step_durations, 10)),     # 9
        float(np.percentile(step_durations, 90)),     # 10
        float(n_steps),                               # 11 total step count
        mean_stride,                                  # 12
        std_stride,                                   # 13
    ], dtype=np.float32)

    feat = np.zeros(42, dtype=np.float32)
    feat[0] = cadence
    feat[1] = mean_step_dur
    feat[2] = std_step_dur
    feat[3] = mean_dbl_support
    feat[4] = gait_asymmetry
    feat[5] = cv_step
    feat[6] = mean_stride
    feat[7] = std_stride
    feat[8:18] = acc_pcts
    feat[18:28] = peak_pcts
    feat[28:42] = extra

    return feat
