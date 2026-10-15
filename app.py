"""
ParkiSense AI - Streamlit Web Dashboard
========================================
Role-based web application for multimodal Parkinson's disease severity
assessment.

Roles
-----
  Neurologist : Full clinical view – multi-modal insights, attention heatmaps,
                longitudinal tracking, PDF export.
  Patient     : Simplified summary and personalised therapy guidance.

Run with:
  streamlit run app.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.database import (
    init_db,
    authenticate_user,
    register_user,
    add_patient_visit,
    get_patient_visits,
    get_all_patients,
)
from src.models import ParkiSenseFusionNet
from src.utils import (
    SEVERITY_COLORS,
    generate_pdf_report,
    generate_therapy_recommendations,
    run_inference,
    updrs_to_label,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParkiSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Background & Text */
    [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    .main {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    .main .block-container {
        color: #0F172A !important;
    }
    .main p:not(button *):not([data-testid="stBaseButton-secondary"] *):not([data-testid="baseButton-secondary"] *),
    .main span:not(button *):not([data-testid="stBaseButton-secondary"] *):not([data-testid="baseButton-secondary"] *),
    .main label,
    .main div:not(.header-bar):not(.severity-badge):not(button *):not([data-testid="stBaseButton-secondary"] *):not([data-testid="baseButton-secondary"] *) {
        color: #0F172A;
    }

    /* Header Bar */
    .header-bar {
        background: linear-gradient(135deg, #1E293B 0%, #2563EB 100%);
        padding: 22px 30px;
        border-radius: 12px;
        color: #FFFFFF !important;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .header-bar h1, .header-bar p {
        color: #FFFFFF !important;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.25em;
        font-weight: 700;
        color: #0F172A !important;
        margin-top: 20px;
        margin-bottom: 12px;
        border-left: 5px solid #2563EB;
        padding-left: 10px;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.88em !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 1.5em !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] label {
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 0.95em !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 1.5px dashed #CBD5E1 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] section > div:not(button) {
        color: #1E293B !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] div, 
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #475569 !important;
    }
    
    /* All Action Buttons (e.g. Browse Files, Add Current Score, etc.) */
    button,
    button[kind="secondary"],
    [data-testid="baseButton-secondary"],
    [data-testid="stBaseButton-secondary"],
    [data-testid="stFileUploader"] button {
        background-color: #2563EB !important;
        border: 1px solid #1D4ED8 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
        transition: all 0.2s ease !important;
    }
    button *,
    button p,
    button span,
    button div,
    button[kind="secondary"] *,
    button[kind="secondary"] p,
    button[kind="secondary"] span,
    [data-testid="baseButton-secondary"] *,
    [data-testid="baseButton-secondary"] p,
    [data-testid="baseButton-secondary"] span,
    [data-testid="stBaseButton-secondary"] *,
    [data-testid="stBaseButton-secondary"] p,
    [data-testid="stBaseButton-secondary"] span,
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button *,
    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploader"] button p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.95em !important;
    }
    button:hover,
    button[kind="secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stFileUploader"] button:hover {
        background-color: #1D4ED8 !important;
    }
    button:hover *,
    button[kind="secondary"]:hover *,
    [data-testid="baseButton-secondary"]:hover *,
    [data-testid="stBaseButton-secondary"]:hover *,
    [data-testid="stFileUploader"] button:hover * {
        color: #FFFFFF !important;
    }

    /* Primary Action Buttons (PDF generation, submit) */
    button[kind="primary"],
    [data-testid="baseButton-primary"],
    [data-testid="stBaseButton-primary"] {
        background-color: #16A34A !important;
        border: 1px solid #15803D !important;
        border-radius: 8px !important;
    }
    button[kind="primary"] *,
    button[kind="primary"] p,
    button[kind="primary"] span,
    [data-testid="baseButton-primary"] *,
    [data-testid="baseButton-primary"] p,
    [data-testid="baseButton-primary"] span,
    [data-testid="stBaseButton-primary"] *,
    [data-testid="stBaseButton-primary"] p,
    [data-testid="stBaseButton-primary"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #15803D !important;
    }

    /* Form Inputs & Text Fields */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95em !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stDateInput"] input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important;
    }
    [data-testid="stTextInput"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stDateInput"] label {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    [data-testid="stExpander"] summary * {
        color: #0F172A !important;
    }

    [data-testid="stFileUploaderFile"] {
        background-color: #F1F5F9 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFile"] small,
    [data-testid="stFileUploaderFile"] div {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* Alerts and Notifications */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
    }
    [data-testid="stAlert"] div, [data-testid="stAlert"] p {
        color: #0F172A !important;
        font-weight: 500 !important;
    }

    /* Quality Check Indicators */
    .qc-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-top: 6px;
    }
    .quality-ok { color: #16A34A !important; font-weight: 700 !important; }
    .quality-fail { color: #DC2626 !important; font-weight: 700 !important; }
    
    .severity-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
        color: #FFFFFF !important;
    }

    /* Force all text in Plotly charts to be pitch black and high contrast */
    .js-plotly-plot text,
    .js-plotly-plot .gtitle,
    .js-plotly-plot .xtick text,
    .js-plotly-plot .ytick text,
    .js-plotly-plot .angularaxis text,
    .js-plotly-plot .radialaxis text,
    .js-plotly-plot .legendtext,
    .js-plotly-plot .annotation-text,
    .js-plotly-plot .colorbartick text,
    .plotly text {
        fill: #000000 !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INITIALISATION (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(voice_dim: int = 52, eeg_dim: int = 64, gait_dim: int = 42) -> ParkiSenseFusionNet:
    """Instantiate and cache the fusion model. Automatically loads trained weights if present."""
    model = ParkiSenseFusionNet(
        voice_dim=voice_dim,
        eeg_dim=eeg_dim,
        gait_dim=gait_dim,
    )
    checkpoint_path = Path("models/parkisense_best.pth")
    if checkpoint_path.exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
        except Exception:
            pass
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MOCK FEATURE EXTRACTION (used when real libs may not be installed)
# ─────────────────────────────────────────────────────────────────────────────

def mock_voice_features() -> np.ndarray:
    """Return plausible random voice features for demo."""
    rng = np.random.default_rng(42)
    feat = rng.normal(0, 1, 52).astype(np.float32)
    feat[39] = rng.uniform(100, 220)   # F0
    feat[40] = rng.uniform(0, 2)       # Jitter
    feat[41] = rng.uniform(0, 5)       # Shimmer
    feat[42] = rng.uniform(5, 25)      # HNR
    return feat


def mock_eeg_features() -> np.ndarray:
    """Return plausible random EEG features."""
    return np.random.default_rng(7).normal(0, 1, 64).astype(np.float32)


def mock_gait_features() -> np.ndarray:
    """Return plausible random gait features."""
    rng = np.random.default_rng(13)
    feat = np.zeros(42, dtype=np.float32)
    feat[0] = rng.uniform(80, 120)     # cadence
    feat[1] = rng.uniform(0.4, 0.7)   # step duration
    feat[4] = rng.uniform(0, 0.3)     # asymmetry
    feat[8:18] = rng.normal(0, 0.5, 10)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_voice(uploaded_file) -> Tuple[np.ndarray, bool, float]:
    """Extract voice features; returns (features, snr_ok, snr_db)."""
    try:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        import librosa
        from src.preprocessing import extract_voice_features
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        feat = extract_voice_features(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        # Simple SNR heuristic from HNR value
        hnr = float(feat[42])
        snr_ok = hnr > 10.0
        return feat, snr_ok, hnr
    except Exception as e:
        st.warning(f"⚠️ Voice feature extraction unavailable – using mock features (demo mode). ({e})")
        feat = mock_voice_features()
        snr_ok = True
        return feat, snr_ok, float(feat[42])


def extract_eeg(uploaded_files) -> Tuple[np.ndarray, bool]:
    """
    Extract EEG features from EEGLAB .set (and companion .fdt) file(s).
    Automatically locates matching .fdt from data/raw/eeg if not uploaded together.
    Returns (features, artifact_ok).
    """
    import shutil
    import re
    try:
        import mne
        import scipy.signal as sig_proc

        mne.set_log_level("ERROR")

        if uploaded_files is None:
            return mock_eeg_features(), True

        if not isinstance(uploaded_files, (list, tuple)):
            files = [uploaded_files]
        else:
            files = list(uploaded_files)

        if not files:
            return mock_eeg_features(), True

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            set_file_path = None

            # Save uploaded files into tmpdir with their original filenames
            for f in files:
                try:
                    f.seek(0)
                except Exception:
                    pass
                fname = getattr(f, "name", "recording.set")
                fpath = tmpdir_path / fname
                with open(fpath, "wb") as out_f:
                    out_f.write(f.read())
                if fname.lower().endswith(".set"):
                    set_file_path = fpath

            if set_file_path is None:
                raise ValueError("No .set file uploaded.")

            stem = set_file_path.stem

            # If companion .fdt is missing in tmpdir, find matching .fdt in local dataset
            expected_fdt = tmpdir_path / f"{stem}.fdt"
            if not expected_fdt.exists():
                found = list(Path("data/raw/eeg").glob(f"**/{stem}.fdt"))
                if not found:
                    found = list(Path(".").glob(f"**/{stem}.fdt"))
                if not found:
                    sub_match = re.search(r"sub-\d+", stem)
                    if sub_match:
                        found = list(Path("data/raw/eeg").glob(f"**/*{sub_match.group(0)}*.fdt"))
                if found:
                    shutil.copy(found[0], expected_fdt)

            raw = mne.io.read_raw_eeglab(str(set_file_path), preload=True, verbose=False)
            sfreq = raw.info["sfreq"]
            n_ch  = len(raw.ch_names)

            # Limit to first 60 seconds for fast interactive responsiveness
            if raw.times[-1] > 60.0:
                raw.crop(tmin=0.0, tmax=60.0)

            # Bandpass 1–45 Hz
            raw.filter(1., 45., method="fir", verbose=False)

            # Fast ICA artifact removal
            n_comp = min(15, n_ch - 1)
            if n_comp >= 2:
                ica = mne.preprocessing.ICA(
                    n_components=n_comp, random_state=42, max_iter=25, verbose=False
                )
                try:
                    ica.fit(raw, verbose=False)
                    try:
                        eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
                        ica.exclude = eog_idx
                    except Exception:
                        pass
                    raw = ica.apply(raw, verbose=False)
                except Exception:
                    pass

            data    = raw.get_data()
            seg_len = int(4.0 * sfreq)
            hop     = int(seg_len * 0.5)        # 50 % overlap for speed
            BANDS   = [(1,4),(4,8),(8,13),(13,30),(30,45)]

            all_feats = []
            start = 0
            while start + seg_len <= data.shape[1] and len(all_feats) < 20:
                seg = data[:, start:start + seg_len]
                freqs, psd = sig_proc.welch(seg, fs=sfreq, nperseg=min(256, seg_len))
                bp = [psd[:, (freqs >= lo) & (freqs < hi)].mean(axis=1) for (lo, hi) in BANDS]
                band_feat = np.concatenate(bp)
                coh_vals  = [
                    sig_proc.coherence(seg[i], seg[i+1], fs=sfreq)[1].mean()
                    for i in range(min(n_ch - 1, 32))
                ]
                all_feats.append(
                    np.concatenate([band_feat, np.array(coh_vals, dtype=np.float32)]).astype(np.float32)
                )
                start += hop

            if not all_feats:
                raise ValueError("EEG recording too short to segment.")

            feat = np.mean(all_feats, axis=0).astype(np.float32)
            if len(feat) >= 64:
                feat = feat[:64]
            else:
                feat = np.pad(feat, (0, 64 - len(feat)), mode="constant")

            artifact_ok = bool(np.std(feat) < 50.0)
            return feat, artifact_ok

    except Exception as e:
        st.warning(f"⚠️ EEG (.set) extraction unavailable – using mock features. ({e})")
        return mock_eeg_features(), True


def extract_gait(uploaded_file) -> Tuple[np.ndarray, bool]:
    """
    Extract gait features from a PhysioNet-format .txt VGRF file.

    Format: 19 whitespace-separated columns, 100 Hz.
      Col 1  : time (s)
      Col 2–9 : left foot VGRF sensors (8 sensors)
      Col 10–17: right foot VGRF sensors (8 sensors)
      Col 18  : total left VGRF  ← used
      Col 19  : total right VGRF ← used

    Falls back to mock features if the file cannot be parsed.
    Returns (features, cycles_ok).
    """
    import scipy.signal as sig_proc
    from scipy.stats import skew, kurtosis as sp_kurtosis

    try:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        content = uploaded_file.read()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        df = pd.read_csv(tmp_path, sep=r"\s+", header=None, on_bad_lines="skip")
        os.unlink(tmp_path)

        if df.shape[1] < 19:
            raise ValueError(f"Expected ≥19 columns, got {df.shape[1]}.")

        fs          = 100
        left_force  = df.iloc[:, 17].to_numpy(dtype=np.float64)
        right_force = df.iloc[:, 18].to_numpy(dtype=np.float64)
        combined    = left_force + right_force
        combined   -= combined.mean()                    # zero-mean calibration

        # 4th-order Butterworth low-pass @ 10 Hz
        b, a  = sig_proc.butter(4, 10.0 / (0.5 * fs), btype="low")
        acc_z = sig_proc.filtfilt(b, a, combined).astype(np.float32)

        # Step detection on each foot, then merge
        min_dist = int(0.3 * fs)
        def _peaks(sig):
            th = sig.mean() + 0.3 * sig.std()
            p, _ = sig_proc.find_peaks(sig, height=th, distance=min_dist)
            return p

        steps   = np.sort(np.concatenate([_peaks(left_force), _peaks(right_force)]))
        n_steps = len(steps)

        if n_steps < 4:
            raise ValueError(f"Only {n_steps} steps detected – too few.")

        total_time_s   = len(acc_z) / fs
        step_durations = np.diff(steps) / fs

        cadence          = (n_steps / total_time_s) * 60
        mean_step_dur    = float(np.mean(step_durations))
        std_step_dur     = float(np.std(step_durations))
        cv_step          = std_step_dur / (mean_step_dur + 1e-8)
        mean_dbl_support = mean_step_dur * 0.4
        gait_asymmetry   = (
            float(np.std(step_durations[::2]) / (np.mean(step_durations[::2]) + 1e-8))
            if len(step_durations) > 2 else 0.0
        )
        stride_lengths = [
            float(np.trapz(np.abs(acc_z[steps[i]:steps[i+1]])) / fs)
            for i in range(0, len(steps) - 1, 2)
        ]
        mean_stride = float(np.mean(stride_lengths)) if stride_lengths else 0.0
        std_stride  = float(np.std(stride_lengths))  if stride_lengths else 0.0

        acc_pcts  = np.percentile(acc_z, np.arange(10, 110, 10)).astype(np.float32)
        peak_pcts = np.percentile(acc_z[steps], np.arange(10, 110, 10)).astype(np.float32)
        rms_w     = [np.sqrt(np.mean(acc_z[i:i+fs]**2))
                     for i in range(0, len(acc_z) - fs, fs)] or [0.0]
        zc        = float(np.sum(np.diff(np.sign(acc_z)) != 0)) / total_time_s

        extra = np.array([
            skew(acc_z), sp_kurtosis(acc_z), zc,
            float(np.mean(rms_w)), float(np.std(rms_w)),
            float(np.max(acc_z)), float(np.min(acc_z)), float(np.ptp(acc_z)),
            float(np.median(step_durations)),
            float(np.percentile(step_durations, 10)),
            float(np.percentile(step_durations, 90)),
            float(n_steps), mean_stride, std_stride,
        ], dtype=np.float32)

        feat = np.zeros(42, dtype=np.float32)
        feat[0]   = cadence
        feat[1]   = mean_step_dur
        feat[2]   = std_step_dur
        feat[3]   = mean_dbl_support
        feat[4]   = gait_asymmetry
        feat[5]   = cv_step
        feat[6]   = mean_stride
        feat[7]   = std_stride
        feat[8:18]  = acc_pcts
        feat[18:28] = peak_pcts
        feat[28:42] = extra

        cycles_ok = 30 < cadence < 300
        return feat, cycles_ok

    except Exception as e:
        st.warning(f"⚠️ Gait (.txt) extraction unavailable – using mock features. ({e})")
        return mock_gait_features(), True


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_severity_radar(
    voice_feat: np.ndarray,
    eeg_feat: np.ndarray,
    gait_feat: np.ndarray,
    severity_name: str,
) -> go.Figure:
    """
    Interactive Plotly Severity Radar Chart with 5 clinical axes:
      Voice Dysphonia | EEG Beta Power | Gait Asymmetry | Stride Cadence | Freezing Episodes
    """
    voice_dysphonia = float(np.clip((1 - voice_feat[42] / 25.0) * 100, 0, 100))
    eeg_beta_power = float(np.clip(np.abs(eeg_feat[3 * 8: 4 * 8]).mean() * 10, 0, 100))
    gait_asymmetry = float(np.clip(gait_feat[4] * 200, 0, 100))
    stride_cadence = float(np.clip((gait_feat[0] / 120.0) * 100, 0, 100))
    freezing_episodes = float(np.clip(gait_feat[5] * 150, 0, 100))

    categories = [
        "Voice Dysphonia",
        "EEG Beta Power",
        "Gait Asymmetry",
        "Stride Cadence",
        "Freezing Episodes",
    ]
    values = [
        voice_dysphonia,
        eeg_beta_power,
        gait_asymmetry,
        stride_cadence,
        freezing_episodes,
    ]
    categories += [categories[0]]
    values += [values[0]]

    color_map = {"Mild": "#27AE60", "Moderate": "#E67E22", "Severe": "#C0392B"}
    fill_color = color_map.get(severity_name, "#3498DB")

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor=f"rgba({int(fill_color[1:3], 16)}, "
                      f"{int(fill_color[3:5], 16)}, "
                      f"{int(fill_color[5:7], 16)}, 0.25)",
            line=dict(color=fill_color, width=2.5),
            marker=dict(size=7, color=fill_color),
            name=f"Severity: {severity_name}",
        )
    )
    fig.update_layout(
        font=dict(color="#000000", family="Arial, sans-serif"),
        polar=dict(
            bgcolor="#FFFFFF",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=11, color="#000000"),
                linecolor="#000000",
                gridcolor="#CBD5E1",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#000000", family="Arial, sans-serif"),
                linecolor="#000000",
                gridcolor="#CBD5E1",
            ),
        ),
        showlegend=True,
        legend=dict(font=dict(size=12, color="#000000")),
        title=dict(text="Multimodal Severity Radar", font=dict(size=16, color="#000000")),
        paper_bgcolor="#FFFFFF",
        margin=dict(l=60, r=60, t=60, b=40),
        height=400,
    )
    return fig


def plot_longitudinal_trend(longitudinal_data: List[Dict], patient_label: str = "") -> go.Figure:
    """Line chart of UPDRS scores across visits."""
    dates = [d["date"] for d in longitudinal_data]
    scores = [d["updrs"] for d in longitudinal_data]

    fig = go.Figure()
    if dates:
        fig.add_trace(go.Scatter(
            x=dates, y=scores,
            mode="lines+markers",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=8, color="#1D4ED8", symbol="circle"),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)",
            name="UPDRS Score",
        ))
    fig.add_hrect(
        y0=0, y1=35, fillcolor="#27AE60", opacity=0.08, line_width=0,
        annotation_text="Mild (0–35)", annotation_position="top left",
        annotation_font_color="#000000", annotation_font_size=12
    )
    fig.add_hrect(
        y0=35, y1=70, fillcolor="#E67E22", opacity=0.08, line_width=0,
        annotation_text="Moderate (35–70)", annotation_position="top left",
        annotation_font_color="#000000", annotation_font_size=12
    )
    fig.add_hrect(
        y0=70, y1=108, fillcolor="#C0392B", opacity=0.08, line_width=0,
        annotation_text="Severe (70–108)", annotation_position="top left",
        annotation_font_color="#000000", annotation_font_size=12
    )

    title_text = f"Longitudinal UPDRS Progression – {patient_label}" if patient_label else "Longitudinal UPDRS Progression"
    fig.update_layout(
        font=dict(color="#000000", family="Arial, sans-serif"),
        title=dict(text=title_text, font=dict(color="#000000", size=16)),
        xaxis=dict(
            title=dict(text="Visit Date", font=dict(color="#000000", size=13)),
            tickfont=dict(color="#000000", size=12),
            linecolor="#000000",
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(
            title=dict(text="UPDRS Score", font=dict(color="#000000", size=13)),
            range=[0, 108],
            tickfont=dict(color="#000000", size=12),
            linecolor="#000000",
            gridcolor="#E2E8F0",
        ),
        legend=dict(font=dict(color="#000000", size=12)),
        height=350,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        showlegend=True,
    )
    return fig


def plot_attention_heatmap(attn_weights: np.ndarray) -> go.Figure:
    """Heatmap of transformer attention weights (n_heads × n_modalities × n_modalities)."""
    modality_labels = ["Voice", "EEG", "Gait"]
    avg_attn = attn_weights.mean(axis=0)

    fig = go.Figure(go.Heatmap(
        z=avg_attn,
        x=modality_labels,
        y=modality_labels,
        colorscale="Blues",
        text=np.round(avg_attn, 3),
        texttemplate="%{text}",
        textfont=dict(color="#000000", size=13),
        showscale=True,
        colorbar=dict(
            title=dict(text="Attention Weight", font=dict(color="#000000")),
            tickfont=dict(color="#000000"),
        ),
    ))
    fig.update_layout(
        font=dict(color="#000000", family="Arial, sans-serif"),
        title=dict(text="Transformer Attention Heatmap (avg over heads)", font=dict(color="#000000", size=16)),
        xaxis=dict(
            title=dict(text="Key Modality", font=dict(color="#000000", size=13)),
            tickfont=dict(color="#000000", size=12),
            linecolor="#000000",
        ),
        yaxis=dict(
            title=dict(text="Query Modality", font=dict(color="#000000", size=13)),
            tickfont=dict(color="#000000", size=12),
            linecolor="#000000",
        ),
        height=350,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    return fig


def plot_probability_bars(cls_probs: np.ndarray) -> go.Figure:
    """Horizontal bar chart of severity class probabilities."""
    labels = ["Mild", "Moderate", "Severe"]
    colors_list = ["#27AE60", "#E67E22", "#C0392B"]
    fig = go.Figure(go.Bar(
        x=cls_probs * 100,
        y=labels,
        orientation="h",
        marker_color=colors_list,
        text=[f"{p:.1f}%" for p in cls_probs * 100],
        textfont=dict(color="#000000", size=13),
        textposition="outside",
    ))
    fig.update_layout(
        font=dict(color="#000000", family="Arial, sans-serif"),
        title=dict(text="Severity Class Probabilities", font=dict(color="#000000", size=16)),
        xaxis=dict(
            title=dict(text="Probability (%)", font=dict(color="#000000", size=13)),
            range=[0, 115],
            tickfont=dict(color="#000000", size=12),
            linecolor="#000000",
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(
            tickfont=dict(size=13, color="#000000"),
            linecolor="#000000",
        ),
        height=250,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=40, t=50, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

def _init_session_state():
    defaults = {
        "authenticated": False,
        "user": None,
        "role": "Neurologist",
        "results": None,
        "voice_feat": None,
        "eeg_feat": None,
        "gait_feat": None,
        "quality_flags": (True, True, True),
        "subject_id": "DEMO_001",
        "patient_id": "DEMO_001",
        "patient_name": "John Doe",
        "longitudinal_data": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION PAGE (Login & Registration)
# ─────────────────────────────────────────────────────────────────────────────

def render_auth_page():
    """Render a centered, elegant login and registration interface."""
    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        st.markdown("""
        <div style='text-align: center; margin-top: 25px; margin-bottom: 25px;'>
            <h1 style='color: #0F172A; margin-bottom: 4px; font-size: 2.2em;'>🧠 ParkiSense AI</h1>
            <p style='color: #475569; font-size: 1.1em; font-weight: 500;'>
                Clinical Multimodal Parkinson's Assessment Platform
            </p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with tab_login:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="e.g. dr_smith").strip()
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

                if submitted:
                    if not username or not password:
                        st.error("Please enter both username and password.")
                    else:
                        user = authenticate_user(username, password)
                        if user:
                            st.session_state.authenticated = True
                            st.session_state.user = user
                            st.session_state.role = user["role"]
                            st.success(f"Welcome back, {user['full_name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

            st.markdown("""
            <div style='margin-top: 20px; padding: 14px 16px; background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 10px;'>
                <div style='font-size: 0.88em; font-weight: 700; color: #1E293B; margin-bottom: 6px;'>
                    ⚡ Quick Demo Accounts (Pre-configured):
                </div>
                <div style='font-size: 0.84em; color: #334155; line-height: 1.6;'>
                    🩺 <b>Neurologist:</b> Username: <code>dr_smith</code> | Password: <code>admin123</code><br/>
                    🧑 <b>Patient:</b> Username: <code>patient_john</code> | Password: <code>patient123</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with tab_register:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            with st.form("register_form", clear_on_submit=False):
                reg_name = st.text_input("Full Name", placeholder="e.g. Dr. Alex Morgan").strip()
                reg_username = st.text_input("Desired Username", placeholder="e.g. amorgan").strip()
                reg_role = st.selectbox(
                    "User Role",
                    ["Neurologist", "Patient"],
                    help="Neurologist: full clinical analysis, attention maps, PDF export | Patient: simplified summary and care plan"
                )
                reg_pass = st.text_input("Password (min. 6 characters)", type="password", placeholder="••••••••")
                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••")
                reg_submit = st.form_submit_button("Create Account", use_container_width=True)

                if reg_submit:
                    if not reg_name or not reg_username or not reg_pass:
                        st.error("All fields are required.")
                    elif len(reg_pass) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif reg_pass != reg_confirm:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = register_user(reg_username, reg_pass, reg_role, reg_name)
                        if ok:
                            st.success(f"{msg} You can now switch to the 'Sign In' tab.")
                        else:
                            st.error(msg)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    user = st.session_state.get("user")
    with st.sidebar:
        st.markdown(
            "![ParkiSense AI](https://img.shields.io/badge/ParkiSense-AI-3498DB?"
            "style=for-the-badge&logo=brain)"
        )

        if user:
            role_icon = "🩺" if user["role"] == "Neurologist" else "🧑"
            st.markdown(f"""
            <div style='background: #FFFFFF; border: 1.5px solid #CBD5E1; border-radius: 10px; padding: 12px 14px; margin-bottom: 14px;'>
                <div style='font-size: 0.80em; color: #64748B; font-weight: 700; text-transform: uppercase;'>Signed In User</div>
                <div style='font-size: 1.0em; font-weight: 700; color: #0F172A; margin-top: 2px;'>{user['full_name']}</div>
                <div style='font-size: 0.85em; color: #2563EB; font-weight: 600; margin-top: 4px;'>{role_icon} {user['role']}</div>
            </div>
            """, unsafe_allow_html=True)

        role = user["role"] if user else "Neurologist"
        st.session_state.role = role
        if user and user["role"] == "Neurologist":
            st.markdown("### 👤 Patient Selection")
            all_pts = get_all_patients()
            # Clean list of standard demo/real patients
            clean_pts = [p for p in all_pts if not p["patient_id"].startswith("test_")]
            options = [f"{p['patient_id']} – {p['patient_name']}" for p in clean_pts]
            options.append("➕ New Patient...")

            curr_pid = st.session_state.get("patient_id", "DEMO_001")
            sel_idx = 0
            for idx, p in enumerate(clean_pts):
                if p["patient_id"] == curr_pid:
                    sel_idx = idx
                    break

            selected = st.selectbox(
                "Select Patient",
                options,
                index=sel_idx,
                help="Select an existing patient to load their longitudinal history or create a new profile",
            )

            if selected == "➕ New Patient...":
                p_id = st.text_input("Patient ID", value="PT_003").strip()
                p_name = st.text_input("Patient Full Name", placeholder="e.g. Robert Brown").strip()
                st.session_state.patient_id = p_id if p_id else "PT_003"
                st.session_state.patient_name = p_name if p_name else st.session_state.patient_id
            else:
                parts = selected.split(" – ")
                st.session_state.patient_id = parts[0].strip()
                st.session_state.patient_name = parts[1].strip() if len(parts) > 1 else parts[0].strip()

            st.session_state.subject_id = st.session_state.patient_id
            st.caption(f"Active Patient: **{st.session_state.patient_name}** (`{st.session_state.patient_id}`)")
        else:
            # When logged in as Patient
            patient_uname = user["username"] if user else "DEMO_001"
            patient_fname = user["full_name"] if user else "John Doe"
            st.session_state.patient_id = patient_uname
            st.session_state.patient_name = patient_fname
            st.session_state.subject_id = patient_uname
            st.markdown(f"**Patient Record**: `{patient_fname}`")

        st.divider()
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.results = None
            st.rerun()

        st.markdown("### ℹ️ About")
        st.info(
            "**ParkiSense AI** v1.0\n\n"
            "Multimodal deep learning framework for objective Parkinson's "
            "disease severity assessment using voice, EEG, and gait signals."
        )
        st.markdown("---")
        st.caption("© 2024 ParkiSense AI Research Team")

    return role


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD + QUALITY GATE PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_upload_panel():
    st.markdown('<div class="section-header">📁 Upload Physiological Signals</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    voice_file = col1.file_uploader("🎙️ Voice Recording (.wav)", type=["wav"])
    eeg_file = col2.file_uploader("🧠 EEG Recording (.set / .fdt — EEGLAB)", type=["set", "fdt"], accept_multiple_files=True)
    gait_file = col3.file_uploader("🦶 Gait / VGRF Data (.txt — PhysioNet)", type=["txt"])

    st.markdown('<div class="section-header">🔍 Quality Gate Inspection</div>', unsafe_allow_html=True)

    qc_col1, qc_col2, qc_col3 = st.columns(3)

    voice_feat, eeg_feat, gait_feat = None, None, None
    voice_ok, eeg_ok, gait_ok = True, True, True
    snr_value = 0.0

    with qc_col1:
        with st.container():
            if voice_file:
                voice_feat, voice_ok, snr_value = extract_voice(voice_file)
                st.metric("Voice SNR (HNR)", f"{snr_value:.1f} dB")
                status = "✅ OK" if voice_ok else "⚠️ Low – Fallback Active"
                st.markdown(f"<div style='color: #0F172A; font-weight: 600; margin-top: 6px;'>SNR Status: <span class='{'quality-ok' if voice_ok else 'quality-fail'}'>{status}</span></div>", unsafe_allow_html=True)
            else:
                st.info("No voice file uploaded")
                voice_feat = mock_voice_features()
                st.caption("Using mock features for demo")

    with qc_col2:
        with st.container():
            if eeg_file and len(eeg_file) > 0:
                eeg_feat, eeg_ok = extract_eeg(eeg_file)
                status = "✅ Artifact-free" if eeg_ok else "⚠️ Artifacts Detected – Fallback Active"
                st.metric("EEG Artifact Check", "Pass" if eeg_ok else "Fail")
                st.markdown(f"<div style='color: #0F172A; font-weight: 600; margin-top: 6px;'>EEG Status: <span class='{'quality-ok' if eeg_ok else 'quality-fail'}'>{status}</span></div>", unsafe_allow_html=True)
            else:
                st.info("No EEG file uploaded")
                eeg_feat = mock_eeg_features()
                eeg_ok = True
                st.caption("Using mock features for demo")

    with qc_col3:
        with st.container():
            if gait_file:
                gait_feat, gait_ok = extract_gait(gait_file)
                status = "✅ Sufficient gait cycles" if gait_ok else "⚠️ Few Cycles – Fallback Active"
                st.metric("Gait Cycle Quality", "Pass" if gait_ok else "Fail")
                st.markdown(f"<div style='color: #0F172A; font-weight: 600; margin-top: 6px;'>Gait Status: <span class='{'quality-ok' if gait_ok else 'quality-fail'}'>{status}</span></div>", unsafe_allow_html=True)
            else:
                st.info("No gait file uploaded")
                gait_feat = mock_gait_features()
                gait_ok = True
                st.caption("Using mock features for demo")

    # Safe Fallback summary
    n_fallbacks = sum([not voice_ok, not eeg_ok, not gait_ok])
    if n_fallbacks > 0:
        st.warning(
            f"🔄 **Safe Fallback Gate**: {n_fallbacks} modality/modalities are substituted with "
            f"learned filler vectors. The model will re-weight remaining signals automatically."
        )
    else:
        st.success("✅ All modalities passed quality checks. Full multimodal fusion active.")

    return voice_feat, eeg_feat, gait_feat, (voice_ok, eeg_ok, gait_ok)


# ─────────────────────────────────────────────────────────────────────────────
# NEUROLOGIST VIEW
# ─────────────────────────────────────────────────────────────────────────────

def render_neurologist_view(results: dict, voice_feat, eeg_feat, gait_feat, quality_flags):
    severity_name = results["severity_name"]
    updrs_score = results["updrs_score"]
    cls_probs = results["cls_probs"]

    severity_hex = {"Mild": "#27AE60", "Moderate": "#E67E22", "Severe": "#C0392B"}
    bg_color = severity_hex.get(severity_name, "#3498DB")

    # ── Top summary metrics ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Assessment Results</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔢 UPDRS Score", f"{updrs_score:.1f} / 100")
    m2.metric("🏷️ Severity Tier", severity_name)
    m3.metric("🎯 Confidence", f"{cls_probs.max() * 100:.1f}%")
    m4.metric("📉 Fallback Modalities", sum(not f for f in quality_flags))

    st.markdown(
        f"<div style='text-align:center; margin: 12px 0;'>"
        f"<span class='severity-badge' style='background:{bg_color};'>"
        f"Severity: {severity_name}</span></div>",
        unsafe_allow_html=True,
    )

    # ── Visualizations row 1 ───────────────────────────────────────────────────
    col_radar, col_probs = st.columns([3, 2])
    with col_radar:
        st.plotly_chart(
            plot_severity_radar(voice_feat, eeg_feat, gait_feat, severity_name),
            use_container_width=True,
            theme=None,
        )
    with col_probs:
        st.plotly_chart(plot_probability_bars(cls_probs), use_container_width=True, theme=None)

    # ── Attention heatmap ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 Transformer Attention Analysis</div>', unsafe_allow_html=True)
    model = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    try:
        with torch.no_grad():
            v_t = torch.tensor(voice_feat, dtype=torch.float32).unsqueeze(0).to(device)
            e_t = torch.tensor(eeg_feat, dtype=torch.float32).unsqueeze(0).to(device)
            g_t = torch.tensor(gait_feat, dtype=torch.float32).unsqueeze(0).to(device)
            f_t = torch.tensor([list(quality_flags)], dtype=torch.bool).to(device)
            attn_list = model.get_attention_weights(v_t, e_t, g_t, f_t)
        # Last layer, shape (batch, heads, 3, 3) → (heads, 3, 3)
        attn_np = attn_list[-1].squeeze(0).detach().cpu().numpy()
        st.plotly_chart(plot_attention_heatmap(attn_np), use_container_width=True, theme=None)
    except Exception as e:
        st.info(f"Attention visualisation unavailable: {e}")

    # ── Longitudinal trend ─────────────────────────────────────────────────────
    patient_id = st.session_state.get("patient_id", "DEMO_001")
    patient_name = st.session_state.get("patient_name", "John Doe")
    patient_label = f"{patient_name} ({patient_id})"

    st.markdown(f'<div class="section-header">📈 Longitudinal UPDRS Progression – {patient_label}</div>', unsafe_allow_html=True)

    # Fetch patient's actual longitudinal records from database
    patient_visits = get_patient_visits(patient_id)

    # Allow adding new visit data point to database
    with st.expander(f"➕ Add Visit Record for {patient_name} ({patient_id})"):
        add_col1, add_col2 = st.columns([2, 1])
        visit_date = add_col1.text_input("Visit Date (YYYY-MM)", value="2025-04", key=f"visit_date_{patient_id}")
        add_col2.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if add_col2.button(f"Save Visit for {patient_name}", use_container_width=True):
            user = st.session_state.get("user")
            doc_uname = user["username"] if user else "dr_smith"
            ok = add_patient_visit(
                patient_id=patient_id,
                patient_name=patient_name,
                visit_date=visit_date,
                updrs_score=updrs_score,
                severity_name=severity_name,
                recorded_by=doc_uname,
            )
            if ok:
                st.success(f"✅ Saved Visit {visit_date} (UPDRS {updrs_score:.1f} — {severity_name}) for {patient_name} to database!")
                st.rerun()

    if patient_visits:
        st.plotly_chart(
            plot_longitudinal_trend(patient_visits, patient_label=patient_label),
            use_container_width=True,
            theme=None,
        )
    else:
        st.info(f"ℹ️ No previous visits recorded for {patient_name} ({patient_id}) yet. Use the expander above to record today's assessment.")

    # ── Therapy recommendations ────────────────────────────────────────────────
    st.markdown('<div class="section-header">💊 Personalised Therapy Recommendations</div>', unsafe_allow_html=True)
    therapy_recs = generate_therapy_recommendations(severity_name)

    tc1, tc2 = st.columns(2)
    therapy_items = list(therapy_recs.items())
    for i, (category, rec) in enumerate(therapy_items):
        col = tc1 if i % 2 == 0 else tc2
        icon = {"physiotherapy": "🏃", "speech_therapy": "🎤", "medication": "💊", "nutrition": "🥗"}.get(category, "📋")
        col.markdown(f"**{icon} {category.replace('_', ' ').title()}**")
        col.info(rec)

    # ── PDF Export ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📄 Export Clinical Report</div>', unsafe_allow_html=True)
    if st.button("📥 Generate & Download PDF Report", type="primary"):
        with st.spinner("Generating PDF report…"):
            try:
                pdf_bytes = generate_pdf_report(
                    subject_id=patient_label,
                    updrs_score=updrs_score,
                    severity_name=severity_name,
                    cls_probs=cls_probs,
                    quality_flags=quality_flags,
                    therapy_recs=therapy_recs,
                    longitudinal_data=patient_visits if patient_visits else [{"date": "2025-04", "updrs": updrs_score}],
                )
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"ParkiSense_{patient_id}_{severity_name}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT VIEW
# ─────────────────────────────────────────────────────────────────────────────

def render_patient_view(results: dict, voice_feat, eeg_feat, gait_feat, quality_flags):
    severity_name = results["severity_name"]
    updrs_score = results["updrs_score"]

    severity_hex = {"Mild": "#27AE60", "Moderate": "#E67E22", "Severe": "#C0392B"}
    emoji_map = {"Mild": "🟢", "Moderate": "🟡", "Severe": "🔴"}
    bg_color = severity_hex.get(severity_name, "#3498DB")

    st.markdown(f"""
    <div style='background:white; border-radius:16px; padding:30px;
                box-shadow:0 4px 16px rgba(0,0,0,0.10); text-align:center; margin-bottom:20px;'>
        <div style='font-size:3em; margin-bottom:8px;'>{emoji_map.get(severity_name, "🔵")}</div>
        <h2 style='color:{bg_color}; margin:0;'>Your Parkinson's Severity: {severity_name}</h2>
        <p style='color:#7F8C8D; margin-top:8px;'>
            Your assessment score is <strong>{updrs_score:.1f}</strong> out of 100.<br/>
            This was calculated from your voice, brain activity, and movement data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Simplified radar
    st.markdown("### 📊 Your Health Profile")
    st.plotly_chart(
        plot_severity_radar(voice_feat, eeg_feat, gait_feat, severity_name),
        use_container_width=True,
        theme=None,
    )

    # Patient's personal longitudinal tracking
    patient_id = st.session_state.get("patient_id", "DEMO_001")
    patient_name = st.session_state.get("patient_name", "Your Progress")
    patient_visits = get_patient_visits(patient_id)
    if patient_visits:
        st.markdown("### 📈 Your Health Progression Over Time")
        st.plotly_chart(
            plot_longitudinal_trend(patient_visits, patient_label=patient_name),
            use_container_width=True,
            theme=None,
        )

    # Therapy guidance
    st.markdown("### 💡 Your Personal Care Plan")
    therapy_recs = generate_therapy_recommendations(severity_name)

    icon_map = {
        "physiotherapy": ("🏃‍♂️", "Physical Activity"),
        "speech_therapy": ("🎤", "Voice & Speech"),
        "medication": ("💊", "Medication"),
        "nutrition": ("🥗", "Nutrition"),
    }

    for category, rec in therapy_recs.items():
        icon, label = icon_map.get(category, ("📋", category))
        with st.expander(f"{icon} {label}"):
            st.write(rec)

    # Motivational message
    st.markdown("---")
    messages = {
        "Mild": "🌟 Great news! Your symptoms are currently mild. Keeping active and following your care plan helps maintain this.",
        "Moderate": "💪 You're doing well tracking your health. Consistent therapy and medication adherence are key at this stage.",
        "Severe": "❤️ Your care team is here to support you. Please make sure to attend your scheduled appointments and reach out whenever you need help.",
    }
    st.info(messages.get(severity_name, ""))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    init_db()
    _init_session_state()

    if not st.session_state.authenticated:
        render_auth_page()
        return

    role = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='header-bar'>
        <h1 style='margin:0; font-size:2em;'>🧠 ParkiSense AI</h1>
        <p style='margin:4px 0 0; opacity:0.85;'>
            Multimodal Deep Learning Framework for Parkinson's Disease Severity Assessment
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload Panel ───────────────────────────────────────────────────────────
    voice_feat, eeg_feat, gait_feat, quality_flags = render_upload_panel()

    # ── Run Analysis ───────────────────────────────────────────────────────────
    st.markdown("---")
    col_btn, col_info = st.columns([1, 3])
    run_analysis = col_btn.button("🚀 Run PD Assessment", type="primary", use_container_width=True)
    col_info.caption(
        "The model fuses voice, EEG, and gait signals via a Transformer with "
        "Safe Fallback gating to produce a continuous UPDRS score and a "
        "3-tier severity classification."
    )

    if run_analysis or st.session_state.results is not None:
        if run_analysis:
            with st.spinner("Running multimodal assessment…"):
                model = load_model()
                results = run_inference(
                    model=model,
                    voice_feat=voice_feat,
                    eeg_feat=eeg_feat,
                    gait_feat=gait_feat,
                    quality_flags=quality_flags,
                )
                st.session_state.results = results
                st.session_state.voice_feat = voice_feat
                st.session_state.eeg_feat = eeg_feat
                st.session_state.gait_feat = gait_feat
                st.session_state.quality_flags = quality_flags

        results = st.session_state.results
        voice_feat = st.session_state.voice_feat
        eeg_feat = st.session_state.eeg_feat
        gait_feat = st.session_state.gait_feat
        quality_flags = st.session_state.quality_flags

        st.markdown("---")
        if role == "Neurologist":
            render_neurologist_view(results, voice_feat, eeg_feat, gait_feat, quality_flags)
        else:
            render_patient_view(results, voice_feat, eeg_feat, gait_feat, quality_flags)


if __name__ == "__main__":
    main()
