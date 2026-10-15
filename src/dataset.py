"""
ParkiSense AI - Dataset Loader & Augmentation
=============================================
Implements `ParkiSenseDataset` (torch.utils.data.Dataset) that reads a
manifest CSV, loads pre-extracted feature vectors, applies optional
data augmentation, and returns modality-quality-gated samples.

Manifest CSV columns expected
------------------------------
subject_id         : unique identifier (str)
voice_feat_path    : path to .npy file with voice features (52-dim)
eeg_feat_path      : path to .npy file with EEG features
gait_feat_path     : path to .npy file with gait features (42-dim)
voice_snr_ok       : bool – quality flag for voice
eeg_artifact_ok    : bool – quality flag for EEG
gait_cycles_ok     : bool – quality flag for gait
updrs_score        : float in [0, 108] – ground-truth UPDRS total
label              : int {0: Mild, 1: Moderate, 2: Severe}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Label Mapping
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP: Dict[int, str] = {0: "Mild", 1: "Moderate", 2: "Severe"}


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _augment_voice(feat: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """
    Apply Gaussian noise injection and a pitch-shift proxy (spectral warping
    of MFCC coefficients) to a 52-dim voice feature vector.
    """
    feat = feat.copy()
    # Additive Gaussian noise
    feat += np.random.randn(*feat.shape).astype(np.float32) * noise_std
    # Pitch-shift proxy: scale F0 value by ±2 %
    pitch_scale = np.random.uniform(0.98, 1.02)
    feat[39] *= pitch_scale  # F0 index
    return feat


def _augment_eeg(feat: np.ndarray, dropout_prob: float = 0.1) -> np.ndarray:
    """
    Random channel dropout: zero-out features from a random fraction of EEG
    channels (applied uniformly across the feature vector).
    """
    feat = feat.copy()
    mask = np.random.binomial(1, 1 - dropout_prob, size=feat.shape).astype(np.float32)
    feat *= mask
    return feat


def _augment_gait(feat: np.ndarray, speed_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
    """
    Apply speed variation to gait features: scale cadence and step-time
    features by a random factor in [0.95, 1.05].
    """
    feat = feat.copy()
    speed_factor = np.random.uniform(*speed_range)
    # Cadence scales linearly with speed; step duration scales inversely
    feat[0] *= speed_factor            # cadence
    feat[1] /= speed_factor            # mean step duration
    feat[2] /= speed_factor            # std step duration
    feat[3] /= speed_factor            # double support time
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ParkiSenseDataset(Dataset):
    """
    PyTorch Dataset for the ParkiSense AI multimodal PD assessment framework.

    Parameters
    ----------
    manifest_path : str | Path
        Path to the manifest CSV file (see module docstring for schema).
    root_dir : str | Path, optional
        Root directory prepended to relative feature file paths in the CSV.
        Defaults to the directory containing the manifest.
    augment : bool
        Whether to apply data augmentation. Intended for training splits.
    updrs_scale : float
        Divisor to normalise UPDRS scores to [0, 1] for regression.
        Default 108 (maximum possible UPDRS total score).

    Returns per __getitem__
    ------------------------
    dict with keys:
      "voice"           : FloatTensor  (52,)
      "eeg"             : FloatTensor  (n_eeg_feat,)
      "gait"            : FloatTensor  (42,)
      "quality_flags"   : BoolTensor   (3,)  [voice_ok, eeg_ok, gait_ok]
      "updrs"           : FloatTensor  scalar – normalised UPDRS ∈ [0, 1]
      "label"           : LongTensor   scalar – {0, 1, 2}
      "subject_id"      : str
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: Optional[str | Path] = None,
        augment: bool = False,
        updrs_scale: float = 108.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir else self.manifest_path.parent
        self.augment = augment
        self.updrs_scale = updrs_scale

        self.df = pd.read_csv(self.manifest_path)
        self._validate_columns()
        logger.info(
            "ParkiSenseDataset loaded: %d samples, augment=%s",
            len(self.df), augment
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_columns(self) -> None:
        required = {
            "subject_id", "voice_feat_path", "eeg_feat_path", "gait_feat_path",
            "voice_snr_ok", "eeg_artifact_ok", "gait_cycles_ok",
            "updrs_score", "label",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest CSV is missing columns: {missing}")

    def _load_npy(self, rel_path: object) -> np.ndarray | None:
        """Load a .npy feature file, returning None if NaN, empty or missing."""
        if rel_path is None or pd.isna(rel_path):
            return None
        rel_str = str(rel_path).strip()
        if not rel_str or rel_str.lower() == "nan":
            return None
        p = self.root_dir / rel_str
        if not p.exists():
            return None
        try:
            return np.load(str(p)).astype(np.float32)
        except Exception:
            return None

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Load feature arrays ───────────────────────────────────────────────
        voice_feat = self._load_npy(row.get("voice_feat_path"))
        eeg_feat = self._load_npy(row.get("eeg_feat_path"))
        gait_feat = self._load_npy(row.get("gait_feat_path"))

        # Fall back to zeros and ensure strict shape matching
        if voice_feat is None:
            voice_feat = np.zeros(52, dtype=np.float32)
        elif len(voice_feat) != 52:
            voice_feat = voice_feat[:52] if len(voice_feat) > 52 else np.pad(voice_feat, (0, 52 - len(voice_feat)))

        if eeg_feat is None:
            eeg_feat = np.zeros(64, dtype=np.float32)
        elif len(eeg_feat) != 64:
            eeg_feat = eeg_feat[:64] if len(eeg_feat) > 64 else np.pad(eeg_feat, (0, 64 - len(eeg_feat)))

        if gait_feat is None:
            gait_feat = np.zeros(42, dtype=np.float32)
        elif len(gait_feat) != 42:
            gait_feat = gait_feat[:42] if len(gait_feat) > 42 else np.pad(gait_feat, (0, 42 - len(gait_feat)))

        # ── Quality flags ─────────────────────────────────────────────────────
        def _parse_bool(val) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            return str(val).strip().lower() in ("true", "1", "yes")

        voice_ok = _parse_bool(row["voice_snr_ok"])
        eeg_ok = _parse_bool(row["eeg_artifact_ok"])
        gait_ok = _parse_bool(row["gait_cycles_ok"])

        # ── Data augmentation (training only) ─────────────────────────────────
        if self.augment:
            if voice_ok:
                voice_feat = _augment_voice(voice_feat)
            if eeg_ok:
                eeg_feat = _augment_eeg(eeg_feat)
            if gait_ok:
                gait_feat = _augment_gait(gait_feat)

        # ── Normalise UPDRS to [0, 1] ─────────────────────────────────────────
        updrs_norm = float(row["updrs_score"]) / self.updrs_scale

        return {
            "voice": torch.tensor(voice_feat, dtype=torch.float32),
            "eeg": torch.tensor(eeg_feat, dtype=torch.float32),
            "gait": torch.tensor(gait_feat, dtype=torch.float32),
            "quality_flags": torch.tensor(
                [voice_ok, eeg_ok, gait_ok], dtype=torch.bool
            ),
            "updrs": torch.tensor(updrs_norm, dtype=torch.float32),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "subject_id": str(row["subject_id"]),
        }

    def get_label_name(self, label: int) -> str:
        """Return human-readable severity tier name for a numeric label."""
        return LABEL_MAP.get(label, "Unknown")
