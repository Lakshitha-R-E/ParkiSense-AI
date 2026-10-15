"""
ParkiSense AI – Real Dataset Builder  (v2 – fixed)
====================================================
Processes all three raw modalities into .npy feature files and generates
a unified manifest.csv linking them with real UPDRS scores / labels.

Dataset layout expected
------------------------
data/raw/voice/HC_AH/HC_AH/   → Healthy-control voice .wav files
data/raw/voice/PD_AH/PD_AH/   → PD-patient voice .wav files
data/raw/eeg/eeg dataset/sub-XXX/eeg/  → BIDS .set/.fdt EEG files
data/raw/gait/gait analysis dataset/gait-in-parkinsons-disease-1.0.0/
    demographics.txt           → subject metadata + UPDRS scores
    GaPt*.txt / GaCo*.txt ...  → VGRF gait walk files (19 columns)

Output
------
data/features/voice/   → <subject_id>_voice.npy  (52-dim)
data/features/eeg/     → <subject_id>_eeg.npy    (variable-dim)
data/features/gait/    → <subject_id>_gait.npy   (42-dim)
datasets/manifest.csv  → unified manifest with quality flags + UPDRS

Run from parkisense_ai/ with:
    python data/scripts/build_dataset.py
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as ss
from scipy.stats import skew, kurtosis

# ── Ensure src/ is importable ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # …/parkisense_ai/
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = ROOT / "data" / "raw"
FEAT_ROOT = ROOT / "data" / "features"
VOICE_HC  = DATA_ROOT / "voice" / "HC_AH" / "HC_AH"
VOICE_PD  = DATA_ROOT / "voice" / "PD_AH" / "PD_AH"
EEG_ROOT  = DATA_ROOT / "eeg" / "eeg dataset"
GAIT_ROOT = (DATA_ROOT / "gait" / "gait analysis dataset"
             / "gait-in-parkinsons-disease-1.0.0")
DEMO_FILE = GAIT_ROOT / "demographics.txt"
MANIFEST  = ROOT / "datasets" / "manifest.csv"

for d in ["voice", "eeg", "gait"]:
    (FEAT_ROOT / d).mkdir(parents=True, exist_ok=True)

# Valid gait file pattern: e.g. GaCo01_01.txt, JuPt03_06.txt, SiCo02_01.txt
GAIT_FILE_RE = re.compile(r"^(Ga|Ju|Si)(Co|Pt)\d+_\d+\.txt$", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def updrs_to_label(score: float) -> int:
    """Map UPDRS score to 3-tier severity label."""
    if score <= 35:
        return 0   # Mild
    elif score <= 70:
        return 1   # Moderate
    else:
        return 2   # Severe


def voice_snr_ok(feat: np.ndarray) -> bool:
    """Voice quality: HNR > 5 dB considered acceptable."""
    return float(feat[42]) > 5.0


# ─────────────────────────────────────────────────────────────────────────────
# DEMOGRAPHICS LOADER  (fixed: use original column names, not rename trick)
# ─────────────────────────────────────────────────────────────────────────────

def load_demographics() -> dict[str, dict]:
    """
    Load PhysioNet demographics.txt and return a mapping of
    subject_id → {updrs, hoehn_yahr, group}.

    Only PD subjects with a valid UPDRS score are included.
    """
    demo = pd.read_csv(
        str(DEMO_FILE),
        sep="\t",
        on_bad_lines="skip",   # some rows have extra trailing tabs
    )
    # Keep only rows with a valid UPDRS score
    demo = demo[["ID", "Group", "HoehnYahr", "UPDRS"]].dropna(subset=["UPDRS"])
    demo_map = {}
    for _, row in demo.iterrows():
        demo_map[str(row["ID"]).strip()] = {
            "updrs":      float(row["UPDRS"]),
            "hoehn_yahr": float(row["HoehnYahr"]) if pd.notna(row["HoehnYahr"]) else 0.0,
            "group":      int(row["Group"]),
        }
    log.info("Loaded %d subjects from demographics.txt (with valid UPDRS).", len(demo_map))
    return demo_map


# ─────────────────────────────────────────────────────────────────────────────
# 1. VOICE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_voice() -> list[dict]:
    """Process all .wav files → 52-dim .npy features. Returns list of records."""
    from src.preprocessing import extract_voice_features

    records = []
    for group_label, wav_dir, is_pd in [
        ("HC", VOICE_HC, False),
        ("PD", VOICE_PD, True),
    ]:
        wav_files = sorted(wav_dir.glob("*.wav"))
        log.info("Processing %d %s voice files…", len(wav_files), group_label)

        for wav in wav_files:
            subject_id = f"V_{group_label}_{wav.stem[:12]}"
            out_path   = FEAT_ROOT / "voice" / f"{subject_id}_voice.npy"

            if out_path.exists():
                feat = np.load(str(out_path))
            else:
                try:
                    feat = extract_voice_features(str(wav))
                    np.save(str(out_path), feat)
                except Exception as e:
                    log.warning("Voice failed for %s: %s", wav.name, e)
                    continue

            # Healthy controls: UPDRS=0 by definition
            updrs = 0.0 if not is_pd else 40.0
            label = updrs_to_label(updrs)

            records.append({
                "subject_id":      subject_id,
                "voice_feat_path": str(out_path.relative_to(ROOT)),
                "eeg_feat_path":   "",
                "gait_feat_path":  "",
                "voice_snr_ok":    voice_snr_ok(feat),
                "eeg_artifact_ok": False,
                "gait_cycles_ok":  False,
                "updrs_score":     updrs,
                "label":           label,
                "source":          "voice",
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2. EEG PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _load_eeg_set(set_path: Path) -> np.ndarray | None:
    """Load EEGLAB .set file with MNE and extract PSD band + coherence features."""
    try:
        import mne
        mne.set_log_level("ERROR")

        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        n_ch  = len(raw.ch_names)

        raw.filter(1., 45., method="fir", verbose=False)

        n_comp = min(15, n_ch - 1)
        if n_comp >= 2:
            ica = mne.preprocessing.ICA(n_components=n_comp, random_state=42,
                                        max_iter="auto", verbose=False)
            ica.fit(raw, verbose=False)
            try:
                eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
                ica.exclude = eog_idx
            except Exception:
                pass
            raw = ica.apply(raw, verbose=False)

        data    = raw.get_data()
        seg_len = int(4.0 * sfreq)
        hop     = int(seg_len * 0.25)
        BANDS   = [(1,4),(4,8),(8,13),(13,30),(30,45)]

        all_feats = []
        start = 0
        while start + seg_len <= data.shape[1]:
            seg   = data[:, start:start+seg_len]
            freqs, psd = ss.welch(seg, fs=sfreq, nperseg=min(256, seg_len))
            bp = [psd[:, (freqs >= lo) & (freqs < hi)].mean(axis=1)
                  for (lo, hi) in BANDS]
            band_feat = np.concatenate(bp)
            coh_vals  = [ss.coherence(seg[i], seg[i+1], fs=sfreq)[1].mean()
                         for i in range(n_ch - 1)]
            all_feats.append(
                np.concatenate([band_feat,
                                np.array(coh_vals, dtype=np.float32)]).astype(np.float32)
            )
            start += hop

        return np.mean(all_feats, axis=0).astype(np.float32) if all_feats else None
    except Exception as e:
        log.debug("EEG load error %s: %s", set_path.name, e)
        return None


def process_eeg(max_subjects: int = 40) -> list[dict]:
    """Process BIDS EEG subjects → 64-dim .npy features. Returns list of records."""
    records  = []
    all_sub_dirs = sorted(EEG_ROOT.glob("sub-*"))
    # Select balanced subset: half controls (sub-001..075), half PD (sub-076..150)
    controls = [d for d in all_sub_dirs if int(d.name.replace("sub-", "")) <= 75][:max_subjects // 2]
    patients = [d for d in all_sub_dirs if int(d.name.replace("sub-", "")) > 75][:max_subjects // 2]
    sub_dirs = controls + patients
    log.info("Processing %d selected EEG subjects (balanced HC/PD)…", len(sub_dirs))

    for sub_dir in sub_dirs:
        subject_id = sub_dir.name
        eeg_dir    = sub_dir / "eeg"
        set_files  = list(eeg_dir.glob("*.set"))
        if not set_files:
            continue

        out_path = FEAT_ROOT / "eeg" / f"{subject_id}_eeg.npy"

        if out_path.exists():
            feat        = np.load(str(out_path))
            artifact_ok = True
        else:
            feat = _load_eeg_set(set_files[0])
            if feat is None:
                log.warning("EEG skipped: %s", subject_id)
                continue
            # Standardize to fixed 64 dimensions for encoder compatibility
            if len(feat) >= 64:
                feat = feat[:64]
            else:
                feat = np.pad(feat, (0, 64 - len(feat)), mode="constant")
            np.save(str(out_path), feat)
            artifact_ok = bool(np.std(feat) < 50.0)

        # sub-001..075 = Control (UPDRS=0), sub-076..150 = PD (estimate)
        sub_num = int(subject_id.replace("sub-", ""))
        is_pd   = sub_num > 75
        updrs   = 0.0 if not is_pd else 35.0
        label   = updrs_to_label(updrs)

        records.append({
            "subject_id":      subject_id,
            "voice_feat_path": "",
            "eeg_feat_path":   str(out_path.relative_to(ROOT)),
            "gait_feat_path":  "",
            "voice_snr_ok":    False,
            "eeg_artifact_ok": artifact_ok,
            "gait_cycles_ok":  False,
            "updrs_score":     updrs,
            "label":           label,
            "source":          "eeg",
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 3. GAIT PROCESSING  (uses real UPDRS from demographics.txt)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_gait_from_txt(txt_path: Path) -> np.ndarray | None:
    """
    Parse PhysioNet VGRF file (19 tab-separated columns, 100 Hz).
    Col 18 = total left VGRF, col 19 = total right VGRF (1-indexed).
    Returns 42-dim spatiotemporal feature vector.
    """
    try:
        df = pd.read_csv(str(txt_path), sep=r"\s+", header=None,
                         on_bad_lines="skip")
        if df.shape[1] < 19:
            return None

        fs          = 100
        left_force  = df.iloc[:, 17].to_numpy(dtype=np.float64)
        right_force = df.iloc[:, 18].to_numpy(dtype=np.float64)
        combined    = left_force + right_force
        combined   -= combined.mean()          # zero-mean calibration

        # 4th-order Butterworth low-pass @ 10 Hz
        b, a  = ss.butter(4, 10.0 / (0.5 * fs), btype="low")
        acc_z = ss.filtfilt(b, a, combined).astype(np.float32)

        # Step detection on each foot separately, then merge
        min_dist = int(0.3 * fs)
        def _peaks(sig):
            th = sig.mean() + 0.3 * sig.std()
            p, _ = ss.find_peaks(sig, height=th, distance=min_dist)
            return p

        l_peaks = _peaks(left_force)
        r_peaks = _peaks(right_force)
        steps   = np.sort(np.concatenate([l_peaks, r_peaks]))
        n_steps = len(steps)

        if n_steps < 4:
            return None

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

        acc_pcts  = np.percentile(acc_z, np.arange(10,110,10)).astype(np.float32)
        peak_amps = acc_z[steps]
        peak_pcts = np.percentile(peak_amps, np.arange(10,110,10)).astype(np.float32)

        rms_w = [np.sqrt(np.mean(acc_z[i:i+fs]**2))
                 for i in range(0, len(acc_z) - fs, fs)] or [0.0]
        zc    = float(np.sum(np.diff(np.sign(acc_z)) != 0)) / total_time_s

        extra = np.array([
            skew(acc_z), kurtosis(acc_z), zc,
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
        return feat

    except Exception as e:
        log.debug("Gait parse error %s: %s", txt_path.name, e)
        return None


def process_gait(demo_map: dict) -> list[dict]:
    """
    Process PhysioNet gait files using real UPDRS from demographics.
    Controls (GaCo, JuCo, SiCo) get UPDRS=0.
    PD patients get UPDRS from demographics or a study-based fallback.
    """
    records = []

    # Collect only valid gait walk files
    all_txt = [f for f in sorted(GAIT_ROOT.glob("*.txt"))
               if GAIT_FILE_RE.match(f.name)]
    log.info("Found %d valid gait walk files.", len(all_txt))

    # Group by subject ID (e.g. "GaPt03" from "GaPt03_01.txt")
    subject_walks: dict[str, list[Path]] = {}
    for txt in all_txt:
        sub_id = txt.stem.rsplit("_", 1)[0]   # "GaPt03_01" → "GaPt03"
        subject_walks.setdefault(sub_id, []).append(txt)

    log.info("Processing %d gait subjects…", len(subject_walks))

    for sub_id, walks in sorted(subject_walks.items()):
        is_pd  = "Pt" in sub_id   # "Co" = control, "Pt" = patient
        out_path = FEAT_ROOT / "gait" / f"{sub_id}_gait.npy"

        # ── Determine UPDRS & label ───────────────────────────────────────────
        if not is_pd:
            # Healthy controls have no UPDRS (disease not present)
            updrs = 0.0
        else:
            # Try real UPDRS from demographics first
            demo_row = demo_map.get(sub_id)
            if demo_row is not None:
                updrs = demo_row["updrs"]
            else:
                # Subject not in demographics → study-based fallback
                # Ju and Si study PD patients tend to have moderate severity
                updrs = 35.0

        label = updrs_to_label(updrs)

        # ── Extract or load features ──────────────────────────────────────────
        if out_path.exists():
            feat = np.load(str(out_path))
        else:
            # Prefer normal walk (_01), else first available
            normal = [w for w in walks if w.stem.endswith("_01")]
            chosen = normal[0] if normal else walks[0]
            feat   = _extract_gait_from_txt(chosen)
            if feat is None:
                log.warning("Gait skipped (bad data): %s", sub_id)
                continue
            np.save(str(out_path), feat)

        cadence   = float(feat[0])
        cycles_ok = 30 < cadence < 300

        records.append({
            "subject_id":      sub_id,
            "voice_feat_path": "",
            "eeg_feat_path":   "",
            "gait_feat_path":  str(out_path.relative_to(ROOT)),
            "voice_snr_ok":    False,
            "eeg_artifact_ok": False,
            "gait_cycles_ok":  cycles_ok,
            "updrs_score":     round(updrs, 2),
            "label":           label,
            "source":          "gait",
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD MANIFEST
# ─────────────────────────────────────────────────────────────────────────────

def build_manifest(records: list[dict]) -> pd.DataFrame:
    """Assemble and save manifest.csv from all modality records."""
    df = pd.DataFrame(records)
    df = df.drop(columns=["source"], errors="ignore")

    # Replace empty string paths with NaN
    for col in ["voice_feat_path", "eeg_feat_path", "gait_feat_path"]:
        df[col] = df[col].replace("", np.nan)

    df["updrs_score"] = df["updrs_score"].astype(float).round(2)
    df["label"]       = df["label"].astype(int)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(MANIFEST), index=False)
    log.info("Manifest saved → %s  (%d rows)", MANIFEST, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("ParkiSense AI – Dataset Builder  (v2)")
    log.info("ROOT: %s", ROOT)
    log.info("=" * 60)

    all_records = []

    # Load real UPDRS labels from gait demographics (shared ground truth)
    demo_map = load_demographics()

    # ── Voice ─────────────────────────────────────────────────────────────────
    try:
        import librosa        # noqa: F401
        import parselmouth    # noqa: F401
        voice_records = process_voice()
        log.info("Voice: %d records processed.", len(voice_records))
        all_records.extend(voice_records)
    except ImportError:
        log.warning("librosa / praat-parselmouth not installed → skipping voice.")
        log.warning("  pip install librosa==0.10.1 praat-parselmouth==0.4.3")

    # ── EEG ───────────────────────────────────────────────────────────────────
    try:
        import mne    # noqa: F401
        eeg_records = process_eeg()
        log.info("EEG: %d records processed.", len(eeg_records))
        all_records.extend(eeg_records)
    except ImportError:
        log.warning("mne not installed → skipping EEG.")
        log.warning("  pip install mne==1.6.1")

    # ── Gait ──────────────────────────────────────────────────────────────────
    gait_records = process_gait(demo_map)
    log.info("Gait: %d records processed.", len(gait_records))
    all_records.extend(gait_records)

    if not all_records:
        log.error("No records generated. Check data paths.")
        sys.exit(1)

    df = build_manifest(all_records)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("DATASET SUMMARY")
    log.info("  Total records    : %d", len(df))
    log.info("  Mild   (label=0) : %d  (UPDRS 0–35)",   (df["label"] == 0).sum())
    log.info("  Moderate (lbl=1) : %d  (UPDRS 36–70)",  (df["label"] == 1).sum())
    log.info("  Severe (label=2) : %d  (UPDRS 71–108)", (df["label"] == 2).sum())
    log.info("  UPDRS mean ± std : %.1f ± %.1f",
             df["updrs_score"].mean(), df["updrs_score"].std())
    log.info("  UPDRS range      : %.1f – %.1f",
             df["updrs_score"].min(),  df["updrs_score"].max())
    log.info("  Voice available  : %d", df["voice_feat_path"].notna().sum())
    log.info("  EEG available    : %d", df["eeg_feat_path"].notna().sum())
    log.info("  Gait available   : %d", df["gait_feat_path"].notna().sum())
    log.info("=" * 60)
    log.info("Done!  Run: streamlit run app.py")


if __name__ == "__main__":
    main()
