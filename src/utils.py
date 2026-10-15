"""
ParkiSense AI - Utility Functions
==================================
Shared helpers for:
  - Logging setup
  - Tensor / NumPy conversion
  - Standard dataset splits
  - Inference helpers
  - ReportLab PDF generation utility
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a human-readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_LABELS: Dict[int, str] = {0: "Mild", 1: "Moderate", 2: "Severe"}
SEVERITY_COLORS: Dict[int, str] = {0: "#2ECC71", 1: "#F39C12", 2: "#E74C3C"}


def updrs_to_label(updrs_score: float) -> Tuple[int, str]:
    """
    Map a predicted UPDRS-equivalent score (model output: 0–100) to a severity tier.

    The model regression head outputs sigmoid × 100, representing the full
    0–108 UPDRS range compressed to 0–100. Thresholds are scaled accordingly:

        0–108  UPDRS  →  0–100  model output
        0–35   Mild   →  0–32.4   (35 / 108 × 100)
        36–70  Mod.   →  32.4–64.8 (70 / 108 × 100)
        71+    Severe →  > 64.8

    Parameters
    ----------
    updrs_score : float
        Model regression output in [0, 100].

    Returns
    -------
    (label_int, label_str) : Tuple[int, str]
    """
    # Thresholds scaled to 0–100 model output range
    MILD_MAX     = (35.0 / 108.0) * 100.0   # ≈ 32.4
    MODERATE_MAX = (70.0 / 108.0) * 100.0   # ≈ 64.8

    if updrs_score <= MILD_MAX:
        return 0, "Mild"
    elif updrs_score <= MODERATE_MAX:
        return 1, "Moderate"
    else:
        return 2, "Severe"


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module,
    voice_feat: np.ndarray,
    eeg_feat: np.ndarray,
    gait_feat: np.ndarray,
    quality_flags: Tuple[bool, bool, bool],
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run a single-sample forward pass through ParkiSenseFusionNet.

    Severity determination strategy
    --------------------------------
    The **regression head** (continuous UPDRS score 0–100) is the primary
    source of truth for the severity tier — it is directly interpretable and
    consistent with the displayed number.

    The **classification head** softmax probabilities are returned as a
    supplementary confidence signal (useful for uncertainty visualisation)
    but do **not** override the regression-derived label.

    Parameters
    ----------
    model : ParkiSenseFusionNet (or any compatible nn.Module)
    voice_feat : np.ndarray (52,)
    eeg_feat   : np.ndarray (n_eeg_feat,)
    gait_feat  : np.ndarray (42,)
    quality_flags : tuple of 3 booleans (voice_ok, eeg_ok, gait_ok)
    device : str | None
        Target device string ("cpu" or "cuda"). Auto-detected if None.

    Returns
    -------
    dict with keys:
      "updrs_score"      : float         – predicted score ∈ [0, 100]
      "updrs_equivalent" : float         – score rescaled to 0–108 UPDRS range
      "severity_label"   : int           – regression-derived class {0,1,2}
      "severity_name"    : str           – "Mild" | "Moderate" | "Severe"
      "cls_probs"        : np.ndarray(3) – classification softmax (supplementary)
      "cls_label"        : int           – classification head argmax (supplementary)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        voice_t = torch.tensor(voice_feat, dtype=torch.float32).unsqueeze(0).to(device)
        eeg_t   = torch.tensor(eeg_feat,   dtype=torch.float32).unsqueeze(0).to(device)
        gait_t  = torch.tensor(gait_feat,  dtype=torch.float32).unsqueeze(0).to(device)
        flags_t = torch.tensor([list(quality_flags)], dtype=torch.bool).to(device)

        updrs_pred, cls_logits = model(voice_t, eeg_t, gait_t, flags_t)

    updrs_score = float(updrs_pred.squeeze().cpu())          # 0–100
    cls_probs   = torch.softmax(cls_logits, dim=-1).squeeze().cpu().numpy()

    # ── Primary: derive severity from the regression score ────────────────────
    severity_label, severity_name = updrs_to_label(updrs_score)

    # ── Supplementary: raw classification head argmax ─────────────────────────
    cls_label = int(np.argmax(cls_probs))

    # Rescale model output (0–100) back to clinical UPDRS (0–108) for reporting
    updrs_equivalent = round(updrs_score * (108.0 / 100.0), 1)

    return {
        "updrs_score":      round(updrs_score, 2),
        "updrs_equivalent": updrs_equivalent,
        "severity_label":   severity_label,
        "severity_name":    severity_name,
        "cls_probs":        cls_probs,
        "cls_label":        cls_label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VALIDATION SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_manifest(
    manifest_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    """
    Stratified split of the manifest CSV into train/val/test DataFrames.

    Parameters
    ----------
    manifest_path : str | Path
    train_ratio   : float (default 0.7)
    val_ratio     : float (default 0.15)
    seed          : int

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(manifest_path)
    rng = np.random.default_rng(seed)

    # Stratify on label
    labels = df["label"].to_numpy()
    idx = np.arange(len(df))

    idx_train, idx_temp, _, labels_temp = train_test_split(
        idx, labels, test_size=(1 - train_ratio), random_state=seed, stratify=labels
    )
    val_size_rel = val_ratio / (1 - train_ratio)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=(1 - val_size_rel), random_state=seed, stratify=labels_temp
    )

    return df.iloc[idx_train], df.iloc[idx_val], df.iloc[idx_test]


# ─────────────────────────────────────────────────────────────────────────────
# THERAPY RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

THERAPY_RULES: Dict[str, Dict] = {
    "Mild": {
        "physiotherapy": "Maintain regular low-impact aerobic exercise (30 min/day, 5×/week). Focus on balance and flexibility training.",
        "speech_therapy": "Daily vocal exercises (LSVT LOUD protocol). Practice sustained phonation for 5 minutes.",
        "medication": "Adhere to current levodopa schedule. Monitor for motor fluctuations. Schedule quarterly neurology review.",
        "nutrition": "Mediterranean diet with high antioxidants (berries, leafy greens). Adequate vitamin D and omega-3 supplementation.",
    },
    "Moderate": {
        "physiotherapy": "Supervised physiotherapy 3×/week. Gait training with visual cues (floor markers). Resistance training to combat rigidity.",
        "speech_therapy": "Intensive LSVT LOUD therapy (16 sessions). Use voice amplification aids in noisy environments.",
        "medication": "Consider DBS candidacy review. Evaluate levodopa dose adjustment for 'off' episodes. Assess dopamine agonists.",
        "nutrition": "Protein redistribution diet to optimise levodopa absorption. Small frequent meals. Monitor dysphagia risk.",
    },
    "Severe": {
        "physiotherapy": "Daily physiotherapy with fall prevention focus. Assistive device prescription (walker/wheelchair). Home environment modification.",
        "speech_therapy": "AAC (Augmentative & Alternative Communication) evaluation. Dysphagia management with SLP for safe swallowing.",
        "medication": "Urgent palliative care review. Deep Brain Stimulation (DBS) or apomorphine pump evaluation. Pain management protocol.",
        "nutrition": "Supervised nutritional support (dietitian consult). Enteral feeding assessment if oral intake is insufficient.",
    },
}


def generate_therapy_recommendations(severity_name: str) -> Dict[str, str]:
    """Return therapy recommendations for a given severity tier."""
    return THERAPY_RULES.get(severity_name, THERAPY_RULES["Mild"])


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    subject_id: str,
    updrs_score: float,
    severity_name: str,
    cls_probs: np.ndarray,
    quality_flags: Tuple[bool, bool, bool],
    therapy_recs: Dict[str, str],
    longitudinal_data: Optional[List[Dict]] = None,
) -> bytes:
    """
    Generate a clinical PDF assessment report using ReportLab.

    Parameters
    ----------
    subject_id     : str
    updrs_score    : float – predicted UPDRS (0–100)
    severity_name  : str – "Mild" | "Moderate" | "Severe"
    cls_probs      : np.ndarray (3,)
    quality_flags  : tuple (voice_ok, eeg_ok, gait_ok)
    therapy_recs   : dict from generate_therapy_recommendations()
    longitudinal_data : list of dicts {"date": str, "updrs": float} | None

    Returns
    -------
    bytes – PDF file content
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
    except ImportError as e:
        raise ImportError(
            "PDF export requires 'reportlab'. Install via requirements.txt."
        ) from e

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=20, spaceAfter=6)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontSize=13, spaceAfter=4)
    body_style = styles["Normal"]

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("ParkiSense AI — Clinical Assessment Report", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2C3E50")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Subject ID:</b> {subject_id}", body_style))
    story.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 12))

    # ── Assessment Summary ────────────────────────────────────────────────────
    story.append(Paragraph("Assessment Summary", heading_style))
    severity_colors_map = {"Mild": "#27AE60", "Moderate": "#E67E22", "Severe": "#C0392B"}
    color_hex = severity_colors_map.get(severity_name, "#7F8C8D")

    summary_data = [
        ["Metric", "Value"],
        ["UPDRS Score (predicted)", f"{updrs_score:.1f} / 100"],
        ["Severity Tier", severity_name],
        ["Mild Probability", f"{cls_probs[0] * 100:.1f}%"],
        ["Moderate Probability", f"{cls_probs[1] * 100:.1f}%"],
        ["Severe Probability", f"{cls_probs[2] * 100:.1f}%"],
        ["Voice Quality (SNR OK)", "✓" if quality_flags[0] else "✗ (Fallback active)"],
        ["EEG Quality (Artifact OK)", "✓" if quality_flags[1] else "✗ (Fallback active)"],
        ["Gait Quality (Cycles OK)", "✓" if quality_flags[2] else "✗ (Fallback active)"],
    ]
    t = Table(summary_data, colWidths=[90 * mm, 80 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ECF0F1")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # ── Therapy Recommendations ───────────────────────────────────────────────
    story.append(Paragraph("Personalised Therapy Recommendations", heading_style))
    for category, recommendation in therapy_recs.items():
        story.append(Paragraph(f"<b>{category.replace('_', ' ').title()}:</b>", body_style))
        story.append(Paragraph(recommendation, body_style))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 10))

    # ── Longitudinal Data ─────────────────────────────────────────────────────
    if longitudinal_data:
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#BDC3C7")))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Longitudinal UPDRS Trend", heading_style))
        trend_data = [["Date", "UPDRS Score"]]
        for entry in longitudinal_data:
            trend_data.append([entry.get("date", ""), f"{entry.get('updrs', 0):.1f}"])
        trend_table = Table(trend_data, colWidths=[90 * mm, 80 * mm])
        trend_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498DB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(trend_table)

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#BDC3C7")))
    story.append(Paragraph(
        "<i>This report was generated automatically by ParkiSense AI. "
        "It is intended for clinical reference only and should be reviewed by a qualified neurologist.</i>",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    ))

    doc.build(story)
    return buffer.getvalue()
