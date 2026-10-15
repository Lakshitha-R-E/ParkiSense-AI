"""
ParkiSense AI - Deep Learning Architecture
==========================================
Components
----------
1. ModalityEncoder       – feed-forward encoder (voice / EEG / gait → embed_dim)
2. SafeFallbackGate      – replaces bad-quality embeddings with a learned filler
3. ParkiSenseFusionNet   – transformer fusion + dual regression/classification heads
4. parkisense_loss       – joint MSE + CrossEntropy multi-task loss
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# HYPER-PARAMETERS (from spec)
# ─────────────────────────────────────────────────────────────────────────────
EMBED_DIM: int = 128
N_HEADS: int = 4
N_LAYERS: int = 2
N_MODALITIES: int = 3          # voice, EEG, gait
N_CLASSES: int = 3             # Mild, Moderate, Severe
LAMBDA_REG: float = 0.6        # weight for MSE loss
LAMBDA_CLS: float = 0.4        # weight for CE loss


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODALITY ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """
    Feed-forward encoder mapping raw feature vectors to a shared embedding space.

    Architecture::

        Linear(in_dim → hidden_dim) → GELU → LayerNorm
        → Linear(hidden_dim → embed_dim) → GELU → LayerNorm

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input feature vector.
    embed_dim : int
        Output embedding dimension (default 128).
    hidden_dim : int
        Hidden layer width (default 256).
    dropout : float
        Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, in_dim)

        Returns
        -------
        Tensor of shape (batch, embed_dim)
        """
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAFE FALLBACK GATE
# ─────────────────────────────────────────────────────────────────────────────

class SafeFallbackGate(nn.Module):
    """
    Quality-aware gating mechanism for multimodal fusion.

    For each modality token in a batch:
      - If quality flag is **True** (signal is clean), pass the true embedding.
      - If quality flag is **False** (signal is noisy/absent), substitute a
        learned *filler* vector so the transformer can re-weight the remaining
        valid modalities without receiving corrupted information.

    The ``filler`` vector is an ``nn.Parameter`` jointly optimised with the
    rest of the network.

    Parameters
    ----------
    embed_dim : int
        Shared embedding dimension (must match ``ModalityEncoder`` output).
    n_modalities : int
        Number of modality streams (default 3).
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        n_modalities: int = N_MODALITIES,
    ) -> None:
        super().__init__()
        # One learnable filler per modality (shape: n_modalities × embed_dim)
        self.filler = nn.Parameter(
            torch.zeros(n_modalities, embed_dim)
        )
        nn.init.normal_(self.filler, mean=0.0, std=0.02)

    def forward(
        self,
        embeddings: torch.Tensor,
        quality_flags: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : Tensor of shape (batch, n_modalities, embed_dim)
            Stacked encoder outputs for all modalities.
        quality_flags : BoolTensor of shape (batch, n_modalities)
            True = good quality signal; False = missing / corrupted.

        Returns
        -------
        Tensor of shape (batch, n_modalities, embed_dim)
            Gated embeddings – true embeddings or learned fillers.
        """
        batch_size = embeddings.size(0)

        # Expand filler: (1, n_modalities, embed_dim) → (batch, n_mod, embed_dim)
        filler_expanded = self.filler.unsqueeze(0).expand(batch_size, -1, -1)

        # quality_flags: (batch, n_modalities) → (batch, n_modalities, 1)
        mask = quality_flags.unsqueeze(-1).float()  # 1.0 = good, 0.0 = bad

        # Blend: good embeddings pass through; bad ones are replaced by filler
        gated = mask * embeddings + (1.0 - mask) * filler_expanded
        return gated


# ─────────────────────────────────────────────────────────────────────────────
# 3. POSITIONAL ENCODING (sinusoidal)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for a sequence of modality tokens."""

    def __init__(self, embed_dim: int = EMBED_DIM, max_len: int = 16) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, embed_dim)"""
        return x + self.pe[:, : x.size(1), :]


# ─────────────────────────────────────────────────────────────────────────────
# 4. PARKISENSE FUSION NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class ParkiSenseFusionNet(nn.Module):
    """
    End-to-end multimodal PD severity assessment network.

    Architecture overview::

        [voice_feat]  →  VoiceEncoder  ─┐
        [eeg_feat]    →  EEGEncoder    ─┤→ SafeFallbackGate → PositionalEncoding
        [gait_feat]   →  GaitEncoder   ─┘         ↓
                                          TransformerEncoder (L=2, H=4)
                                                   ↓
                                             CLS token / mean pool
                                          ┌────────┴────────┐
                                    RegressionHead     ClassificationHead
                                          ↓                  ↓
                                   UPDRS score (0-100)   3-class logits

    Parameters
    ----------
    voice_dim : int
        Voice feature dimensionality (default 52).
    eeg_dim : int
        EEG feature dimensionality (default 64 – varies with channel count).
    gait_dim : int
        Gait feature dimensionality (default 42).
    embed_dim : int
        Shared embedding dimension (default 128).
    n_heads : int
        Multi-head attention heads (default 4).
    n_layers : int
        Transformer encoder layers (default 2).
    n_classes : int
        Severity classes (default 3).
    dropout : float
        Dropout in transformer layers (default 0.1).
    """

    def __init__(
        self,
        voice_dim: int = 52,
        eeg_dim: int = 64,
        gait_dim: int = 42,
        embed_dim: int = EMBED_DIM,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        n_classes: int = N_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Modality encoders ─────────────────────────────────────────────────
        self.voice_encoder = ModalityEncoder(voice_dim, embed_dim, dropout=dropout)
        self.eeg_encoder = ModalityEncoder(eeg_dim, embed_dim, dropout=dropout)
        self.gait_encoder = ModalityEncoder(gait_dim, embed_dim, dropout=dropout)

        # ── Safe Fallback Gate ────────────────────────────────────────────────
        self.fallback_gate = SafeFallbackGate(embed_dim, N_MODALITIES)

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=N_MODALITIES)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # ── Regression head → UPDRS score ∈ [0, 100] ─────────────────────────
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),  # outputs ∈ (0, 1); multiply by 100 at inference
        )

        # ── Classification head → 3-class softmax ────────────────────────────
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes),
            # Raw logits returned; softmax applied externally or via CE loss
        )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        voice: torch.Tensor,
        eeg: torch.Tensor,
        gait: torch.Tensor,
        quality_flags: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        voice : Tensor (batch, voice_dim)
        eeg   : Tensor (batch, eeg_dim)
        gait  : Tensor (batch, gait_dim)
        quality_flags : BoolTensor (batch, 3)  [voice_ok, eeg_ok, gait_ok]

        Returns
        -------
        updrs_score : Tensor (batch,)  – continuous score ∈ [0, 100]
        cls_logits  : Tensor (batch, 3) – raw classification logits
        """
        # ── Encode each modality ──────────────────────────────────────────────
        v_emb = self.voice_encoder(voice)   # (batch, embed_dim)
        e_emb = self.eeg_encoder(eeg)       # (batch, embed_dim)
        g_emb = self.gait_encoder(gait)     # (batch, embed_dim)

        # ── Stack → (batch, n_modalities, embed_dim) ──────────────────────────
        tokens = torch.stack([v_emb, e_emb, g_emb], dim=1)

        # ── Safe Fallback gating ──────────────────────────────────────────────
        tokens = self.fallback_gate(tokens, quality_flags)

        # ── Positional encoding ───────────────────────────────────────────────
        tokens = self.pos_encoding(tokens)

        # ── Transformer fusion ────────────────────────────────────────────────
        fused = self.transformer(tokens)  # (batch, n_modalities, embed_dim)

        # ── Global average pooling across modality tokens ─────────────────────
        pooled = fused.mean(dim=1)        # (batch, embed_dim)

        # ── Prediction heads ──────────────────────────────────────────────────
        updrs_score = self.regression_head(pooled).squeeze(-1) * 100.0  # (batch,)
        cls_logits = self.classification_head(pooled)                    # (batch, 3)

        return updrs_score, cls_logits

    # ── Attention weights for visualisation ───────────────────────────────────

    @torch.no_grad()
    def get_attention_weights(
        self,
        voice: torch.Tensor,
        eeg: torch.Tensor,
        gait: torch.Tensor,
        quality_flags: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Extract per-layer attention weight matrices for heatmap visualisation.

        Returns
        -------
        list of Tensor (batch, n_heads, seq_len, seq_len) per layer
        """
        v_emb = self.voice_encoder(voice)
        e_emb = self.eeg_encoder(eeg)
        g_emb = self.gait_encoder(gait)
        tokens = torch.stack([v_emb, e_emb, g_emb], dim=1)
        tokens = self.fallback_gate(tokens, quality_flags)
        tokens = self.pos_encoding(tokens)

        attn_weights = []
        x = tokens
        for layer in self.transformer.layers:
            # Hook into self-attention
            attn_out, attn_w = layer.self_attn(x, x, x, need_weights=True)
            attn_weights.append(attn_w)
            x = layer.norm1(x + attn_out)
            x = layer.norm2(x + layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x)))
            ))
        return attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# 5. JOINT MULTI-TASK LOSS
# ─────────────────────────────────────────────────────────────────────────────

def parkisense_loss(
    updrs_pred: torch.Tensor,
    updrs_target: torch.Tensor,
    cls_logits: torch.Tensor,
    cls_target: torch.Tensor,
    lambda1: float = LAMBDA_REG,
    lambda2: float = LAMBDA_CLS,
) -> torch.Tensor:
    """
    Weighted multi-task joint loss:

    .. math::

        \\mathcal{L} = \\lambda_1 \\cdot \\text{MSE}(\\hat{s}, s)
                     + \\lambda_2 \\cdot \\text{CE}(\\hat{y}, y)

    Parameters
    ----------
    updrs_pred : Tensor (batch,)
        Predicted continuous UPDRS score (0–100).
    updrs_target : Tensor (batch,)
        Ground-truth normalised UPDRS ∈ [0, 1] (model internally outputs 0-100
        from a Sigmoid × 100; targets should be scaled accordingly).
    cls_logits : Tensor (batch, 3)
        Raw classification logits.
    cls_target : LongTensor (batch,)
        Integer class labels {0, 1, 2}.
    lambda1 : float
        Weight for regression loss (default 0.6).
    lambda2 : float
        Weight for classification loss (default 0.4).

    Returns
    -------
    Tensor – scalar joint loss.
    """
    # Normalise predictions to [0, 1] for MSE (target is already normalised)
    updrs_pred_norm = updrs_pred / 100.0
    mse_loss = F.mse_loss(updrs_pred_norm, updrs_target)
    ce_loss = F.cross_entropy(cls_logits, cls_target)
    return lambda1 * mse_loss + lambda2 * ce_loss
