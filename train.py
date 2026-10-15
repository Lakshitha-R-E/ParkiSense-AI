"""
ParkiSense AI - Model Training Pipeline
=======================================
Trains the multimodal ParkiSenseFusionNet end-to-end on extracted features:
  - Voice (52-dim), EEG (64-dim), Gait (42-dim)
  - Quality-aware SafeFallbackGate
  - Multi-task objective: Loss = 0.6 * MSE(UPDRS) + 0.4 * CE(Severity)
  - Saves best checkpoint to: models/parkisense_best.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset import ParkiSenseDataset
from src.models import ParkiSenseFusionNet, parkisense_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    correct = 0
    total = 0

    for batch in loader:
        voice = batch["voice"].to(device)
        eeg = batch["eeg"].to(device)
        gait = batch["gait"].to(device)
        flags = batch["quality_flags"].to(device)
        updrs_target = batch["updrs"].to(device)  # normalised [0, 1]
        cls_target = batch["label"].to(device)    # {0, 1, 2}

        optimizer.zero_grad()

        # Forward pass: updrs_pred is in [0, 100]
        updrs_pred, cls_logits = model(voice, eeg, gait, flags)

        loss = parkisense_loss(
            updrs_pred=updrs_pred,
            updrs_target=updrs_target,
            cls_logits=cls_logits,
            cls_target=cls_target,
            lambda1=0.6,
            lambda2=0.4,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        batch_size = voice.size(0)
        total_loss += loss.item() * batch_size

        # MAE against un-normalised UPDRS target
        target_updrs_raw = updrs_target * 108.0
        pred_updrs_scaled = (updrs_pred / 100.0) * 108.0
        total_mae += torch.abs(pred_updrs_scaled - target_updrs_raw).sum().item()

        preds = cls_logits.argmax(dim=-1)
        correct += (preds == cls_target).sum().item()
        total += batch_size

    return {
        "loss": total_loss / max(total, 1),
        "mae": total_mae / max(total, 1),
        "acc": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    correct = 0
    total = 0

    for batch in loader:
        voice = batch["voice"].to(device)
        eeg = batch["eeg"].to(device)
        gait = batch["gait"].to(device)
        flags = batch["quality_flags"].to(device)
        updrs_target = batch["updrs"].to(device)
        cls_target = batch["label"].to(device)

        updrs_pred, cls_logits = model(voice, eeg, gait, flags)

        loss = parkisense_loss(
            updrs_pred=updrs_pred,
            updrs_target=updrs_target,
            cls_logits=cls_logits,
            cls_target=cls_target,
            lambda1=0.6,
            lambda2=0.4,
        )

        batch_size = voice.size(0)
        total_loss += loss.item() * batch_size

        target_updrs_raw = updrs_target * 108.0
        pred_updrs_scaled = (updrs_pred / 100.0) * 108.0
        total_mae += torch.abs(pred_updrs_scaled - target_updrs_raw).sum().item()

        preds = cls_logits.argmax(dim=-1)
        correct += (preds == cls_target).sum().item()
        total += batch_size

    return {
        "loss": total_loss / max(total, 1),
        "mae": total_mae / max(total, 1),
        "acc": correct / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train ParkiSense AI Fusion Network")
    parser.add_argument("--manifest", type=str, default="datasets/manifest.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="models")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "parkisense_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using compute device: %s", device)

    # Load dataset
    full_dataset = ParkiSenseDataset(manifest_path=manifest_path, root_dir=ROOT, augment=True)
    n_samples = len(full_dataset)
    log.info("Loaded dataset with %d samples.", n_samples)

    # Train / Val Split (80 / 20)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = Subset(full_dataset, train_idx)
    # Val dataset without augmentation
    val_base = ParkiSenseDataset(manifest_path=manifest_path, root_dir=ROOT, augment=False)
    val_ds = Subset(val_base, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Instantiate FusionNet
    model = ParkiSenseFusionNet(
        voice_dim=52,
        eeg_dim=64,
        gait_dim=42,
        embed_dim=128,
        n_heads=4,
        n_layers=2,
        n_classes=3,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    log.info("Starting training for %d epochs...", args.epochs)
    log.info("-" * 65)
    log.info(f"{'Epoch':<7} | {'Train Loss':<11} | {'Val Loss':<11} | {'Val MAE':<9} | {'Val Acc':<8}")
    log.info("-" * 65)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        log.info(
            f"{epoch:<7} | {train_metrics['loss']:<11.4f} | {val_metrics['loss']:<11.4f} | "
            f"{val_metrics['mae']:<9.2f} | {val_metrics['acc'] * 100:<7.1f}%"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_acc": val_metrics["acc"],
            }, str(best_model_path))

    log.info("-" * 65)
    log.info("Training complete! Best checkpoint saved to: %s (Val Loss: %.4f)", best_model_path, best_val_loss)


if __name__ == "__main__":
    main()
