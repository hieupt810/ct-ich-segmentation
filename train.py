"""
train.py — Training loop for ICH binary segmentation.

Features:
  - Mixed-precision (AMP) training
  - AdamW optimizer + cosine annealing LR
  - Early stopping on validation Dice
  - TensorBoard logging (loss, Dice, IoU, LR)
  - Saves best_model.pth and last.pth to checkpoints/

Usage:
    conda activate ct-segmentation
    python train.py
"""

import logging
import random

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_loader
from losses import get_loss
from metrics import aggregate_metrics, compute_all
from model import build_model, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Reproducibility ──────────────────────────────────────────────────────────


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── One Epoch ────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=config.AMP):
            logits = model(images)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_metrics = []

    for batch in tqdm(loader, desc="Val  ", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=config.AMP):
            logits = model(images)
            loss = loss_fn(logits, masks)

        total_loss += loss.item() * images.size(0)

        # Compute metrics per sample in the batch
        probs = torch.sigmoid(logits)
        for i in range(images.size(0)):
            m = compute_all(probs[i], masks[i], threshold=config.THRESHOLD)
            all_metrics.append(m)

    avg_loss = total_loss / len(loader.dataset)
    agg = aggregate_metrics(all_metrics)
    return avg_loss, agg


# ─── Main Training Loop ───────────────────────────────────────────────────────


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = get_loader("train")
    val_loader = get_loader("val", shuffle=False)
    log.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(device)
    log.info("Model: %s | Encoder: %s", config.MODEL_NAME, config.ENCODER)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable parameters: %s", f"{total_params:,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    loss_fn = get_loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=config.LR_MIN
    )
    scaler = GradScaler(enabled=config.AMP)

    # ── Logging ───────────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(config.LOG_DIR))

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_dice = 0.0
    patience_counter = 0
    best_ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
    last_ckpt_path = config.CHECKPOINT_DIR / "last.pth"

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device
        )
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        val_dice = val_metrics["dice"]["mean"]
        val_iou = val_metrics["iou"]["mean"]
        cur_lr = optimizer.param_groups[0]["lr"]

        log.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
            "Dice=%.4f | IoU=%.4f | LR=%.2e",
            epoch,
            config.EPOCHS,
            train_loss,
            val_loss,
            val_dice,
            val_iou,
            cur_lr,
        )

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metric/Dice", val_dice, epoch)
        writer.add_scalar("Metric/IoU", val_iou, epoch)
        writer.add_scalar("LR", cur_lr, epoch)

        # Checkpoint
        save_checkpoint(model, last_ckpt_path, epoch, val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            save_checkpoint(model, best_ckpt_path, epoch, val_dice)
            log.info(
                "  --> New best Dice: %.4f  (saved to %s)", best_dice, best_ckpt_path
            )
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                log.info(
                    "Early stopping triggered at epoch %d (patience=%d)",
                    epoch,
                    config.EARLY_STOP_PATIENCE,
                )
                break

    writer.close()
    log.info("Training complete. Best Dice: %.4f", best_dice)
    log.info("Best checkpoint: %s", best_ckpt_path)


if __name__ == "__main__":
    main()
