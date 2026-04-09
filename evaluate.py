"""
evaluate.py — Full evaluation on the test split.

Loads checkpoints/best_model.pth, runs inference on test set,
computes per-case and aggregate metrics, and saves:
  results/metrics.csv       — per-slice metrics
  results/summary.txt       — aggregate mean ± std
  results/overlays/         — visual prediction overlays (optional)

Features:
  - Test-Time Augmentation (TTA): horizontal flip averaging (config.TTA_ENABLED)
  - Saves qualitative overlays for visual inspection

Usage:
    conda activate ct-segmentation
    python evaluate.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

import config
from dataset import get_loader
from metrics import aggregate_metrics, compute_all
from model import build_model, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── TTA ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def predict_with_tta(model, images: torch.Tensor) -> torch.Tensor:
    """Return sigmoid probabilities averaged over original + H-flip."""
    logits_orig = model(images)
    probs_orig = torch.sigmoid(logits_orig)

    if config.TTA_ENABLED:
        flipped = torch.flip(images, dims=[-1])  # horizontal flip
        logits_flip = model(flipped)
        probs_flip = torch.sigmoid(logits_flip)
        probs_flip = torch.flip(probs_flip, dims=[-1])  # flip back
        return (probs_orig + probs_flip) / 2.0

    return probs_orig


# ─── Overlay Visualization ────────────────────────────────────────────────────


def save_overlay(
    image_np: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    uid: str,
    out_dir: Path,
) -> None:
    """Save a 3-panel overlay: CT slice | GT mask | Prediction."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("CT (channel 1)")
    axes[1].imshow(gt_mask, cmap="hot")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_mask, cmap="hot")
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(uid, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / f"{uid}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Main Evaluation ──────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(device)
    ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
    if not ckpt_path.exists():
        log.error("No checkpoint found at %s — run train.py first.", ckpt_path)
        return

    epoch, best_dice = load_checkpoint(model, ckpt_path, device)
    model.eval()
    log.info("Loaded checkpoint from epoch %d (val Dice=%.4f)", epoch, best_dice)

    # ── Data ──────────────────────────────────────────────────────────────────
    test_loader = get_loader("test", shuffle=False)
    overlay_dir = config.RESULTS_DIR / "overlays"
    all_records = []
    max_overlays = 20  # save at most this many visual overlays

    log.info("TTA enabled: %s", config.TTA_ENABLED)

    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        uids = batch["uid"]

        with autocast(device=device.type, enabled=config.AMP):
            probs = predict_with_tta(model, images)

        for i in range(images.size(0)):
            m = compute_all(probs[i], masks[i], threshold=config.THRESHOLD)
            m["uid"] = uids[i]
            all_records.append(m)

            # Save overlay for positive cases up to max_overlays
            if (
                batch["has_hemorrhage"][i]
                and len(list(overlay_dir.glob("*.png"))) < max_overlays
            ):
                img_np = images[i, 1].cpu().numpy()  # middle channel
                pred_np = (
                    (probs[i, 0] > config.THRESHOLD).cpu().numpy().astype(np.uint8)
                )
                gt_np = masks[i, 0].cpu().numpy().astype(np.uint8)
                save_overlay(img_np, pred_np, gt_np, uids[i], overlay_dir)

    # ── Metrics CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    log.info("Per-slice metrics saved to %s", config.RESULTS_DIR / "metrics.csv")

    # ── Aggregate Summary ─────────────────────────────────────────────────────
    numeric_records = [{k: v for k, v in r.items() if k != "uid"} for r in all_records]
    agg = aggregate_metrics(numeric_records)

    summary_lines = ["ICH Segmentation — Test Set Results", "=" * 40]
    for metric, stats in agg.items():
        summary_lines.append(
            f"{metric:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}  (n={stats['n']})"
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = config.RESULTS_DIR / "summary.txt"
    summary_path.write_text(summary_text)
    log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
