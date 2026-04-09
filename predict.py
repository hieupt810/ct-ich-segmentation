"""
predict.py — Single-case inference from a NIfTI CT volume.

Given a NIfTI file (.nii / .nii.gz) for one patient, outputs:
  - Per-slice binary mask .npy files
  - Overlay PNG per positive slice (CT + predicted mask)
  - A summary panel PNG showing all predicted hemorrhage slices

Usage:
    conda activate ct-segmentation
    python predict.py --input data/raw/ct_scans/049.nii --output results/case_049/
    python predict.py --input data/raw/ct_scans/049.nii --output results/ --threshold 0.4
"""

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

import config
from dataset import build_multiwindow_image
from model import build_model, load_checkpoint
from preprocess import clip_hu

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Load & Preprocess ────────────────────────────────────────────────────────


def load_nifti_slices(nii_path: Path) -> tuple[list[np.ndarray], int]:
    """
    Load a NIfTI volume and return a list of HU-clipped 2-D axial slices.
    Slices are along axis 2: vol[:, :, i].
    """
    nii = nib.load(str(nii_path))
    vol = nii.get_fdata(dtype=np.float32)

    while vol.ndim > 3:
        vol = vol[..., 0]

    vol = clip_hu(vol)  # HU-clipped float32, same as preprocess.py
    slices = [vol[..., i] for i in range(vol.shape[2])]
    return slices, len(slices)


def build_input_tensor(
    hu_slice: np.ndarray, img_size: int = config.IMG_SIZE
) -> torch.Tensor:
    """
    Convert a single HU-clipped slice to the 3-channel multi-window tensor
    expected by the model — identical to dataset.py's __getitem__ logic.
    Returns (1, 3, H, W) float32 tensor.
    """
    image = build_multiwindow_image(hu_slice)  # (H, W, 3) [0,1]
    image = cv2.resize(
        image, (img_size, img_size), interpolation=cv2.INTER_LINEAR
    )  # resize spatial dims
    tensor = torch.from_numpy(image.transpose(2, 0, 1))  # (3, H, W)
    return tensor.unsqueeze(0)  # (1, 3, H, W)


# ─── Inference ────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_inference(
    model,
    slices: list[np.ndarray],
    device: torch.device,
    threshold: float = config.THRESHOLD,
) -> list[np.ndarray]:
    """Return list of binary mask arrays (H, W) for each input slice."""
    model.eval()
    masks = []

    for hu_slice in tqdm(slices, desc="Inferring"):
        tensor = build_input_tensor(hu_slice).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits)

        if config.TTA_ENABLED:
            flipped = torch.flip(tensor, dims=[-1])
            logits_flip = model(flipped)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[-1])
            probs = (probs + probs_flip) / 2.0

        mask = (probs[0, 0] > threshold).cpu().numpy().astype(np.uint8)
        masks.append(mask)

    return masks


# ─── Visualization ────────────────────────────────────────────────────────────


def save_results(
    slices: list[np.ndarray],
    masks: list[np.ndarray],
    out_dir: Path,
    case_name: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_indices = [i for i, m in enumerate(masks) if m.sum() > 0]
    log.info(
        "%d / %d slices with predicted hemorrhage.",
        len(positive_indices),
        len(slices),
    )

    if not positive_indices:
        log.info("No hemorrhage predicted in this case.")
        return

    for i in positive_indices:
        uid = f"{case_name}_s{i:03d}"
        np.save(out_dir / f"{uid}_mask.npy", masks[i])

        # Show brain/ICH window (Ch0) for the overlay
        display = np.clip(slices[i], config.HU_WIN0_MIN, config.HU_WIN0_MAX)
        display = (display - config.HU_WIN0_MIN) / (
            config.HU_WIN0_MAX - config.HU_WIN0_MIN
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(display, cmap="gray")
        ax1.set_title("CT (ICH window)")
        ax1.axis("off")
        ax2.imshow(masks[i], cmap="hot")
        ax2.set_title("Prediction")
        ax2.axis("off")
        fig.suptitle(uid)
        plt.tight_layout()
        plt.savefig(out_dir / f"{uid}_overlay.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # Summary panel (up to 9 slices)
    n = min(len(positive_indices), 9)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).ravel()

    for j, i in enumerate(positive_indices[:n]):
        display = np.clip(slices[i], config.HU_WIN0_MIN, config.HU_WIN0_MAX)
        display = (display - config.HU_WIN0_MIN) / (
            config.HU_WIN0_MAX - config.HU_WIN0_MIN
        )
        axes[j].imshow(display, cmap="gray")
        axes[j].imshow(masks[i], cmap="hot", alpha=0.4)
        axes[j].set_title(f"slice {i}", fontsize=7)
        axes[j].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Predicted Hemorrhage — {case_name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Summary saved to %s", out_dir / "summary.png")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="ICH segmentation — single-case inference"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a NIfTI CT volume (.nii or .nii.gz)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for masks and overlays",
    )
    parser.add_argument(
        "--threshold",
        default=config.THRESHOLD,
        type=float,
        help=f"Binary threshold (default: {config.THRESHOLD})",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help="Path to model checkpoint (default: checkpoints/best_model.pth)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    ckpt_path = args.checkpoint or (config.CHECKPOINT_DIR / "best_model.pth")
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s — run train.py first.", ckpt_path)
        return

    if not args.input.exists():
        log.error("Input NIfTI not found: %s", args.input)
        return

    model = build_model().to(device)
    epoch, _ = load_checkpoint(model, ckpt_path, device)
    log.info("Loaded checkpoint (epoch %d) from %s", epoch, ckpt_path)

    log.info("Loading NIfTI: %s", args.input)
    slices, n = load_nifti_slices(args.input)
    log.info("Loaded %d slices.", n)

    masks = run_inference(model, slices, device, threshold=args.threshold)
    save_results(slices, masks, args.output, case_name=args.input.stem)


if __name__ == "__main__":
    main()
