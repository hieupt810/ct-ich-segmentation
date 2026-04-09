"""
dataset.py — PyTorch Dataset for ICH binary segmentation.

Each sample returns:
  image : (3, H, W) float32 tensor  — 3-channel multi-window input
  mask  : (1, H, W) float32 tensor  — binary hemorrhage mask {0, 1}

Multi-window channel construction (Hssayeni et al., PMC10417300):
  Channel 0 — Brain/ICH window  [HU_WIN0_MIN, HU_WIN0_MAX] normalized to [0,1]
               Tight window highlighting blood (50-90 HU range within -10…90)
  Channel 1 — Broad window      [HU_WIN1_MIN, HU_WIN1_MAX] normalized to [0,1]
               Paper's W.L=80, W.D=180 → more anatomical context
  Channel 2 — Inverted Ch0      1.0 - Ch0
               Negative transform: hyperattenuating ICH becomes dark,
               improves gradient signal for the encoder.

The stored .npy files contain raw HU-clipped values (not normalized) so all
three normalizations can be computed from a single stored array.

Augmentation strategy (matches paper):
  Training : geometric (rotations at 15/30/60/90/180/270°, translation,
             flips, elastic) + intensity (brightness/contrast, noise)
  Val/Test : resize only (no augmentation)
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

import config

# ─── Multi-Window Normalization ───────────────────────────────────────────────


def normalize_window(hu_slice: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Clip HU values to [hu_min, hu_max] and normalize to [0, 1]."""
    clipped = np.clip(hu_slice, hu_min, hu_max)
    return (clipped - hu_min) / (hu_max - hu_min)


def build_multiwindow_image(hu_slice: np.ndarray) -> np.ndarray:
    """
    Convert a single HU-clipped slice (H, W) into a 3-channel (H, W, 3) image.

    Channel 0: Brain/ICH window  → tight, makes blood stand out
    Channel 1: Broad window      → more context (paper's main window)
    Channel 2: Inverted Ch0      → negative transform for contrast enhancement
    """
    ch0 = normalize_window(hu_slice, config.HU_WIN0_MIN, config.HU_WIN0_MAX)
    ch1 = normalize_window(hu_slice, config.HU_WIN1_MIN, config.HU_WIN1_MAX)
    ch2 = 1.0 - ch0  # inversion

    return np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)  # (H, W, 3)


# ─── Augmentation Pipelines ───────────────────────────────────────────────────


def get_train_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    """
    Augmentation matching the paper:
    - Rotations: 15°, 30°, 60°, 90°, 180°, 270° (paper Table 1)
    - Translations: ±10 % of image size
    - Additional: flips, elastic deformation, brightness/contrast, noise
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            # Paper's rotation set: 15, 30, 60, 90, 180, 270 degrees
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.6),  # covers ±15° and ±30° range
            A.OneOf(
                [
                    A.Rotate(limit=(60, 60), p=1.0),
                    A.Rotate(limit=(90, 90), p=1.0),
                    A.Rotate(limit=(180, 180), p=1.0),
                    A.Rotate(limit=(270, 270), p=1.0),
                ],
                p=0.3,
            ),
            # Paper's translation: ±0.1 of image size
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.05, rotate_limit=0, p=0.4
            ),
            # Additional geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.2),
            A.GridDistortion(p=0.15),
            # Intensity augmentations
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussNoise(var_limit=(0.001, 0.005), p=0.2),
            # Data is already [0,1]; identity normalization so ToTensorV2 works
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2(),
        ]
    )


# ─── Dataset ──────────────────────────────────────────────────────────────────


class ICHDataset(Dataset):
    """
    Loads HU-clipped .npy slices and returns 3-channel multi-window tensors.

    Args:
        csv_path  : Path to split CSV (train.csv / val.csv / test.csv).
        transform : Albumentations Compose pipeline.
    """

    def __init__(self, csv_path: Path, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform
        self.root = config.ROOT_DIR

    def __len__(self) -> int:
        return len(self.df)

    def _load_npy(self, rel_path: str) -> np.ndarray:
        return np.load(self.root / rel_path).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load raw HU-clipped slice and build 3-channel multi-window image
        hu_slice = self._load_npy(row["image_path"])  # (H, W) HU-clipped float32
        image = build_multiwindow_image(hu_slice)  # (H, W, 3) float32 [0,1]

        mask = self._load_npy(row["mask_path"]).astype(np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]

        # Align spatial size if needed
        h, w = image.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # (3, H, W) tensor
            mask = augmented["mask"]  # (H, W) tensor

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "has_hemorrhage": int(row["has_hemorrhage"]),
            "uid": row["uid"],
        }


# ─── Convenience Loaders ──────────────────────────────────────────────────────


def get_loader(
    split: str, batch_size: int = config.BATCH_SIZE, shuffle: bool | None = None
) -> DataLoader:
    """Return a DataLoader for the given split name ("train", "val", "test")."""
    csv_path = config.SPLITS_DIR / f"{split}.csv"
    transform = get_train_transforms() if split == "train" else get_val_transforms()
    dataset = ICHDataset(csv_path, transform=transform)

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=(split == "train"),
    )
