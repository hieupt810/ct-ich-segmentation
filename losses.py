"""
losses.py — Loss functions for binary segmentation.

All losses accept raw logits (no sigmoid applied beforehand).

Available:
  DiceLoss        — Pure soft Dice loss
  BCEDiceLoss     — BCE + Dice (recommended default)
  FocalDiceLoss   — Focal + Dice (better for extreme class imbalance)

Select via config.LOSS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# ─── Soft Dice Loss ───────────────────────────────────────────────────────────


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Operates on sigmoid-activated predictions.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten spatial dims
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ─── BCE + Dice ───────────────────────────────────────────────────────────────


class BCEDiceLoss(nn.Module):
    """
    Weighted sum of Binary Cross-Entropy and soft Dice loss.
    pos_weight: tensor scalar to up-weight positive class (hemorrhage).
                Pass None to compute automatically from the batch.
    """

    def __init__(
        self,
        bce_weight: float = config.BCE_WEIGHT,
        dice_weight: float = config.DICE_WEIGHT,
        pos_weight: float | None = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logits.device)
        else:
            # Automatic: ratio of negatives to positives in batch
            n_pos = targets.sum().clamp(min=1)
            n_neg = (1 - targets).sum().clamp(min=1)
            pw = torch.tensor([n_neg / n_pos], device=logits.device)

        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


# ─── Focal + Dice ─────────────────────────────────────────────────────────────


class FocalDiceLoss(nn.Module):
    """
    Focal loss + soft Dice loss.
    Focal loss down-weights easy negatives, focusing on hard positives.
    Recommended when positive ratio < 5 %.
    """

    def __init__(
        self, gamma: float = config.FOCAL_GAMMA, dice_weight: float = config.DICE_WEIGHT
    ):
        super().__init__()
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()
        dice_loss = self.dice(logits, targets)
        return focal_loss + self.dice_weight * dice_loss


# ─── Factory ──────────────────────────────────────────────────────────────────

_LOSS_REGISTRY = {
    "dice": DiceLoss,
    "bce_dice": BCEDiceLoss,
    "focal_dice": FocalDiceLoss,
}


def get_loss(name: str = config.LOSS) -> nn.Module:
    if name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[name]()
