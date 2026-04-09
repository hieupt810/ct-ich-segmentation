"""
metrics.py — Evaluation metrics for binary segmentation.

All functions work with numpy arrays or torch tensors (converted internally).
Primary metric: Dice coefficient.

Metrics:
  dice_score        — Overlap-based, primary ranking metric
  iou_score         — Jaccard index
  sensitivity       — True positive rate (recall) — critical for hemorrhage
  specificity       — True negative rate
  hausdorff_95      — 95th percentile Hausdorff distance (boundary accuracy)
  volume_similarity — |V_pred - V_gt| / (V_pred + V_gt)
  compute_all       — Returns dict of all metrics for one prediction
"""

import numpy as np
import torch

try:
    from medpy.metric.binary import hd95 as _hd95_fn

    _MEDPY_AVAILABLE = True
except ImportError:
    _MEDPY_AVAILABLE = False


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _to_numpy_binary(x, threshold: float = 0.5) -> np.ndarray:
    """Accept torch tensor (logits or probs) or numpy array → binary numpy."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    return (x > threshold).astype(np.uint8)


def _flatten(pred: np.ndarray, target: np.ndarray):
    return pred.ravel(), target.ravel()


# ─── Metrics ──────────────────────────────────────────────────────────────────


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    p, t = _flatten(pred, target)
    inter = (p * t).sum()
    return float((2.0 * inter + smooth) / (p.sum() + t.sum() + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    p, t = _flatten(pred, target)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return float((inter + smooth) / (union + smooth))


def sensitivity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Recall / True Positive Rate — fraction of hemorrhage correctly detected."""
    p, t = _flatten(pred, target)
    tp = (p * t).sum()
    fn = ((1 - p) * t).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def specificity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """True Negative Rate — fraction of healthy tissue correctly excluded."""
    p, t = _flatten(pred, target)
    tn = ((1 - p) * (1 - t)).sum()
    fp = (p * (1 - t)).sum()
    return float((tn + smooth) / (tn + fp + smooth))


def hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    95th-percentile Hausdorff distance in pixels.
    Returns NaN if medpy is not installed or either mask is empty.
    """
    if not _MEDPY_AVAILABLE:
        return float("nan")
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    try:
        return float(_hd95_fn(pred, target))
    except Exception:
        return float("nan")


def volume_similarity(
    pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6
) -> float:
    """
    Volumetric overlap similarity: 1 - |V_pred - V_gt| / (V_pred + V_gt).
    Returns 1.0 for perfect volume match, 0.0 for complete mismatch.
    """
    vp = pred.sum()
    vt = target.sum()
    return float(1.0 - abs(vp - vt) / (vp + vt + smooth))


def compute_all(logits_or_probs, target, threshold: float = 0.5) -> dict:
    """
    Compute all metrics for a single prediction.

    Args:
        logits_or_probs : (1, H, W) or (H, W) tensor or array (raw logits or probs)
        target          : (1, H, W) or (H, W) tensor or array (binary ground truth)
        threshold       : Decision threshold

    Returns:
        dict with keys: dice, iou, sensitivity, specificity, hd95, vol_sim
    """
    pred = _to_numpy_binary(logits_or_probs, threshold)
    target = _to_numpy_binary(target, threshold=0.5)

    # Remove channel dim if present
    if pred.ndim == 3:
        pred = pred[0]
    if target.ndim == 3:
        target = target[0]

    return {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
        "sensitivity": sensitivity(pred, target),
        "specificity": specificity(pred, target),
        "hd95": hausdorff_95(pred, target),
        "vol_sim": volume_similarity(pred, target),
    }


def aggregate_metrics(records: list[dict]) -> dict:
    """
    Compute mean ± std for each metric across a list of per-sample dicts.
    NaN values (e.g. from empty masks) are excluded from the mean.
    """
    keys = records[0].keys()
    result = {}
    for k in keys:
        vals = [r[k] for r in records if not np.isnan(r[k])]
        result[k] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"),
            "n": len(vals),
        }
    return result
