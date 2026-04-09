"""
model.py — Model factory for ICH binary segmentation.

Supported models:
  "attention_unet" — Attention U-Net via segmentation_models_pytorch (default)
  "swin_unetr"     — Swin Transformer U-Net via MONAI

Default encoder: "densenet201" — best performer in Hssayeni et al. (PMC10417300).
Any smp-compatible encoder works (resnet34, efficientnet-b4, etc.).

Add new architectures by implementing a function and registering it in build_model().
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

import config

# ─── Attention U-Net ──────────────────────────────────────────────────────────


def build_attention_unet(
    encoder: str = config.ENCODER,
    encoder_weights: str = config.ENCODER_WEIGHTS,
    in_channels: int = config.IN_CHANNELS,
    num_classes: int = config.NUM_CLASSES,
) -> nn.Module:
    """
    Attention U-Net using an ImageNet-pretrained encoder.
    Uses segmentation_models_pytorch's UnetPlusPlus with attention decoder,
    which is the closest smp equivalent and consistently outperforms plain U-Net.
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # Raw logits; sigmoid applied in loss/metric
        decoder_attention_type="scse",  # Squeeze-and-excitation channel+spatial attention
    )
    return model


# ─── Swin UNETR ───────────────────────────────────────────────────────────────


def build_swin_unetr(
    img_size: int = config.IMG_SIZE,
    in_channels: int = config.IN_CHANNELS,
    num_classes: int = config.NUM_CLASSES,
    feature_size: int = config.SWIN_FEATURE_SIZE,
) -> nn.Module:
    """
    SwinUNETR for 2-D segmentation via MONAI.
    Note: MONAI's SwinUNETR is 3-D by default; we wrap it for 2-D use.
    For 2-D, we treat HxW as the spatial dims with depth=1.
    """
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as exc:
        raise ImportError(
            "MONAI is required for SwinUNETR. Install with: pip install monai"
        ) from exc

    # MONAI SwinUNETR expects (B, C, D, H, W); we use D=1 for 2-D.
    class SwinUNETR2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.swin = SwinUNETR(
                img_size=(1, img_size, img_size),
                in_channels=in_channels,
                out_channels=num_classes,
                feature_size=feature_size,
                use_checkpoint=True,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, H, W) → (B, C, 1, H, W)
            x = x.unsqueeze(2)
            out = self.swin(x)  # (B, num_classes, 1, H, W)
            return out.squeeze(2)  # (B, num_classes, H, W)

    return SwinUNETR2D()


# ─── Factory ──────────────────────────────────────────────────────────────────

_MODEL_REGISTRY = {
    "attention_unet": build_attention_unet,
    "swin_unetr": build_swin_unetr,
}


def build_model(name: str = config.MODEL_NAME) -> nn.Module:
    """
    Instantiate a model by name.

    Usage:
        model = build_model()                        # uses config.MODEL_NAME
        model = build_model("swin_unetr")
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]()


# ─── Checkpoint Helpers ───────────────────────────────────────────────────────


def save_checkpoint(model: nn.Module, path, epoch: int, metric: float) -> None:
    torch.save(
        {"epoch": epoch, "metric": metric, "state_dict": model.state_dict()}, path
    )


def load_checkpoint(model: nn.Module, path, device: torch.device) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metric", 0.0)
