"""
config.py — Single source of truth for all hyperparameters and paths.
Edit this file to change model, loss, data paths, or training settings.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"

for _dir in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ─── Preprocessing — HU Windows ───────────────────────────────────────────────
# Raw slices are saved as HU-clipped float32 (NOT normalized).
# The dataset constructs a 3-channel input on the fly using the windows below.
# This matches the multi-window approach from Hssayeni et al. (PMC10417300).
#
#   Channel 0 — Brain/ICH window   : tight, highlights blood (50–90 HU)
#   Channel 1 — Broad window       : paper's W.L=80, W.D=180 → more context
#   Channel 2 — Inverted Ch0       : 1 – Ch0, makes hyperattenuating ICH dark
#                                    (negative transform for contrast enhancement)
#
HU_WIN0_MIN = -10  # Brain/ICH window  (W.L=40, W.D=100)  → [-10,  90]
HU_WIN0_MAX = 90
HU_WIN1_MIN = -10  # Broad window      (W.L=80, W.D=180)  → [-10, 170]
HU_WIN1_MAX = 170

# Clip range used when saving raw slices — use the broadest window so no info is lost
HU_SAVE_MIN = HU_WIN1_MIN  # -10
HU_SAVE_MAX = HU_WIN1_MAX  # 170

IMG_SIZE = 512  # Spatial resolution of saved slices (pixels)
SKIP_EMPTY = True  # Drop empty (no-hemorrhage) slices during preprocessing
EMPTY_RATIO = 0.2  # Fraction of empty slices to KEEP (class-balance control)

# ─── Data Splits ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # Must sum to 1.0

# ─── Model ────────────────────────────────────────────────────────────────────
# Options: "attention_unet", "swin_unetr"
MODEL_NAME = "attention_unet"

# Encoder backbone (segmentation_models_pytorch)
# Recommended: "densenet201" (best performer in Hssayeni et al.)
#              "resnet34"    (lighter, fast baseline)
#              "efficientnet-b4" (good accuracy/speed trade-off)
ENCODER = "densenet201"
ENCODER_WEIGHTS = "imagenet"  # None → train from scratch
IN_CHANNELS = 3  # 3-channel multi-window input (see HU windows above)
NUM_CLASSES = 1  # Binary segmentation

# SwinUNETR-specific (used only when MODEL_NAME == "swin_unetr")
SWIN_FEATURE_SIZE = 48

# ─── Loss ─────────────────────────────────────────────────────────────────────
# Options: "bce_dice", "focal_dice", "dice"
LOSS = "bce_dice"
DICE_WEIGHT = 0.5
BCE_WEIGHT = 0.5
FOCAL_GAMMA = 2.0

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 8  # Paper used batch_size=8
EPOCHS = 100
LR = 1e-4  # Paper used lr=0.0001 (Adam); we use AdamW
WEIGHT_DECAY = 1e-5
AMP = True  # Mixed-precision (requires CUDA)
NUM_WORKERS = 4
PIN_MEMORY = True

LR_MIN = 1e-6  # Cosine annealing floor
EARLY_STOP_PATIENCE = 15  # Paper used patience=5; 15 is safer for cosine LR

# ─── Test-Time Augmentation ───────────────────────────────────────────────────
TTA_ENABLED = True  # Horizontal flip TTA at inference

# ─── Threshold ────────────────────────────────────────────────────────────────
THRESHOLD = 0.5  # Sigmoid threshold for binary mask
