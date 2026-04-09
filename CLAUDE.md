# CLAUDE.md — Intracranial Hemorrhage Binary Segmentation

## Project Overview

Binary segmentation model for detecting and segmenting intracranial hemorrhage (ICH) in CT scans.
Dataset: **"Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation"** from PhysioNet.
Target: match or exceed **Dice ≥ 0.857, IoU ≥ 0.843** from Hssayeni et al. (PMC10417300).

Preprocessing and augmentation strategy follows **Hssayeni et al. (2023), PMC10417300**.

---

## Environment

**Always activate the conda environment before running any Python or pip command:**

```bash
conda activate ct-segmentation
python ...
pip ...
```

**Setup (first time only):**

```bash
conda create -n ct-segmentation python=3.12 -y
conda activate ct-segmentation
pip install -r requirements.txt
```

---

## Project Structure

```
claude-segmentation/
├── CLAUDE.md                  # This file
├── requirements.txt           # All dependencies
├── config.py                  # Centralized hyperparameters and paths
│
├── data/
│   ├── raw/                   # Downloaded PhysioNet data (DO NOT modify)
│   ├── processed/             # Preprocessed .npy slices (HU-clipped, not normalized)
│   └── splits/                # train.csv / val.csv / test.csv
│
├── dataset.py                 # ICHDataset + multi-window channels + augmentations
├── model.py                   # Model definitions (AttentionUNet, SwinUNETR wrapper)
├── losses.py                  # DiceLoss, BCE+Dice combo, Focal+Dice combo
├── metrics.py                 # Dice, IoU, Hausdorff, sensitivity, specificity
├── train.py                   # Training loop with checkpointing and logging
├── evaluate.py                # Full evaluation on test split, saves predictions
├── predict.py                 # Single-case inference utility (NIfTI input)
├── preprocess.py              # NIfTI → HU-clipped 2D slice extraction
│
├── checkpoints/               # Saved model weights (best_model.pth, last.pth)
├── logs/                      # TensorBoard event files
└── results/                   # Predicted masks, metric CSVs, visual overlays
```

---

## File Responsibilities

| File | Purpose |
|------|---------|
| `config.py` | Single source of truth for all paths, hyperparameters, and flags |
| `preprocess.py` | NIfTI → HU-clipped float32 .npy slices (NOT normalized — multi-window applied at train time) |
| `dataset.py` | `ICHDataset`: builds 3-channel multi-window image on the fly; paper-matched augmentations |
| `model.py` | `AttentionUNet` (DenseNet201 encoder, default), optional `SwinUNETR` |
| `losses.py` | `DiceLoss`, `BCEDiceLoss`, `FocalDiceLoss` — selected via `config.LOSS` |
| `metrics.py` | All evaluation metrics; used identically in training and evaluation |
| `train.py` | Epoch loop, mixed precision, LR scheduler, early stopping, checkpointing |
| `evaluate.py` | Load best checkpoint, compute per-case and aggregate metrics on test set |
| `predict.py` | CLI: given a NIfTI file, output binary masks + overlay PNGs |

---

## Preprocessing Methodology (Hssayeni et al.)

### Why raw HU is saved (not normalized)
`preprocess.py` saves HU-clipped values as float32 **without** normalizing to [0,1].
This allows `dataset.py` to apply three different normalizations to build the 3-channel input.

### Multi-Window 3-Channel Input
Instead of stacking neighboring slices, each slice becomes a 3-channel image:

| Channel | Window | HU Range | Purpose |
|---------|--------|----------|---------|
| Ch0 | Brain/ICH | [-10, 90] | Highlights blood (50–90 HU hyperattenuating) |
| Ch1 | Broad | [-10, 170] | Paper's W.L=80, W.D=180 — more anatomical context |
| Ch2 | Inverted Ch0 | 1 − Ch0 | Negative transform — makes ICH dark, improves gradients |

### Augmentation (training only)
Matches the paper's Table 1:
- Rotations: **15°, 30°, 60°, 90°, 180°, 270°**
- Translations: **±10%** of image width/height
- Horizontal/vertical flips, elastic deformation, brightness/contrast, Gaussian noise

---

## Workflow

### 1. Preprocess

```bash
conda activate ct-segmentation
python preprocess.py
```

Reads NIfTI volumes from `data/raw/ct_scans/` and masks from `data/raw/masks/`.
Clips HU to [-10, 170] (broadest window), extracts 2D axial slices, saves to `data/processed/`.
Generates `data/splits/train.csv`, `val.csv`, `test.csv` (70/15/15, patient-stratified).

### 2. Train

```bash
conda activate ct-segmentation
python train.py
```

Trains with BCE+Dice loss, AdamW optimizer, cosine annealing LR, AMP.
Checkpoints saved to `checkpoints/`. Logs to `logs/` (TensorBoard).

```bash
tensorboard --logdir logs/
```

### 3. Evaluate

```bash
conda activate ct-segmentation
python evaluate.py
```

Loads `checkpoints/best_model.pth`, runs on test split with TTA (horizontal flip),
prints per-case Dice/IoU and aggregate stats, saves CSVs and overlays to `results/`.

### 4. Predict (single case)

```bash
conda activate ct-segmentation
python predict.py --input data/raw/ct_scans/049.nii --output results/case_049/
```

---

## Key Hyperparameters (all in `config.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MODEL_NAME` | `"attention_unet"` | Options: `"attention_unet"`, `"swin_unetr"` |
| `ENCODER` | `"densenet201"` | Best performer in Hssayeni et al.; any smp encoder works |
| `LOSS` | `"bce_dice"` | Options: `"bce_dice"`, `"focal_dice"`, `"dice"` |
| `IMG_SIZE` | `512` | Input resolution |
| `BATCH_SIZE` | `8` | Paper used 8; increase if GPU memory allows |
| `EPOCHS` | `100` | With early stopping (patience=15) |
| `LR` | `1e-4` | AdamW initial LR (paper used Adam 1e-4) |
| `HU_WIN0_MIN/MAX` | `-10 / 90` | Brain/ICH window (Ch0) |
| `HU_WIN1_MIN/MAX` | `-10 / 170` | Broad window — paper's W.L=80, W.D=180 (Ch1) |
| `HU_SAVE_MIN/MAX` | `-10 / 170` | Clip range saved to disk (broadest window) |
| `TTA_ENABLED` | `True` | Horizontal flip at inference |

---

## Multi-Window HU Convention

| Window | Center | Width | HU Range | Use |
|--------|--------|-------|----------|-----|
| Brain/ICH (Ch0) | 40 | 100 | [-10, 90] | Primary ICH detection |
| Broad (Ch1) | 80 | 180 | [-10, 170] | Paper's window, more context |
| Inverted (Ch2) | — | — | 1 − Ch0 | Contrast enhancement |

ICH appears **hyperattenuating** (50–90 HU) relative to normal brain tissue (~37 HU).
The inversion makes ICH appear dark, which can improve gradient flow in certain encoders.

---

## Metrics Reported

- **Dice coefficient** (primary metric) — target ≥ 0.857
- **IoU / Jaccard index** — target ≥ 0.843
- **Sensitivity (recall)** — critical: missing a bleed is dangerous
- **Specificity**
- **Hausdorff Distance 95** (HD95) — boundary accuracy
- **Volume Similarity**

---

## Coding Conventions

- All hyperparameters live in `config.py` — never hardcode values in other files.
- Use `pathlib.Path` for all file paths.
- Use `logging` module (not `print`) in `train.py` and `evaluate.py`.
- Keep model definitions and training logic strictly separate.
- Prefer `segmentation_models_pytorch` (smp) for encoder-decoder architectures.
- TTA (horizontal flip) is applied in `evaluate.py` and `predict.py` by default.
- Random seeds are fixed via `config.SEED` for reproducibility.

---

## Extending the Project

- **New model**: add a factory function in `model.py`, register name in `config.MODEL_NAME` options.
- **New loss**: add class in `losses.py`, register in the `get_loss()` factory.
- **New metric**: add function in `metrics.py`, import in `evaluate.py`.
- **New HU window**: add `HU_WINx_MIN/MAX` in `config.py`, update `build_multiwindow_image()` in `dataset.py`.
- **3D model**: swap `dataset.py` to return volumetric patches; `model.py` to 3D U-Net.

---

## Dependencies (see `requirements.txt`)

Core: `torch`, `torchvision`, `segmentation-models-pytorch`, `monai`,
`albumentations`, `nibabel`, `scikit-learn`, `pandas`, `numpy`,
`tensorboard`, `tqdm`, `matplotlib`, `scipy`, `medpy` (for HD95).

---

## PhysioNet Dataset Notes

- DOI: https://doi.org/10.13026/w377-9n47
- Dataset name: **ct-ich** by Hssayeni et al. (2020)
- 82 patients (75 with NIfTI files); 36 with ICH, 46 without.
- Each patient volume contains ~30 axial slices at 5 mm thickness.
- Patients 059–065 have no raw data and are automatically skipped.

**Actual raw data layout:**
```
data/raw/
  ct_scans/<NNN>.nii                  — 3-D CT volume (H, W, slices), raw HU
  masks/<NNN>.nii                     — 3-D binary mask (H, W, slices)
  hemorrhage_diagnosis_raw_ct.csv     — per-slice ICH subtype labels
  Patient_demographics.csv            — age, gender, diagnosis per patient
```

- Volumes are NIfTI format; slices are along axis 2: `vol[:, :, i]`.
- `preprocess.py` clips HU to [-10, 170] and saves **unnormalized** float32 arrays.
- `dataset.py` applies the 3-channel multi-window normalization at training time.
- `predict.py` accepts a NIfTI file directly: `--input data/raw/ct_scans/049.nii`.
- Cite using PhysioNet citation guidelines.

## Reference

Hssayeni, M. D., et al. (2023). *Automatic ICH Segmentation Using Deep Learning*.
PMC10417300. https://pmc.ncbi.nlm.nih.gov/articles/PMC10417300/
