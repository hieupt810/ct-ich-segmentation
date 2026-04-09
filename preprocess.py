"""
preprocess.py — Convert raw PhysioNet NIfTI CT scans to 2D normalized slices.

Actual dataset layout (ct-ich, PhysioNet):
  data/raw/
    ct_scans/<NNN>.nii                  — 3-D CT volume (H, W, slices), raw HU
    masks/<NNN>.nii                     — 3-D binary mask volume (H, W, slices)
    hemorrhage_diagnosis_raw_ct.csv     — per-slice ICH subtype labels
    Patient_demographics.csv            — patient-level demographics

Steps:
  1. Read hemorrhage_diagnosis_raw_ct.csv for per-slice labels.
  2. For each patient NIfTI volume:
     a. Clip HU values to [HU_SAVE_MIN, HU_SAVE_MAX] — the broadest window.
        Raw HU-clipped values are saved (NOT normalized) so the dataset can
        apply different normalizations per channel at training time.
     b. Extract each 2-D axial slice and its binary mask.
     c. Save as (H, W) float32 .npy to data/processed/.
  3. Generate stratified train/val/test split CSVs at the patient level.

Why save raw HU instead of normalized [0,1]:
  The multi-window 3-channel approach (Hssayeni et al., PMC10417300) requires
  applying different HU windows to the same slice.  Saving pre-normalized data
  with a single window would make it impossible to recover the other windows.

Patient IDs: 049-130, skipping 059-065 (no raw data for those).
"""

import logging
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Patients known to be missing from the dataset
_MISSING_PATIENTS = set(range(59, 66))  # 059-065 inclusive


# ─── HU Clipping ──────────────────────────────────────────────────────────────


def clip_hu(
    volume: np.ndarray,
    hu_min: float = config.HU_SAVE_MIN,
    hu_max: float = config.HU_SAVE_MAX,
) -> np.ndarray:
    """
    Clip HU values to [hu_min, hu_max] and return as float32.
    Values are NOT normalized to [0,1] here — normalization is done
    per-channel in dataset.py using the specific window for each channel.
    """
    return np.clip(volume, hu_min, hu_max).astype(np.float32)


# ─── Patient Discovery ────────────────────────────────────────────────────────


def list_patients(raw_dir: Path) -> list[int]:
    """Return sorted list of patient IDs that have both CT and mask NIfTI files."""
    ct_dir = raw_dir / "ct_scans"
    mask_dir = raw_dir / "masks"

    patients = []
    for nii_path in sorted(ct_dir.glob("*.nii")):
        pid = int(nii_path.stem)
        if pid in _MISSING_PATIENTS:
            continue
        mask_path = mask_dir / nii_path.name
        if not mask_path.exists():
            log.warning("No mask found for patient %03d — skipping.", pid)
            continue
        patients.append(pid)

    log.info("Found %d patients with CT + mask pairs.", len(patients))
    return patients


# ─── Volume Loading ───────────────────────────────────────────────────────────


def load_nifti_volume(path: Path) -> np.ndarray:
    """
    Load a NIfTI file and return a 3-D array with shape (H, W, num_slices).
    Handles both .nii and .nii.gz.
    """
    nii = nib.load(str(path))
    vol = nii.get_fdata(dtype=np.float32)

    # Squeeze any extra singleton dimensions (some tools add a 4th dim)
    while vol.ndim > 3:
        vol = vol[..., 0]

    return vol


# ─── Per-Patient Processing ───────────────────────────────────────────────────


def process_patient(
    pid: int, raw_dir: Path, out_dir: Path, slice_labels: pd.DataFrame
) -> list[dict]:
    """
    Process all slices for one patient.

    Saves:
      processed/images/<NNN>_s<SSS>.npy  — HU-clipped float32 (H, W)
      processed/masks/<NNN>_s<SSS>.npy   — binary uint8 (H, W)

    Returns list of metadata dicts (one per saved slice).
    """
    filename = f"{pid:03d}.nii"
    ct_path = raw_dir / "ct_scans" / filename
    mask_path = raw_dir / "masks" / filename

    ct_vol = clip_hu(load_nifti_volume(ct_path))  # (H, W, Z) HU-clipped, float32
    mask_vol = load_nifti_volume(mask_path)
    mask_vol = (mask_vol > 0).astype(np.uint8)  # binarize

    num_slices = ct_vol.shape[2]

    # Load per-slice CSV labels (not strictly needed for mask-based labelling,
    # but kept for potential subtype analysis in future work)
    patient_df = (
        slice_labels[slice_labels["PatientNumber"] == pid]
        .sort_values("SliceNumber")
        .reset_index(drop=True)
    )
    if len(patient_df) > 0 and len(patient_df) != num_slices:
        log.warning(
            "Patient %03d: CSV has %d slice labels but NIfTI has %d slices.",
            pid,
            len(patient_df),
            num_slices,
        )

    img_dir = out_dir / "images"
    msk_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for slice_idx in range(num_slices):
        img_slice = ct_vol[..., slice_idx]  # (H, W) HU-clipped float32
        mask_slice = mask_vol[..., slice_idx]  # (H, W) binary uint8

        has_hemorrhage = int(mask_slice.sum() > 0)

        # Randomly drop a fraction of empty slices to reduce class imbalance
        if (
            config.SKIP_EMPTY
            and not has_hemorrhage
            and random.random() > config.EMPTY_RATIO
        ):
            continue

        uid = f"{pid:03d}_s{slice_idx:03d}"
        img_out = img_dir / f"{uid}.npy"
        msk_out = msk_dir / f"{uid}.npy"

        np.save(img_out, img_slice)
        np.save(msk_out, mask_slice)

        records.append(
            {
                "uid": uid,
                "patient_id": f"{pid:03d}",
                "slice_index": slice_idx,
                "image_path": str(img_out.relative_to(config.ROOT_DIR)),
                "mask_path": str(msk_out.relative_to(config.ROOT_DIR)),
                "has_hemorrhage": has_hemorrhage,
            }
        )

    return records


# ─── Split Generation ─────────────────────────────────────────────────────────


def make_splits(metadata: list[dict]) -> None:
    """
    Patient-level stratified train/val/test split to prevent data leakage.
    A patient is labelled positive if ANY of its slices contains hemorrhage.
    """
    df = pd.DataFrame(metadata)

    patient_label = (
        df.groupby("patient_id")["has_hemorrhage"]
        .max()
        .reset_index()
        .rename(columns={"has_hemorrhage": "label"})
    )

    patients = patient_label["patient_id"].tolist()
    labels = patient_label["label"].tolist()

    train_val_pts, test_pts = train_test_split(
        patients,
        test_size=config.TEST_RATIO,
        stratify=labels,
        random_state=config.SEED,
    )

    tv_labels = patient_label[patient_label["patient_id"].isin(train_val_pts)][
        "label"
    ].tolist()
    val_frac = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_pts, val_pts = train_test_split(
        train_val_pts,
        test_size=val_frac,
        stratify=tv_labels,
        random_state=config.SEED,
    )

    for split_name, pts in [("train", train_pts), ("val", val_pts), ("test", test_pts)]:
        split_df = df[df["patient_id"].isin(pts)].reset_index(drop=True)
        out_path = config.SPLITS_DIR / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        n_pos = split_df["has_hemorrhage"].sum()
        log.info(
            "%-5s — %4d slices, %4d with hemorrhage (%d patients)",
            split_name,
            len(split_df),
            n_pos,
            len(pts),
        )


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    raw_dir = config.RAW_DIR
    if not (raw_dir / "ct_scans").exists():
        log.error(
            "ct_scans/ folder not found in %s\n"
            "Place the PhysioNet ct-ich dataset in data/raw/ first.",
            raw_dir,
        )
        return

    label_csv = raw_dir / "hemorrhage_diagnosis_raw_ct.csv"
    slice_labels = pd.read_csv(label_csv)

    patients = list_patients(raw_dir)
    all_records: list[dict] = []

    for pid in tqdm(patients, desc="Processing patients"):
        try:
            records = process_patient(pid, raw_dir, config.PROCESSED_DIR, slice_labels)
            all_records.extend(records)
        except Exception as exc:
            log.warning("Failed patient %03d: %s", pid, exc)

    log.info("Total slices saved: %d", len(all_records))
    make_splits(all_records)
    log.info("Done. Splits saved to %s", config.SPLITS_DIR)


if __name__ == "__main__":
    main()
