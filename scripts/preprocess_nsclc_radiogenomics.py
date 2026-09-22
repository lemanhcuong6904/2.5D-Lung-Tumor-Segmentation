"""Preprocess NSCLC-Radiogenomics with TotalSegmentator lung masks."""

from __future__ import annotations

import sys
from pathlib import Path

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.preprocess_nsclc import (
    TOTALSEGMENTATOR_FAST_TASK_ID,
    TOTALSEGMENTATOR_LUNG_TASK_ID as DEFAULT_TOTALSEGMENTATOR_LUNG_TASK_ID,
    _write_report,
    output_case_id,
    process_radiogenomics_case,
)

RAW_ROOT = Path(r"D:\NSCLC-Radiogenomics")
DATASET_ROOT = Path("data/nsclc_radiogenomics")
# Set to ``None`` to process every case, or e.g. ``["R01-001"]``.
CASE_IDS: list[str] | None = None
MARGIN_MM = 15.0
# True: crop to the lung bbox along Z (with no Z margin). False: retain every
# source Z slice and crop/margin only in the XY plane.
CROP_Z_TO_LUNG_BBOX = False
HU_WINDOW_LOW = -700
HU_WINDOW_HIGH = 500
OVERWRITE = False
# Set True to regenerate existing processed/ and nifti/ outputs with the new
# post-crop Z spacing. This does not re-run TotalSegmentator while a persisted
# intermediate lung_mask.nii.gz is available.
REPROCESS_EXISTING_OUTPUTS = True
# If True, reuse intermediate/<case>/totalsegmentator/lung_mask.nii.gz. A case
# without that file still invokes TotalSegmentator normally.
REUSE_EXISTING_LUNG_MASK = True
DRY_RUN = False
SHOW_PROGRESS = True
# Keep processing later cases if one case fails. Every result is persisted to
# processing_report.csv immediately for inspection.
STOP_ON_FAILURE = False
# ``fast`` uses TotalSegmentator's official 3 mm total model (Task 297), which
# is sufficient for a lung bounding mask. ``accurate`` uses 1.5 mm Task 291.
LUNG_SEGMENTATION_MODE = "accurate"  # "fast" or "accurate"
# Target spacing after the 3D lung-bbox crop and XY resize. Set None to retain
# the acquired Z spacing.
OUTPUT_Z_SPACING_MM: float | None = 2.0
# Clean tumour masks on the native 3D CT grid before crop/resize.
TUMOR_FILL_HOLES = True
TUMOR_KEEP_LARGEST_COMPONENT = False
# Discard disconnected tumour fragments below this 3D voxel count. Set 0 to
# retain every component, or set TUMOR_KEEP_LARGEST_COMPONENT=True to retain
# only the largest connected component.
TUMOR_MIN_COMPONENT_VOXELS = 100
LUNG_SEGMENTATION_SETTINGS = {
    "fast": {
        "task_id": TOTALSEGMENTATOR_FAST_TASK_ID,
        "trainer": "nnUNetTrainer_4000epochs_NoMirroring",
        "resample_mm": 3.0,
    },
    "accurate": {
        "task_id": DEFAULT_TOTALSEGMENTATOR_LUNG_TASK_ID,
        "trainer": "nnUNetTrainerNoMirroring",
        "resample_mm": 1.5,
    },
}
# API device values: "cuda", "cpu", or "mps". "gpu" remains accepted by
# the shared helper as a backward-compatible alias for "cuda".
TOTALSEGMENTATOR_DEVICE: str | None = "cuda"
# Five lung-lobe ROIs belong to the detailed organs model (Task 291).
TOTALSEGMENTATOR_LUNG_ROIS = (
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
)
# Native DICOM SEG tumour labels accepted by the shared processing code:
# Heart, Tissue, and Segmentation are all tumour contours in this collection.


def main() -> int:
    if MARGIN_MM < 0:
        raise SystemExit("MARGIN_MM must be non-negative")
    if OUTPUT_Z_SPACING_MM is not None and OUTPUT_Z_SPACING_MM <= 0:
        raise SystemExit("OUTPUT_Z_SPACING_MM must be positive or None")
    if LUNG_SEGMENTATION_MODE not in LUNG_SEGMENTATION_SETTINGS:
        raise SystemExit("LUNG_SEGMENTATION_MODE must be 'fast' or 'accurate'")
    lung_settings = LUNG_SEGMENTATION_SETTINGS[LUNG_SEGMENTATION_MODE]
    cases = sorted(path for path in RAW_ROOT.glob("R*") if path.is_dir())
    requested = set(CASE_IDS or [])
    cases = [path for path in cases if not requested or path.name in requested]
    if requested and len(cases) != len(requested):
        raise SystemExit(
            f"requested cases not found: {', '.join(sorted(requested - {path.name for path in cases}))}"
        )
    rows = []
    report_path = DATASET_ROOT / "processing_report.csv"
    case_iterator = (
        tqdm(cases, desc="Preprocessing", unit="case", dynamic_ncols=True)
        if SHOW_PROGRESS
        else cases
    )
    for case_dir in case_iterator:
        if SHOW_PROGRESS:
            case_iterator.set_postfix_str(case_dir.name)
        try:
            row = process_radiogenomics_case(
                case_dir,
                DATASET_ROOT / "processed",
                DATASET_ROOT / "nifti",
                DATASET_ROOT / "intermediate" / "totalsegmentator",
                MARGIN_MM,
                OVERWRITE or REPROCESS_EXISTING_OUTPUTS,
                lung_settings["task_id"],
                TOTALSEGMENTATOR_DEVICE,
                DRY_RUN,
                HU_WINDOW_LOW,
                HU_WINDOW_HIGH,
                TOTALSEGMENTATOR_LUNG_ROIS,
                lung_settings["trainer"],
                lung_settings["resample_mm"],
                OUTPUT_Z_SPACING_MM,
                REUSE_EXISTING_LUNG_MASK,
                tumor_fill_holes=TUMOR_FILL_HOLES,
                tumor_keep_largest_component=TUMOR_KEEP_LARGEST_COMPONENT,
                tumor_min_component_voxels=TUMOR_MIN_COMPONENT_VOXELS,
                crop_z_to_lung_bbox=CROP_Z_TO_LUNG_BBOX,
            )
        except Exception as error:
            row = {
                "raw_case_id": case_dir.name,
                "case_id": output_case_id(case_dir.name),
                "status": "failed",
                "message": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
        message = row.get("message", "")
        if SHOW_PROGRESS:
            tqdm.write(f"{row['status'].upper()}: {case_dir.name} {message}")
        else:
            print(f"{row['status'].upper()}: {case_dir.name} {message}")
        # Persist after every case so an interruption never hides its error.
        _write_report(rows, report_path)
        if row["status"] == "failed" and STOP_ON_FAILURE:
            return 1
    return 0 if all(row["status"] != "failed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
