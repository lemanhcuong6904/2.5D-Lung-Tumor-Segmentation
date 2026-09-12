"""Preprocess NSCLC-Radiogenomics with TotalSegmentator lung masks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.preprocess_nsclc import (
    TOTALSEGMENTATOR_TASK as DEFAULT_TOTALSEGMENTATOR_TASK,
    _write_report,
    output_case_id,
    process_radiogenomics_case,
)

RAW_ROOT = Path(r"D:\NSCLC-Radiogenomics")
DATASET_ROOT = Path("data/nsclc-radiomics/nsclc_radiogenomics")
# Set to ``None`` to process every case, or e.g. ``["R01-001"]``.
CASE_IDS: list[str] | None = None
MARGIN_MM = 15.0
HU_WINDOW_LOW = -700
HU_WINDOW_HIGH = 500
OVERWRITE = False
DRY_RUN = False
TOTALSEGMENTATOR_BIN = "TotalSegmentator"
TOTALSEGMENTATOR_TASK = DEFAULT_TOTALSEGMENTATOR_TASK
TOTALSEGMENTATOR_DEVICE: str | None = "gpu"
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
    cases = sorted(path for path in RAW_ROOT.glob("R*") if path.is_dir())
    requested = set(CASE_IDS or [])
    cases = [path for path in cases if not requested or path.name in requested]
    if requested and len(cases) != len(requested):
        raise SystemExit(
            f"requested cases not found: {', '.join(sorted(requested - {path.name for path in cases}))}"
        )
    rows = []
    for case_dir in cases:
        try:
            row = process_radiogenomics_case(
                case_dir,
                DATASET_ROOT / "processed",
                DATASET_ROOT / "nifti",
                DATASET_ROOT / "intermediate" / "totalsegmentator",
                MARGIN_MM,
                OVERWRITE,
                TOTALSEGMENTATOR_BIN,
                TOTALSEGMENTATOR_TASK,
                TOTALSEGMENTATOR_DEVICE,
                DRY_RUN,
                HU_WINDOW_LOW,
                HU_WINDOW_HIGH,
                TOTALSEGMENTATOR_LUNG_ROIS,
            )
        except Exception as error:
            row = {
                "raw_case_id": case_dir.name,
                "case_id": output_case_id(case_dir.name),
                "status": "failed",
                "message": str(error),
            }
        rows.append(row)
        print(f"{row['status'].upper()}: {case_dir.name} {row.get('message', '')}")
    _write_report(rows, DATASET_ROOT / "processing_report.csv")
    return 0 if all(row["status"] != "failed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
