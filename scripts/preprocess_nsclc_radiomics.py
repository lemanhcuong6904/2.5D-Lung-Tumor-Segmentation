"""Preprocess NSCLC-Radiomics DICOM SEG/RTSTRUCT labels."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.preprocess_nsclc import _write_report, output_case_id, process_case

RAW_ROOT = Path(r"D:\NSCLC-Radiomics")
DATASET_ROOT = Path("data/nsclc-radiomics")
# Set to ``None`` to process every case, or e.g. ``["LUNG1-001"]``.
CASE_IDS: list[str] | None = None
MARGIN_MM = 15.0
# True: crop to the lung bbox along Z (with no Z margin). False: retain every
# source Z slice and crop/margin only in the XY plane.
CROP_Z_TO_LUNG_BBOX = False
HU_WINDOW_LOW = -700
HU_WINDOW_HIGH = 500
OVERWRITE = True
DRY_RUN = False


def main() -> int:
    if MARGIN_MM < 0:
        raise SystemExit("MARGIN_MM must be non-negative")
    cases = sorted(path for path in RAW_ROOT.glob("LUNG*") if path.is_dir())
    requested = set(CASE_IDS or [])
    cases = [path for path in cases if not requested or path.name in requested]
    if requested and len(cases) != len(requested):
        raise SystemExit(
            f"requested cases not found: {', '.join(sorted(requested - {path.name for path in cases}))}"
        )
    rows = []
    for case_dir in cases:
        try:
            row = process_case(
                case_dir,
                DATASET_ROOT / "processed",
                DATASET_ROOT / "nifti",
                MARGIN_MM,
                OVERWRITE,
                DRY_RUN,
                HU_WINDOW_LOW,
                HU_WINDOW_HIGH,
                CROP_Z_TO_LUNG_BBOX,
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
