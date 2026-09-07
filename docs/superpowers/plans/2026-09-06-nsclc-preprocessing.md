# NSCLC-Radiomics Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe CLI that converts every NSCLC-Radiomics DICOM case into per-case 256x256 PNG pairs and 3D cropped NIfTI image/mask pairs.

**Architecture:** A single focused Python module discovers CT and SEG objects, converts selected SEG frames to CT geometry, obtains a distinct lung bounding box for each case, then crops/resizes and writes both output formats. Pure geometry and image helpers are tested using synthetic SimpleITK volumes; the CLI reports rather than aborts per-case errors.

**Tech Stack:** Python 3.11, SimpleITK, pydicom, NumPy, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-nsclc-preprocessing-design.md`

## Global Constraints

- Read-only raw source: `D:\NSCLC-Radiomics`; never rename, move, or modify its files.
- Process all 422 `LUNG1-xxx` cases by default, mapping `LUNG1-001` to `LUNG-001`.
- Convert CT to HU, clip `[-910, 590]`, and normalize to uint8 `[0, 255]`.
- Compute the crop box independently for every case from its 3D lung mask.
- Prefer tumor SEG label `GTV-1`; fall back to `Neoplasm, Primary`; select `Lung` for the crop mask.
- Resize only XY to `256 x 256`, preserve Z, and ensure image/mask NIfTI geometry matches exactly.
- Output image/mask PNGs to `data/processed/LUNG-xxx/{images,labels}` and NIfTI to `data/nifti/LUNG-xxx`.
- Skip existing case output unless `--overwrite`; record all outcomes in `data/processing_report.csv`.

---

### Task 1: Pure preprocessing and geometry helpers

**Files:**
- Create: `tests/test_preprocess_nsclc.py`
- Create: `scripts/preprocess_nsclc.py`

**Interfaces:**
- Produces `output_case_id(raw_case_id: str) -> str`.
- Produces `normalize_hu_to_uint8(array: np.ndarray, low: int = -910, high: int = 590) -> np.ndarray`.
- Produces `lung_crop_region(lung_mask: sitk.Image, margin_mm: float = 0.0) -> tuple[tuple[int, int, int], tuple[int, int, int]]` as `(index, size)` in SITK `(x, y, z)` order.
- Produces `crop_and_resize(image: sitk.Image, region: tuple[tuple[int, int, int], tuple[int, int, int]], is_label: bool) -> sitk.Image`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import SimpleITK as sitk

from scripts.preprocess_nsclc import (
    crop_and_resize,
    lung_crop_region,
    normalize_hu_to_uint8,
    output_case_id,
)


def test_output_case_id_removes_only_lung1_prefix():
    assert output_case_id("LUNG1-001") == "LUNG-001"


def test_normalize_hu_clips_and_scales_to_uint8():
    result = normalize_hu_to_uint8(np.array([-1000, -910, -160, 590, 700]))
    np.testing.assert_array_equal(result, np.array([0, 0, 128, 255, 255], dtype=np.uint8))


def test_lung_crop_region_uses_this_cases_mask_and_margin():
    array = np.zeros((4, 10, 12), dtype=np.uint8)
    array[1:3, 3:8, 2:10] = 1
    lung = sitk.GetImageFromArray(array)
    lung.SetSpacing((2.0, 2.0, 3.0))
    assert lung_crop_region(lung, margin_mm=2.0) == ((1, 2, 0), (10, 7, 4))


def test_crop_and_resize_preserves_z_and_matching_geometry():
    image = sitk.GetImageFromArray(np.arange(4 * 10 * 12, dtype=np.uint8).reshape(4, 10, 12))
    image.SetSpacing((1.0, 2.0, 3.0))
    cropped = crop_and_resize(image, ((2, 3, 1), (8, 5, 2)), is_label=False)
    assert cropped.GetSize() == (256, 256, 2)
    assert cropped.GetSpacing() == (8 / 256, 10 / 256, 3.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_preprocess_nsclc.py -v`

Expected: FAIL because `scripts.preprocess_nsclc` does not exist.

- [ ] **Step 3: Implement the minimal helpers**

```python
def output_case_id(raw_case_id: str) -> str:
    return raw_case_id.replace("LUNG1-", "LUNG-", 1)


def normalize_hu_to_uint8(array: np.ndarray, low: int = -910, high: int = 590) -> np.ndarray:
    clipped = np.clip(array, low, high).astype(np.float32)
    return np.rint((clipped - low) * 255.0 / (high - low)).astype(np.uint8)
```

Implement `lung_crop_region` with `sitk.LabelShapeStatisticsImageFilter` and convert physical margin with `ceil(margin_mm / spacing[axis])`; implement `crop_and_resize` with `sitk.RegionOfInterest` followed by a `sitk.ResampleImageFilter` using linear interpolation for CT and nearest-neighbour for masks.

- [ ] **Step 4: Run helper tests to verify they pass**

Run: `python -m pytest tests/test_preprocess_nsclc.py -v`

Expected: PASS (4 tests).

### Task 2: DICOM SEG conversion and case writer

**Files:**
- Modify: `scripts/preprocess_nsclc.py`
- Modify: `tests/test_preprocess_nsclc.py`

**Interfaces:**
- Produces `decode_segments(seg_path: Path, labels: set[str]) -> sitk.Image`.
- Produces `process_case(case_dir: Path, output_root: Path, nifti_root: Path, margin_mm: float, overwrite: bool) -> dict[str, str]`.
- `process_case` writes `images/{index:04d}.png`, `labels/{index:04d}.png`, `image.nii.gz`, and `mask.nii.gz` only after all in-memory processing succeeds.

- [ ] **Step 1: Write the failing writer test**

```python
def test_write_case_outputs_matching_png_pairs_and_nifti(tmp_path):
    image = sitk.GetImageFromArray(np.full((2, 3, 4), 30, dtype=np.uint8))
    mask = sitk.GetImageFromArray(np.array([[[0, 1, 0, 0]] * 3] * 2, dtype=np.uint8))
    image.SetSpacing((1.0, 1.0, 3.0))
    mask.CopyInformation(image)

    write_case_outputs(tmp_path / "processed", tmp_path / "nifti", "LUNG-001", image, mask)

    assert sorted(p.name for p in (tmp_path / "processed/LUNG-001/images").glob("*.png")) == ["0000.png", "0001.png"]
    assert np.unique(np.asarray(Image.open(tmp_path / "processed/LUNG-001/labels/0000.png"))).tolist() == [0, 255]
    assert sitk.ReadImage(str(tmp_path / "nifti/LUNG-001/image.nii.gz")).GetSize() == (4, 3, 2)
```

- [ ] **Step 2: Run the writer test to verify it fails**

Run: `python -m pytest tests/test_preprocess_nsclc.py::test_write_case_outputs_matching_png_pairs_and_nifti -v`

Expected: FAIL because `write_case_outputs` is not defined.

- [ ] **Step 3: Implement SEG decoding and atomic case output**

Decode `PerFrameFunctionalGroupsSequence` frames selected by segment label, build a binary image from their image positions/orientation, and resample it to CT with nearest-neighbour interpolation. Use `PIL.Image.fromarray` for PNG output, writing mask values as `(mask > 0) * 255`. Write to a case-local temporary directory under `data` and rename it only after both PNG directories and NIfTI files have been created.

- [ ] **Step 4: Run the writer test to verify it passes**

Run: `python -m pytest tests/test_preprocess_nsclc.py::test_write_case_outputs_matching_png_pairs_and_nifti -v`

Expected: PASS.

### Task 3: CLI, report, and real-case verification

**Files:**
- Modify: `scripts/preprocess_nsclc.py`
- Modify: `tests/test_preprocess_nsclc.py`
- Create: `requirements.txt`

**Interfaces:**
- Produces `main(argv: Sequence[str] | None = None) -> int`.
- CLI options: `--raw-root`, `--data-root`, `--case`, `--margin-mm`, `--overwrite`, `--dry-run`.
- Writes `processing_report.csv` with one row for every selected case.

- [ ] **Step 1: Write failing CLI/report tests**

```python
def test_build_parser_defaults_to_all_cases_and_default_roots():
    args = build_parser().parse_args([])
    assert args.raw_root == Path(r"D:\NSCLC-Radiomics")
    assert args.data_root == Path("data")
    assert args.case is None
```

- [ ] **Step 2: Run the CLI test to verify it fails**

Run: `python -m pytest tests/test_preprocess_nsclc.py::test_build_parser_defaults_to_all_cases_and_default_roots -v`

Expected: FAIL because `build_parser` is not defined.

- [ ] **Step 3: Implement CLI and report writer**

Iterate `raw_root.glob("LUNG*")` in sorted order. For each case, catch expected processing errors, retain a `status` and `message`, and append a CSV row. Do not delete or overwrite a completed output unless `--overwrite` is explicit. Pin runtime dependencies in `requirements.txt`: `numpy`, `pydicom`, `SimpleITK`, `Pillow`, and `pytest`.

- [ ] **Step 4: Run all tests, then a one-case real validation**

Run: `python -m pytest -v`

Expected: PASS.

Run: `python scripts/preprocess_nsclc.py --case LUNG1-001`

Verify with a short inspection script that every PNG image/mask pair has 256x256 dimensions, masks are binary, pair counts equal NIfTI Z, and NIfTI image/mask metadata matches exactly.

- [ ] **Step 5: Run all cases**

Run: `python scripts/preprocess_nsclc.py`

Verify `data/processing_report.csv` has 422 rows and report successes/failures without making unverified claims about failed cases.
