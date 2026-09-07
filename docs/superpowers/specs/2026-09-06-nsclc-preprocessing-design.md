# NSCLC-Radiomics preprocessing pipeline

## Purpose

Convert all 422 `LUNG1-xxx` DICOM cases under `D:\NSCLC-Radiomics` into
per-patient 2D PNG pairs and cropped 3D NIfTI image/mask pairs. The raw
dataset is read-only: the pipeline must never rename, move, or modify it.

## Inputs

- CT DICOM series discovered recursively in each patient directory.
- DICOM SEG object discovered recursively in the same directory.
- Lung segment: exact label `Lung` (case-insensitive).
- Tumor segment: prefer `GTV-1`; otherwise accept `Neoplasm, Primary`.

Cases without a usable CT, SEG, lung segment, or tumor segment fail only that
case and are recorded in the report. The remaining cases continue.

## Per-case processing

1. Read the CT series with SimpleITK, preserving its spatial metadata.
2. Convert pixels to HU from DICOM rescale parameters, clip to `[-910, 590]`,
   and linearly normalize to 8-bit `[0, 255]`.
3. Decode the lung and tumor DICOM SEG frames and resample both to the CT grid
   with nearest-neighbour interpolation.
4. Compute a tight 3D bounding box from the lung mask of that individual case,
   expand it by a configurable physical margin (default `0 mm`), and clip it
   to the CT grid.
5. Crop CT and tumor mask with that same 3D box. Resize only XY to `256 x 256`
   while preserving Z; update XY spacing so NIfTI retains the cropped physical
   field of view.
6. Write matching axial PNG pairs. Masks are binary 8-bit (`0` or `255`).

## Outputs

```
data/
  processed/
    LUNG-001/
      images/0000.png
      labels/0000.png
  nifti/
    LUNG-001/
      image.nii.gz
      mask.nii.gz
  processing_report.csv
```

Raw case IDs lose only their `1` prefix: `LUNG1-001` becomes `LUNG-001`.
NIfTI image and mask have identical size, origin, direction, and spacing.
Existing case output is skipped unless `--overwrite` is supplied.

## Interface and verification

The CLI exposes raw root, output root, case selection, lung margin,
overwrite behavior, and dry-run mode. The CSV report records status, source
paths, selected SEG labels, per-case crop bbox, geometry, and failures.

Tests will cover case-ID conversion, HU normalization, per-case crop bounds,
geometry-safe resize, and PNG generation using synthetic SimpleITK volumes.
Before the full run, the pipeline is executed on one real case and checked for
PNG dimensions, mask binarity, image/mask alignment, and matching NIfTI
geometry.
