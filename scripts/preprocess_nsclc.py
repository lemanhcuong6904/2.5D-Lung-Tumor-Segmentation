"""Preprocess NSCLC-Radiomics DICOM cases into PNG and NIfTI datasets."""

from __future__ import annotations

import csv
import shutil
import uuid
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image, ImageDraw

# -----------------------------------------------------------------------------
# Run configuration
# -----------------------------------------------------------------------------
# Set CASE_IDS to None to process every LUNG* case, or provide raw case IDs
# such as ["LUNG1-001", "LUNG1-002"].
RAW_ROOT = Path(r"D:\NSCLC-Radiomics")
DATA_ROOT = Path("data")
CASE_IDS: list[str] | None = None
MARGIN_PX = 5
OVERWRITE = True
DRY_RUN = False
HU_WINDOW_LOW = -700
HU_WINDOW_HIGH = 500

# Source DICOM SEG labels. Lung is used only to define the crop; the saved
# label contains tumor only, binarized to 0/1.
LUNG_SEGMENT_LABELS = {"lung"}
TUMOR_SEGMENT_LABELS = {"neoplasm, primary"}
# Turn this on only when source labels contain variants such as "Lung Left".
SEGMENT_LABEL_CONTAINS_MATCH = False
# Remove disconnected lung-label fragments smaller than this before computing
# the crop bbox. This is measured in 3D voxels, not pixels per slice.
LUNG_MIN_COMPONENT_VOXELS = 1_000


def output_case_id(raw_case_id: str) -> str:
    """Map the raw TCIA identifier to the project's output identifier."""
    return raw_case_id.replace("LUNG1-", "LUNG-", 1)


def normalize_hu_to_uint8(
    array: np.ndarray, low: int = -910, high: int = 590
) -> np.ndarray:
    """Clip HU values and linearly scale them to the uint8 range."""
    if low >= high:
        raise ValueError("low must be less than high")
    clipped = np.clip(array, low, high).astype(np.float32)
    return np.rint((clipped - low) * 255.0 / (high - low)).astype(np.uint8)


def lung_crop_region(
    lung_mask: sitk.Image, margin_px: int = 0
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return the per-case lung bbox, expanded by a voxel/pixel margin."""
    if margin_px < 0:
        raise ValueError("margin_px must be non-negative")

    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(sitk.Cast(lung_mask > 0, sitk.sitkUInt8))
    labels = statistics.GetLabels()
    if not labels:
        raise ValueError("lung mask is empty")

    index = list(statistics.GetBoundingBox(labels[0])[:3])
    size = list(statistics.GetBoundingBox(labels[0])[3:])
    image_size = lung_mask.GetSize()
    square_side = max(size[0], size[1])
    if square_side > min(image_size[0], image_size[1]):
        raise ValueError("lung bbox cannot fit into an in-plane square")
    for axis in (0, 1):
        extra_before = (square_side - size[axis]) // 2
        start = index[axis] - extra_before
        start = min(max(start, 0), image_size[axis] - square_side)
        index[axis] = start
        size[axis] = square_side

    for axis in range(3):
        start = max(0, index[axis] - margin_px)
        stop = min(image_size[axis], index[axis] + size[axis] + margin_px)
        index[axis] = start
        size[axis] = stop - start

    return tuple(index), tuple(size)


def crop_and_resize(
    image: sitk.Image,
    region: tuple[tuple[int, int, int], tuple[int, int, int]],
    is_label: bool,
) -> sitk.Image:
    """Crop an image then resize only XY to 256 pixels, retaining Z."""
    index, size = region
    cropped = sitk.RegionOfInterest(image, size=list(size), index=list(index))
    output_size = (256, 256, size[2])
    input_spacing = cropped.GetSpacing()
    output_spacing = (
        input_spacing[0] * size[0] / output_size[0],
        input_spacing[1] * size[1] / output_size[1],
        input_spacing[2],
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputOrigin(cropped.GetOrigin())
    resampler.SetOutputDirection(cropped.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(cropped)


def write_case_outputs(
    processed_root: Path,
    nifti_root: Path,
    case_id: str,
    image: sitk.Image,
    mask: sitk.Image,
    overwrite: bool = False,
) -> None:
    """Write paired PNGs and NIfTI volumes, without overwriting a case."""
    if image.GetSize() != mask.GetSize() or image.GetSpacing() != mask.GetSpacing():
        raise ValueError("image and mask geometry must match")

    final_processed = processed_root / case_id
    final_nifti = nifti_root / case_id
    if (final_processed.exists() or final_nifti.exists()) and not overwrite:
        raise FileExistsError(f"output already exists for {case_id}")

    processed_root.mkdir(parents=True, exist_ok=True)
    nifti_root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    tmp_processed = processed_root / f".{case_id}.{token}.tmp"
    tmp_nifti = nifti_root / f".{case_id}.{token}.tmp"
    backup_processed = processed_root / f".{case_id}.{token}.backup"
    backup_nifti = nifti_root / f".{case_id}.{token}.backup"
    processed_replaced = False
    nifti_replaced = False

    try:
        images_dir = tmp_processed / "images"
        labels_dir = tmp_processed / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        image_array = sitk.GetArrayFromImage(image).astype(np.uint8)
        mask_array = (sitk.GetArrayFromImage(mask) > 0).astype(np.uint8) * 255
        for slice_index, (image_slice, mask_slice) in enumerate(
            zip(image_array, mask_array, strict=True)
        ):
            Image.fromarray(image_slice).save(images_dir / f"{slice_index:04d}.png")
            Image.fromarray(mask_slice).save(labels_dir / f"{slice_index:04d}.png")

        tmp_nifti.mkdir(parents=True)
        sitk.WriteImage(image, str(tmp_nifti / "image.nii.gz"))
        sitk.WriteImage(
            sitk.Cast(mask > 0, sitk.sitkUInt8), str(tmp_nifti / "mask.nii.gz")
        )
        if final_processed.exists():
            final_processed.replace(backup_processed)
        if final_nifti.exists():
            final_nifti.replace(backup_nifti)
        tmp_processed.replace(final_processed)
        processed_replaced = True
        tmp_nifti.replace(final_nifti)
        nifti_replaced = True
        shutil.rmtree(backup_processed, ignore_errors=True)
        shutil.rmtree(backup_nifti, ignore_errors=True)
    except Exception:
        if processed_replaced:
            shutil.rmtree(final_processed, ignore_errors=True)
        if nifti_replaced:
            shutil.rmtree(final_nifti, ignore_errors=True)
        if backup_processed.exists():
            backup_processed.replace(final_processed)
        if backup_nifti.exists():
            backup_nifti.replace(final_nifti)
        shutil.rmtree(tmp_processed, ignore_errors=True)
        shutil.rmtree(tmp_nifti, ignore_errors=True)
        raise


def _dicom_headers(case_dir: Path) -> Iterable[tuple[Path, pydicom.Dataset]]:
    for path in case_dir.rglob("*.dcm"):
        try:
            yield path, pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue


def sort_slice_positions(
    positions: Sequence[Sequence[float]], orientation: Sequence[float]
) -> list[int]:
    """Return indices sorted along the DICOM slice normal."""
    row = np.asarray(orientation[:3], dtype=np.float64)
    column = np.asarray(orientation[3:6], dtype=np.float64)
    normal = np.cross(row, column)
    return sorted(
        range(len(positions)),
        key=lambda index: float(np.dot(np.asarray(positions[index]), normal)),
    )


def _load_ct(case_dir: Path) -> tuple[sitk.Image, list[Path]]:
    series: dict[str, list[Path]] = {}
    for path, dataset in _dicom_headers(case_dir):
        if getattr(dataset, "Modality", "") == "CT":
            series.setdefault(str(dataset.SeriesInstanceUID), []).append(path)
    if not series:
        raise ValueError("no CT DICOM series found")

    paths = max(series.values(), key=len)
    headers = [
        pydicom.dcmread(path, stop_before_pixels=True, force=True) for path in paths
    ]
    orientation = [float(value) for value in headers[0].ImageOrientationPatient]
    positions = [
        [float(value) for value in header.ImagePositionPatient] for header in headers
    ]
    paths = [paths[index] for index in sort_slice_positions(positions, orientation)]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in paths])
    return reader.Execute(), paths


def _find_seg(case_dir: Path) -> Path:
    for path, dataset in _dicom_headers(case_dir):
        if getattr(dataset, "Modality", "") == "SEG":
            return path
    raise ValueError("no DICOM SEG object found")


def _find_rtstruct(case_dir: Path) -> Path:
    for path, dataset in _dicom_headers(case_dir):
        if getattr(dataset, "Modality", "") == "RTSTRUCT":
            return path
    raise ValueError("no DICOM RTSTRUCT object found")


def _match_segment_label(segment_label: str, expected_label: str) -> bool:
    """Use the same exact/contains SEG-label matching as the reference script."""
    actual = segment_label.strip().casefold()
    expected = expected_label.strip().casefold()
    return expected in actual if SEGMENT_LABEL_CONTAINS_MATCH else actual == expected


def _segment_numbers(dataset: pydicom.Dataset, target_labels: set[str]) -> list[int]:
    return [
        int(segment.SegmentNumber)
        for segment in dataset.SegmentSequence
        if any(
            _match_segment_label(str(getattr(segment, "SegmentLabel", "")), label)
            for label in target_labels
        )
    ]


def _decode_segments_to_ct(
    seg_path: Path, target_labels: set[str], ct_image: sitk.Image
) -> tuple[sitk.Image, list[str]]:
    dataset = pydicom.dcmread(seg_path, force=True)
    if getattr(dataset, "Modality", "") != "SEG":
        raise ValueError(f"not a SEG file: {seg_path}")

    segment_numbers = _segment_numbers(dataset, target_labels)
    if not segment_numbers:
        raise ValueError(f"SEG labels not found: {sorted(target_labels)}")
    segment_labels = [
        str(segment.SegmentLabel)
        for segment in dataset.SegmentSequence
        if int(segment.SegmentNumber) in segment_numbers
    ]

    mask = np.zeros(sitk.GetArrayFromImage(ct_image).shape, dtype=np.uint8)
    ct_size = ct_image.GetSize()
    frames = dataset.pixel_array
    for frame, groups in zip(
        frames, dataset.PerFrameFunctionalGroupsSequence, strict=True
    ):
        segment_number = int(
            groups.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        )
        if segment_number not in segment_numbers:
            continue
        if frame.shape != (ct_size[1], ct_size[0]):
            raise ValueError("SEG in-plane size does not match CT")
        position = tuple(
            float(value)
            for value in groups.PlanePositionSequence[0].ImagePositionPatient
        )
        try:
            x, y, z = ct_image.TransformPhysicalPointToIndex(position)
        except RuntimeError as error:
            raise ValueError("SEG frame lies outside CT geometry") from error
        if not (0 <= z < ct_size[2] and x == 0 and y == 0):
            raise ValueError("SEG frame is not aligned to the CT grid")
        mask[z] = np.maximum(mask[z], (frame > 0).astype(np.uint8))

    output = sitk.GetImageFromArray(mask)
    output.CopyInformation(ct_image)
    return output, segment_labels


def derive_lung_mask_from_ct(ct_image: sitk.Image) -> sitk.Image:
    """Derive a two-lung air mask when a DICOM SEG does not annotate lungs."""
    hu = sitk.GetArrayFromImage(ct_image)
    air = (hu >= -1000) & (hu <= -320)
    connected = sitk.ConnectedComponent(sitk.GetImageFromArray(air.astype(np.uint8)))
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(connected)
    size_x, size_y, _ = ct_image.GetSize()
    candidates = []
    for label in statistics.GetLabels():
        x, y, _, width, height, _ = statistics.GetBoundingBox(label)
        touches_image_edge = (
            x == 0 or y == 0 or x + width == size_x or y + height == size_y
        )
        if not touches_image_edge:
            candidates.append((statistics.GetNumberOfPixels(label), label))
    selected = [label for _, label in sorted(candidates, reverse=True)[:2]]
    if not selected:
        raise ValueError("could not derive internal lung components from CT")
    derived = sitk.GetImageFromArray(
        np.isin(sitk.GetArrayFromImage(connected), selected).astype(np.uint8)
    )
    derived.CopyInformation(ct_image)
    return derived


def remove_small_lung_components(
    lung_mask: sitk.Image, minimum_voxels: int
) -> tuple[sitk.Image, int]:
    """Remove small disconnected 3D fragments from a lung mask."""
    if minimum_voxels < 1:
        raise ValueError("minimum_voxels must be at least 1")

    connected = sitk.ConnectedComponent(sitk.Cast(lung_mask > 0, sitk.sitkUInt8))
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(connected)
    kept_labels = [
        label
        for label in statistics.GetLabels()
        if statistics.GetNumberOfPixels(label) >= minimum_voxels
    ]
    if not kept_labels:
        raise ValueError("lung mask has no component above the minimum size")

    cleaned_array = np.isin(
        sitk.GetArrayFromImage(connected), kept_labels
    ).astype(np.uint8)
    cleaned = sitk.GetImageFromArray(cleaned_array)
    cleaned.CopyInformation(lung_mask)
    return cleaned, len(statistics.GetLabels()) - len(kept_labels)


def _decode_rtstruct_tumor_to_ct(
    rtstruct_path: Path, ct_image: sitk.Image
) -> tuple[sitk.Image, list[str]]:
    """Rasterize GTV contours from an RTSTRUCT into the CT voxel grid."""
    dataset = pydicom.dcmread(rtstruct_path, force=True)
    roi_names = {
        int(roi.ROINumber): str(roi.ROIName) for roi in dataset.StructureSetROISequence
    }
    exact = {number for number, name in roi_names.items() if name.casefold() == "gtv-1"}
    selected = exact or {
        number
        for number, name in roi_names.items()
        if name.casefold().startswith("gtv")
    }
    if not selected:
        raise ValueError("RTSTRUCT does not contain a GTV contour")

    ct_size = ct_image.GetSize()
    mask = np.zeros(sitk.GetArrayFromImage(ct_image).shape, dtype=np.uint8)
    for roi_contour in dataset.ROIContourSequence:
        if int(roi_contour.ReferencedROINumber) not in selected:
            continue
        for contour in getattr(roi_contour, "ContourSequence", []):
            points = np.asarray(contour.ContourData, dtype=np.float64).reshape(-1, 3)
            indices = [
                ct_image.TransformPhysicalPointToContinuousIndex(tuple(point))
                for point in points
            ]
            z_values = [point[2] for point in indices]
            z_index = int(round(float(np.mean(z_values))))
            if not 0 <= z_index < ct_size[2]:
                continue
            polygon = [(point[0], point[1]) for point in indices]
            raster = Image.new("L", (ct_size[0], ct_size[1]), 0)
            ImageDraw.Draw(raster).polygon(polygon, outline=1, fill=1)
            mask[z_index] = np.maximum(
                mask[z_index], np.asarray(raster, dtype=np.uint8)
            )

    if not mask.any():
        raise ValueError("RTSTRUCT GTV contours did not rasterize onto CT")
    output = sitk.GetImageFromArray(mask)
    output.CopyInformation(ct_image)
    return output, [roi_names[number] for number in sorted(selected)]


def _normalize_ct_image(
    ct_image: sitk.Image, hu_window_low: int, hu_window_high: int
) -> sitk.Image:
    normalized = sitk.GetImageFromArray(
        normalize_hu_to_uint8(
            sitk.GetArrayFromImage(ct_image), hu_window_low, hu_window_high
        )
    )
    normalized.CopyInformation(ct_image)
    return normalized


def process_case(
    case_dir: Path,
    processed_root: Path,
    nifti_root: Path,
    margin_px: int,
    overwrite: bool,
    dry_run: bool = False,
) -> dict[str, str]:
    case_id = output_case_id(case_dir.name)
    result = {
        "raw_case_id": case_dir.name,
        "case_id": case_id,
        "status": "ok",
        "message": "",
    }
    output_exists = (processed_root / case_id).exists() or (
        nifti_root / case_id
    ).exists()
    if output_exists and not overwrite:
        result.update(status="skipped", message="output already exists")
        return result

    ct_image, ct_paths = _load_ct(case_dir)
    try:
        seg_path = _find_seg(case_dir)
    except ValueError:
        seg_path = None

    lung_labels: list[str]
    if seg_path is not None:
        try:
            lung_mask, lung_labels = _decode_segments_to_ct(
                seg_path, LUNG_SEGMENT_LABELS, ct_image
            )
        except ValueError:
            lung_mask = derive_lung_mask_from_ct(ct_image)
            lung_labels = ["CT-derived lung fallback"]
    else:
        lung_mask = derive_lung_mask_from_ct(ct_image)
        lung_labels = ["CT-derived lung fallback"]
    lung_mask, removed_lung_components = remove_small_lung_components(
        lung_mask, LUNG_MIN_COMPONENT_VOXELS
    )

    if seg_path is not None:
        try:
            tumor_mask, selected_tumor_labels = _decode_segments_to_ct(
                seg_path, TUMOR_SEGMENT_LABELS, ct_image
            )
        except ValueError:
            rtstruct_path = _find_rtstruct(case_dir)
            tumor_mask, selected_tumor_labels = _decode_rtstruct_tumor_to_ct(
                rtstruct_path, ct_image
            )
    else:
        rtstruct_path = _find_rtstruct(case_dir)
        tumor_mask, selected_tumor_labels = _decode_rtstruct_tumor_to_ct(
            rtstruct_path, ct_image
        )
    region = lung_crop_region(lung_mask, margin_px)
    image_final = crop_and_resize(
        _normalize_ct_image(ct_image, HU_WINDOW_LOW, HU_WINDOW_HIGH),
        region,
        is_label=False,
    )
    mask_final = crop_and_resize(tumor_mask, region, is_label=True)

    result.update(
        ct_slices=str(len(ct_paths)),
        seg_path=str(seg_path or ""),
        lung_labels=";".join(lung_labels),
        removed_lung_components=str(removed_lung_components),
        tumor_labels=";".join(selected_tumor_labels),
        crop_index=str(region[0]),
        crop_size=str(region[1]),
        final_size=str(image_final.GetSize()),
        final_spacing=str(image_final.GetSpacing()),
    )
    if not dry_run:
        write_case_outputs(
            processed_root,
            nifti_root,
            case_id,
            image_final,
            mask_final,
            overwrite=overwrite,
        )
    return result


def _write_report(rows: list[dict[str, str]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with report_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if MARGIN_PX < 0:
        raise SystemExit("MARGIN_PX must be non-negative")
    all_cases = sorted(path for path in RAW_ROOT.glob("LUNG*") if path.is_dir())
    requested = set(CASE_IDS or [])
    cases = [path for path in all_cases if not requested or path.name in requested]
    if requested and len(cases) != len(requested):
        missing = sorted(requested - {path.name for path in cases})
        raise SystemExit(f"requested cases not found: {', '.join(missing)}")

    rows: list[dict[str, str]] = []
    for case_dir in cases:
        try:
            row = process_case(
                case_dir,
                DATA_ROOT / "processed",
                DATA_ROOT / "nifti",
                MARGIN_PX,
                OVERWRITE,
                DRY_RUN,
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
    _write_report(rows, DATA_ROOT / "processing_report.csv")
    return 0 if all(row["status"] != "failed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
