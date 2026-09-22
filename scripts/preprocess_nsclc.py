"""Shared image, DICOM, and output utilities for NSCLC preprocessing.

Use ``preprocess_nsclc_radiomics.py`` or
``preprocess_nsclc_radiogenomics.py`` as the executable entry point.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import time
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
HU_WINDOW_LOW = -700
HU_WINDOW_HIGH = 500
# The TotalSegmentator ``total`` task is split into five nnU-Net models.
# Task 291 is the only part containing the five lung lobes used here.
TOTALSEGMENTATOR_LUNG_TASK_ID = 291
TOTALSEGMENTATOR_FAST_TASK_ID = 297

# Source DICOM SEG labels. Lung is used only to define the crop; the saved
# label contains tumor only, binarized to 0/1.
LUNG_SEGMENT_LABELS = {"lung"}
TUMOR_SEGMENT_LABELS = {"neoplasm, primary"}
# NSCLC-Radiogenomics uses inconsistent SEG label text for the same tumour
# contour. These are tumour masks, not cardiac/normal-tissue annotations.
RADIOGENOMICS_TUMOR_SEGMENT_LABELS = {"heart", "tissue", "segmentation"}
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
    lung_mask: sitk.Image,
    margin_px: int = 0,
    *,
    margin_mm: float | None = None,
    crop_z_to_lung_bbox: bool = True,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return a square XY lung bbox, with optional Z-bbox cropping.

    The margin is always in-plane only. With ``crop_z_to_lung_bbox=False``,
    the returned region retains every source CT slice along Z.
    """
    if margin_mm is not None:
        if margin_mm < 0:
            raise ValueError("margin_mm must be non-negative")
        # A physical margin cannot be represented by one universal pixel
        # count on anisotropic CT.  Use the most conservative in-plane count.
        margin_px = int(np.ceil(margin_mm / min(lung_mask.GetSpacing()[:2])))
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

    # Preserve the lung-mask Z extent exactly.  Margin is deliberately an
    # in-plane context margin, not an extension to superior/inferior slices.
    for axis in (0, 1):
        start = max(0, index[axis] - margin_px)
        stop = min(image_size[axis], index[axis] + size[axis] + margin_px)
        index[axis] = start
        size[axis] = stop - start

    if not crop_z_to_lung_bbox:
        index[2] = 0
        size[2] = image_size[2]

    return tuple(index), tuple(size)


def crop_and_resize(
    image: sitk.Image,
    region: tuple[tuple[int, int, int], tuple[int, int, int]],
    is_label: bool,
    *,
    output_z_spacing_mm: float | None = None,
) -> sitk.Image:
    """Crop an image, resize XY to 256 pixels, and optionally resample Z."""
    index, size = region
    cropped = sitk.RegionOfInterest(image, size=list(size), index=list(index))
    input_spacing = cropped.GetSpacing()
    if output_z_spacing_mm is not None and output_z_spacing_mm <= 0:
        raise ValueError("output_z_spacing_mm must be positive when provided")

    # Use the physical distance between the first and last slice centres so
    # the resampled volume covers the cropped Z extent without silently
    # truncating its superior or inferior end.  The requested spacing remains
    # exact; a small endpoint difference is unavoidable for discrete voxels.
    output_z_size = size[2]
    output_z_spacing = input_spacing[2]
    if output_z_spacing_mm is not None:
        output_z_spacing = float(output_z_spacing_mm)
        output_z_size = max(
            1,
            int(round((size[2] - 1) * input_spacing[2] / output_z_spacing)) + 1,
        )

    output_size = (256, 256, output_z_size)
    output_spacing = (
        input_spacing[0] * size[0] / output_size[0],
        input_spacing[1] * size[1] / output_size[1],
        output_z_spacing,
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
            _replace_path_with_retry(final_processed, backup_processed)
        if final_nifti.exists():
            _replace_path_with_retry(final_nifti, backup_nifti)
        _replace_path_with_retry(tmp_processed, final_processed)
        processed_replaced = True
        _replace_path_with_retry(tmp_nifti, final_nifti)
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


def _replace_path_with_retry(
    source: Path, destination: Path, *, attempts: int = 5, delay_seconds: float = 2.0
) -> None:
    """Rename an output path, retrying transient Windows access/share locks."""
    for attempt in range(1, attempts + 1):
        try:
            source.replace(destination)
            return
        except PermissionError as error:
            # WinError 5 is ``Access is denied`` and 32 is a sharing violation.
            # Both are commonly caused by Explorer, an image viewer, antivirus,
            # or another Python process briefly inspecting a just-written file.
            if getattr(error, "winerror", None) not in {5, 32} or attempt == attempts:
                raise
            print(
                f"Windows is temporarily locking {source.name}; retrying output "
                f"replacement ({attempt}/{attempts})..."
            )
            time.sleep(delay_seconds)


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
    shared_groups = getattr(dataset, "SharedFunctionalGroupsSequence", [])
    shared_segment_identification = (
        getattr(shared_groups[0], "SegmentIdentificationSequence", None)
        if shared_groups
        else None
    )
    for frame, groups in zip(
        frames, dataset.PerFrameFunctionalGroupsSequence, strict=True
    ):
        # Some valid SEG objects store this value in the shared functional
        # group because every frame belongs to the same segment.
        segment_identification = getattr(
            groups, "SegmentIdentificationSequence", None
        ) or shared_segment_identification
        if not segment_identification:
            raise ValueError("SEG frame has no segment identification")
        segment_number = int(segment_identification[0].ReferencedSegmentNumber)
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


def clean_tumor_mask(
    tumor_mask: sitk.Image,
    *,
    closing_radius_mm: float = 0.0,
    fill_holes: bool = True,
    keep_largest_component: bool = True,
    minimum_component_voxels: int = 0,
    dilation_radius_mm: float = 0.0,
) -> tuple[sitk.Image, dict[str, int]]:
    """Conservatively clean a binary tumour mask on its native CT grid.

    Every radius is converted independently along X/Y/Z, so the morphology
    remains in physical millimetres for anisotropic CT spacing.  Dilation is
    deliberately the final operation: only the retained lesion is allowed to
    grow or merge with another structure.
    """
    if closing_radius_mm < 0 or dilation_radius_mm < 0:
        raise ValueError("tumour morphology radii must be non-negative")
    if minimum_component_voxels < 0:
        raise ValueError("minimum_component_voxels must be non-negative")
    cleaned = sitk.Cast(tumor_mask > 0, sitk.sitkUInt8)
    original_voxels = int(sitk.GetArrayViewFromImage(cleaned).sum())
    spacing = cleaned.GetSpacing()

    def physical_radius(radius_mm: float) -> list[int]:
        return [int(np.ceil(radius_mm / axis_spacing)) for axis_spacing in spacing]

    if closing_radius_mm > 0:
        cleaned = sitk.BinaryMorphologicalClosing(
            cleaned,
            kernelRadius=physical_radius(closing_radius_mm),
            kernelType=sitk.sitkBall,
        )
    if fill_holes:
        cleaned = sitk.BinaryFillhole(cleaned, fullyConnected=True)

    component_filter = sitk.ConnectedComponentImageFilter()
    component_filter.FullyConnectedOn()
    connected = component_filter.Execute(cleaned)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(connected)
    component_count_before = len(statistics.GetLabels())
    if keep_largest_component and statistics.GetLabels():
        kept_labels = [
            max(statistics.GetLabels(), key=statistics.GetNumberOfPixels)
        ]
    elif minimum_component_voxels:
        kept_labels = [
            label
            for label in statistics.GetLabels()
            if statistics.GetNumberOfPixels(label) >= minimum_component_voxels
        ]
    else:
        kept_labels = list(statistics.GetLabels())

    if len(kept_labels) != component_count_before:
        array = np.isin(
            sitk.GetArrayViewFromImage(connected), kept_labels
        ).astype(np.uint8)
        cleaned = sitk.GetImageFromArray(array)
        cleaned.CopyInformation(tumor_mask)

    if dilation_radius_mm > 0 and kept_labels:
        cleaned = sitk.BinaryDilate(
            cleaned,
            kernelRadius=physical_radius(dilation_radius_mm),
            kernelType=sitk.sitkBall,
        )
    cleaned = sitk.Cast(cleaned > 0, sitk.sitkUInt8)
    return cleaned, {
        "tumor_original_voxels": original_voxels,
        "tumor_cleaned_voxels": int(sitk.GetArrayViewFromImage(cleaned).sum()),
        "tumor_components_before_cleanup": component_count_before,
        "tumor_components_removed": component_count_before - len(kept_labels),
        "tumor_kept_largest_component": int(keep_largest_component),
        "tumor_min_component_voxels": minimum_component_voxels,
    }


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


def clip_hu_image(ct_image: sitk.Image, low: int, high: int) -> sitk.Image:
    """Clip a CT in HU while preserving its complete physical geometry.

    This is deliberately different from ``_normalize_ct_image``: the
    TotalSegmentator model must receive CT intensities in HU, not the uint8
    PNG representation used by the training dataset.
    """
    if low >= high:
        raise ValueError("HU lower bound must be less than upper bound")
    clipped = np.clip(sitk.GetArrayFromImage(ct_image), low, high).astype(np.int16)
    output = sitk.GetImageFromArray(clipped)
    output.CopyInformation(ct_image)
    return output


def _read_totalsegmentator_lung_mask(
    output_dir: Path, reference: sitk.Image
) -> sitk.Image:
    """Combine TotalSegmentator lung-lobe masks into a CT-grid lung mask."""
    candidates = sorted(
        path for path in output_dir.glob("*.nii*")
        if path.name.casefold().startswith("lung_")
    )
    if not candidates:
        available = ", ".join(path.name for path in sorted(output_dir.glob("*.nii*")))
        raise ValueError(
            "TotalSegmentator produced no lung-lobe mask"
            + (f" (found: {available})" if available else "")
        )

    combined = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.uint8)
    for path in candidates:
        mask = sitk.ReadImage(str(path))
        if mask.GetSize() != reference.GetSize() or not np.allclose(mask.GetSpacing(), reference.GetSpacing()):
            mask = sitk.Resample(mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        combined |= sitk.GetArrayFromImage(mask).astype(bool).astype(np.uint8)
    if not combined.any():
        raise ValueError("TotalSegmentator lung mask is empty")
    output = sitk.GetImageFromArray(combined)
    output.CopyInformation(reference)
    return output


def _load_lung_mask(path: Path, reference: sitk.Image) -> sitk.Image:
    """Load a persisted binary lung mask and align it to the CT grid."""
    mask = sitk.ReadImage(str(path))
    if mask.GetSize() != reference.GetSize() or not np.allclose(
        mask.GetSpacing(), reference.GetSpacing()
    ):
        mask = sitk.Resample(
            mask,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
    output = sitk.Cast(mask > 0, sitk.sitkUInt8)
    output.CopyInformation(reference)
    return output


def _totalsegmentator_api_device(device: str | None) -> str:
    """Translate the former CLI device spelling to the Python API spelling."""
    normalized = (device or "cuda").casefold()
    aliases = {"gpu": "cuda", "cuda": "cuda", "cpu": "cpu", "mps": "mps"}
    if normalized not in aliases:
        raise ValueError(
            "TotalSegmentator device must be one of: cuda, cpu, mps "
            "(gpu is accepted as an alias for cuda)."
        )
    return aliases[normalized]


def _is_windows_file_lock(error: PermissionError) -> bool:
    """Return whether Windows reports a transient file-sharing violation."""
    return getattr(error, "winerror", None) == 32


def segment_lung_with_totalsegmentator(
    ct_hu_clipped: sitk.Image,
    case_id: str,
    staging_root: Path,
    task_id: int = TOTALSEGMENTATOR_LUNG_TASK_ID,
    trainer: str = "nnUNetTrainerNoMirroring",
    resample_mm: float = 1.5,
    device: str | None = None,
    overwrite: bool = False,
    lung_rois: Sequence[str] | None = None,
) -> tuple[sitk.Image, Path, Path]:
    """Persist a HU NIfTI and create/load a TotalSegmentator lung mask.

    This directly invokes the Task 291 nnU-Net model through TotalSegmentator's
    installed Python API.  The standard ``TotalSegmentator --task total`` CLI
    downloads all five total-body task weights before applying ``--roi_subset``;
    direct Task 291 inference downloads only the organs/lung-lobes weights.
    """
    case_root = staging_root / case_id
    input_path = case_root / "ct_hu_clipped.nii.gz"
    output_dir = case_root / "totalsegmentator"
    combined_mask_path = output_dir / "lung_mask.nii.gz"
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    case_root.mkdir(parents=True, exist_ok=True)
    if overwrite or not input_path.exists():
        sitk.WriteImage(ct_hu_clipped, str(input_path))
    if combined_mask_path.exists() and not overwrite:
        return _load_lung_mask(combined_mask_path, ct_hu_clipped), input_path, output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove a legacy persistent multilabel output from earlier script versions.
    legacy_lobes_path = output_dir / "lung_lobes.nii.gz"
    if legacy_lobes_path.exists():
        legacy_lobes_path.unlink()
    try:
        from totalsegmentator.config import setup_nnunet
        from totalsegmentator.libs import download_pretrained_weights

        # The public TotalSegmentator API calls this itself.  We invoke
        # nnUNet directly to restrict execution to Task 291, therefore
        # initialise its results path explicitly before importing nnunet.
        setup_nnunet()
        from totalsegmentator.nnunet import nnUNet_predict_image
    except ImportError as error:
        raise RuntimeError(
            "TotalSegmentator was not found. Install it with `pip install TotalSegmentator`."
        ) from error
    download_pretrained_weights(task_id)
    # nnUNet writes its multilabel lobe prediction only to this temporary
    # directory.  The only persistent model result is the binary union below.
    with tempfile.TemporaryDirectory(
        prefix=".lung_lobes_", dir=output_dir, ignore_cleanup_errors=True
    ) as temp_dir:
        temporary_output = Path(temp_dir) / "lung_lobes.nii.gz"
        for attempt in range(1, 4):
            try:
                nnUNet_predict_image(
                    input_path,
                    temporary_output,
                    task_id,
                    model="3d_fullres",
                    folds=[0],
            trainer=trainer,
                    tta=False,
                    multilabel_image=True,
            resample=resample_mm,
                    task_name="total",
                    roi_subset=list(lung_rois or LUNG_SEGMENT_LABELS),
                    device=_totalsegmentator_api_device(device),
                    quiet=False,
                )
                break
            except PermissionError as error:
                if not _is_windows_file_lock(error) or attempt == 3:
                    raise
                print(
                    f"{case_id}: Windows is temporarily locking nnU-Net files; "
                    f"retrying Task {task_id} ({attempt}/3)..."
                )
                time.sleep(2)
        lung_mask = _read_totalsegmentator_lung_mask(Path(temp_dir), ct_hu_clipped)
        # Persist before the temporary lobe output is cleaned up. Downstream
        # cropping consumes this binary union, never the lobe labels.
        sitk.WriteImage(lung_mask, str(combined_mask_path))
    return lung_mask, input_path, output_dir


def process_case(
    case_dir: Path,
    processed_root: Path,
    nifti_root: Path,
    margin_mm: float,
    overwrite: bool,
    dry_run: bool = False,
    hu_window_low: int = HU_WINDOW_LOW,
    hu_window_high: int = HU_WINDOW_HIGH,
    crop_z_to_lung_bbox: bool = True,
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
    region = lung_crop_region(
        lung_mask, margin_mm=margin_mm, crop_z_to_lung_bbox=crop_z_to_lung_bbox
    )
    image_final = crop_and_resize(
        _normalize_ct_image(ct_image, hu_window_low, hu_window_high),
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
        crop_z_to_lung_bbox=str(crop_z_to_lung_bbox),
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


def process_radiogenomics_case(
    case_dir: Path,
    processed_root: Path,
    nifti_root: Path,
    staging_root: Path,
    margin_mm: float,
    overwrite: bool,
    totalsegmentator_task_id: int,
    totalsegmentator_device: str | None,
    dry_run: bool = False,
    hu_window_low: int = HU_WINDOW_LOW,
    hu_window_high: int = HU_WINDOW_HIGH,
    totalsegmentator_lung_rois: Sequence[str] | None = None,
    totalsegmentator_trainer: str = "nnUNetTrainerNoMirroring",
    totalsegmentator_resample_mm: float = 1.5,
    output_z_spacing_mm: float | None = None,
    reuse_existing_lung_mask: bool = True,
    tumor_closing_radius_mm: float = 0.0,
    tumor_fill_holes: bool = True,
    tumor_keep_largest_component: bool = True,
    tumor_min_component_voxels: int = 0,
    tumor_dilation_radius_mm: float = 0.0,
    crop_z_to_lung_bbox: bool = True,
) -> dict[str, str]:
    """Process one Rxx-xxx case with reusable TotalSegmentator lung masks."""
    case_id = output_case_id(case_dir.name)
    result = {"raw_case_id": case_dir.name, "case_id": case_id, "status": "ok", "message": ""}
    if ((processed_root / case_id).exists() or (nifti_root / case_id).exists()) and not overwrite:
        result.update(status="skipped", message="output already exists")
        return result

    ct_image, ct_paths = _load_ct(case_dir)
    hu_clipped = clip_hu_image(ct_image, hu_window_low, hu_window_high)
    if dry_run:
        lung_mask = derive_lung_mask_from_ct(ct_image)
        lung_label = "CT-derived lung fallback (dry run)"
        input_path, totalseg_dir = staging_root / case_id / "ct_hu_clipped.nii.gz", staging_root / case_id / "totalsegmentator"
    else:
        lung_mask, input_path, totalseg_dir = segment_lung_with_totalsegmentator(
            hu_clipped, case_id, staging_root, task_id=totalsegmentator_task_id,
            trainer=totalsegmentator_trainer,
            resample_mm=totalsegmentator_resample_mm,
            device=totalsegmentator_device,
            # Output regeneration must not force an expensive new lung
            # segmentation. If a persisted mask is absent, the helper still
            # runs TotalSegmentator automatically.
            overwrite=overwrite and not reuse_existing_lung_mask,
            lung_rois=totalsegmentator_lung_rois,
        )
        lung_label = f"TotalSegmentator:Task{totalsegmentator_task_id}:lung_lobes"

    try:
        seg_path = _find_seg(case_dir)
        tumor_mask, selected_tumor_labels = _decode_segments_to_ct(
            seg_path, RADIOGENOMICS_TUMOR_SEGMENT_LABELS, ct_image
        )
    except ValueError:
        rtstruct_path = _find_rtstruct(case_dir)
        tumor_mask, selected_tumor_labels = _decode_rtstruct_tumor_to_ct(
            rtstruct_path, ct_image
        )

    tumor_mask, tumor_cleanup = clean_tumor_mask(
        tumor_mask,
        closing_radius_mm=tumor_closing_radius_mm,
        fill_holes=tumor_fill_holes,
        keep_largest_component=tumor_keep_largest_component,
        minimum_component_voxels=tumor_min_component_voxels,
        dilation_radius_mm=tumor_dilation_radius_mm,
    )

    lung_mask, removed_lung_components = remove_small_lung_components(lung_mask, LUNG_MIN_COMPONENT_VOXELS)
    region = lung_crop_region(
        lung_mask, margin_mm=margin_mm, crop_z_to_lung_bbox=crop_z_to_lung_bbox
    )
    image_final = crop_and_resize(
        _normalize_ct_image(ct_image, hu_window_low, hu_window_high),
        region,
        is_label=False,
        output_z_spacing_mm=output_z_spacing_mm,
    )
    mask_final = crop_and_resize(
        tumor_mask,
        region,
        is_label=True,
        output_z_spacing_mm=output_z_spacing_mm,
    )
    result.update(
        ct_slices=str(len(ct_paths)), lung_source="TotalSegmentator", lung_labels=lung_label,
        tumor_source="DICOM SEG/RTSTRUCT", tumor_labels=";".join(selected_tumor_labels),
        crop_index=str(region[0]), crop_size=str(region[1]),
        crop_z_to_lung_bbox=str(crop_z_to_lung_bbox), final_size=str(image_final.GetSize()),
        final_spacing=str(image_final.GetSpacing()),
    )
    result.update(totalsegmentator_input=str(input_path), totalsegmentator_output=str(totalseg_dir))
    result.update({key: str(value) for key, value in tumor_cleanup.items()})
    if not dry_run:
        write_case_outputs(processed_root, nifti_root, case_id, image_final, mask_final, overwrite=overwrite)
    return result


def _write_report(rows: list[dict[str, str]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with report_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _same_image_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    """Return whether two images occupy the same voxel grid."""
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing())
        and np.allclose(first.GetOrigin(), second.GetOrigin())
        and np.allclose(first.GetDirection(), second.GetDirection())
    )


def resample_image_to_fixed_size(
    image: sitk.Image,
    target_size_xyz: tuple[int, int, int],
    *,
    is_label: bool,
) -> sitk.Image:
    """Resample an image while retaining its first/final voxel-centre extent."""
    if any(size < 1 for size in target_size_xyz):
        raise ValueError("all target dimensions must be positive")
    source_size = image.GetSize()
    source_spacing = image.GetSpacing()
    target_spacing = tuple(
        source_spacing[axis]
        if source_size[axis] == 1 or target_size_xyz[axis] == 1
        else source_spacing[axis]
        * (source_size[axis] - 1)
        / (target_size_xyz[axis] - 1)
        for axis in range(3)
    )
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size_xyz)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)


def write_3d_nifti_case(
    processed_3d_root: Path,
    case_id: str,
    image: sitk.Image,
    mask: sitk.Image,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write a paired 3D NIfTI case to ``processed-3d``."""
    if not _same_image_geometry(image, mask):
        raise ValueError("image and mask geometry must match")
    destination = processed_3d_root / case_id
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists for {case_id}")

    processed_3d_root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = processed_3d_root / f".{case_id}.{token}.tmp"
    backup = processed_3d_root / f".{case_id}.{token}.backup"
    replaced = False
    try:
        temporary.mkdir()
        sitk.WriteImage(image, str(temporary / "image.nii.gz"))
        sitk.WriteImage(sitk.Cast(mask > 0, sitk.sitkUInt8), str(temporary / "mask.nii.gz"))
        if destination.exists():
            _replace_path_with_retry(destination, backup)
        _replace_path_with_retry(temporary, destination)
        replaced = True
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        if replaced:
            shutil.rmtree(destination, ignore_errors=True)
        if backup.exists():
            backup.replace(destination)
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def resize_nifti_case_to_3d(
    source_case_dir: Path,
    processed_3d_root: Path,
    *,
    target_size_xyz: tuple[int, int, int] = (256, 256, 128),
    overwrite: bool = False,
) -> dict[str, str]:
    """Resize one paired NIfTI case to a fixed 3D grid.

    The input grid geometry is preserved in the return value so an inference
    mask can later be resampled back to the original NIfTI grid.  Image data
    uses linear interpolation and binary labels use nearest-neighbour.
    """
    if any(size < 1 for size in target_size_xyz):
        raise ValueError("all target dimensions must be positive")

    case_id = source_case_dir.name
    image_path = source_case_dir / "image.nii.gz"
    mask_path = source_case_dir / "mask.nii.gz"
    if not image_path.is_file() or not mask_path.is_file():
        missing = [
            path.name for path in (image_path, mask_path) if not path.is_file()
        ]
        raise FileNotFoundError(f"missing paired NIfTI file(s): {', '.join(missing)}")

    destination = processed_3d_root / case_id
    if destination.exists() and not overwrite:
        return {
            "case_id": case_id,
            "status": "skipped",
            "message": "output already exists",
        }

    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if not _same_image_geometry(image, mask):
        raise ValueError("image.nii.gz and mask.nii.gz do not share the same geometry")

    source_size = image.GetSize()
    source_spacing = image.GetSpacing()
    image_3d = resample_image_to_fixed_size(
        image, target_size_xyz, is_label=False
    )
    mask_3d = sitk.Cast(
        resample_image_to_fixed_size(
            sitk.Cast(mask > 0, sitk.sitkUInt8), target_size_xyz, is_label=True
        )
        > 0,
        sitk.sitkUInt8,
    )
    write_3d_nifti_case(
        processed_3d_root, case_id, image_3d, mask_3d, overwrite=overwrite
    )

    return {
        "case_id": case_id,
        "status": "ok",
        "message": "",
        # SimpleITK size/spacing are XYZ; arrays exposed to PyTorch are DHW.
        "original_shape_dhw": str((source_size[2], source_size[1], source_size[0])),
        "original_depth": str(source_size[2]),
        "original_height": str(source_size[1]),
        "original_width": str(source_size[0]),
        "original_spacing_xyz_mm": str(source_spacing),
        "original_spacing_zxy_mm": str(
            (source_spacing[2], source_spacing[0], source_spacing[1])
        ),
        "original_origin_xyz_mm": str(image.GetOrigin()),
        "original_direction": str(image.GetDirection()),
        "processed_shape_dhw": str(
            (target_size_xyz[2], target_size_xyz[1], target_size_xyz[0])
        ),
        "processed_spacing_xyz_mm": str(image_3d.GetSpacing()),
    }


def _three_d_geometry_report(
    source_image: sitk.Image, resized_image: sitk.Image
) -> dict[str, str]:
    """Build restoration metadata for a volume represented as a DHW array."""
    source_size = source_image.GetSize()
    source_spacing = source_image.GetSpacing()
    return {
        "original_shape_dhw": str((source_size[2], source_size[1], source_size[0])),
        "original_depth": str(source_size[2]),
        "original_height": str(source_size[1]),
        "original_width": str(source_size[0]),
        "original_spacing_xyz_mm": str(source_spacing),
        "original_spacing_zxy_mm": str(
            (source_spacing[2], source_spacing[0], source_spacing[1])
        ),
        "original_origin_xyz_mm": str(source_image.GetOrigin()),
        "original_direction": str(source_image.GetDirection()),
        "processed_shape_dhw": str(
            (
                resized_image.GetSize()[2],
                resized_image.GetSize()[1],
                resized_image.GetSize()[0],
            )
        ),
        "processed_spacing_xyz_mm": str(resized_image.GetSpacing()),
    }


def process_case_3d(
    case_dir: Path,
    processed_3d_root: Path,
    margin_mm: float,
    overwrite: bool,
    *,
    target_size_xyz: tuple[int, int, int] = (256, 256, 128),
    dry_run: bool = False,
    hu_window_low: int = HU_WINDOW_LOW,
    hu_window_high: int = HU_WINDOW_HIGH,
    crop_z_to_lung_bbox: bool = True,
) -> dict[str, str]:
    """Create a fixed-size 3D Radiomics case directly from raw DICOM."""
    case_id = output_case_id(case_dir.name)
    result = {
        "raw_case_id": case_dir.name,
        "case_id": case_id,
        "status": "ok",
        "message": "",
    }
    if (processed_3d_root / case_id).exists() and not overwrite:
        result.update(status="skipped", message="output already exists")
        return result

    ct_image, ct_paths = _load_ct(case_dir)
    try:
        seg_path = _find_seg(case_dir)
    except ValueError:
        seg_path = None
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
            tumor_mask, tumor_labels = _decode_segments_to_ct(
                seg_path, TUMOR_SEGMENT_LABELS, ct_image
            )
        except ValueError:
            tumor_mask, tumor_labels = _decode_rtstruct_tumor_to_ct(
                _find_rtstruct(case_dir), ct_image
            )
    else:
        tumor_mask, tumor_labels = _decode_rtstruct_tumor_to_ct(
            _find_rtstruct(case_dir), ct_image
        )

    region = lung_crop_region(
        lung_mask, margin_mm=margin_mm, crop_z_to_lung_bbox=crop_z_to_lung_bbox
    )
    # This is the (z, 256, 256) grid retained in the CSV for restoration.
    image_before_3d = crop_and_resize(
        _normalize_ct_image(ct_image, hu_window_low, hu_window_high),
        region,
        is_label=False,
    )
    mask_before_3d = crop_and_resize(tumor_mask, region, is_label=True)
    image_final = resample_image_to_fixed_size(
        image_before_3d, target_size_xyz, is_label=False
    )
    mask_final = sitk.Cast(
        resample_image_to_fixed_size(
            sitk.Cast(mask_before_3d > 0, sitk.sitkUInt8),
            target_size_xyz,
            is_label=True,
        )
        > 0,
        sitk.sitkUInt8,
    )
    result.update(
        ct_slices=str(len(ct_paths)),
        seg_path=str(seg_path or ""),
        lung_labels=";".join(lung_labels),
        removed_lung_components=str(removed_lung_components),
        tumor_labels=";".join(tumor_labels),
        crop_index=str(region[0]),
        crop_size=str(region[1]),
        crop_z_to_lung_bbox=str(crop_z_to_lung_bbox),
    )
    result.update(_three_d_geometry_report(image_before_3d, image_final))
    if not dry_run:
        write_3d_nifti_case(
            processed_3d_root, case_id, image_final, mask_final, overwrite=overwrite
        )
    return result


def process_radiogenomics_case_3d(
    case_dir: Path,
    processed_3d_root: Path,
    staging_root: Path,
    margin_mm: float,
    overwrite: bool,
    totalsegmentator_task_id: int,
    totalsegmentator_device: str | None,
    *,
    target_size_xyz: tuple[int, int, int] = (256, 256, 128),
    dry_run: bool = False,
    hu_window_low: int = HU_WINDOW_LOW,
    hu_window_high: int = HU_WINDOW_HIGH,
    totalsegmentator_lung_rois: Sequence[str] | None = None,
    totalsegmentator_trainer: str = "nnUNetTrainerNoMirroring",
    totalsegmentator_resample_mm: float = 1.5,
    reuse_existing_lung_mask: bool = True,
    tumor_closing_radius_mm: float = 0.0,
    tumor_fill_holes: bool = True,
    tumor_keep_largest_component: bool = True,
    tumor_min_component_voxels: int = 0,
    tumor_dilation_radius_mm: float = 0.0,
    crop_z_to_lung_bbox: bool = True,
) -> dict[str, str]:
    """Create a fixed-size 3D Radiogenomics case directly from raw DICOM."""
    case_id = output_case_id(case_dir.name)
    result = {
        "raw_case_id": case_dir.name,
        "case_id": case_id,
        "status": "ok",
        "message": "",
    }
    if (processed_3d_root / case_id).exists() and not overwrite:
        result.update(status="skipped", message="output already exists")
        return result

    ct_image, ct_paths = _load_ct(case_dir)
    hu_clipped = clip_hu_image(ct_image, hu_window_low, hu_window_high)
    if dry_run:
        lung_mask = derive_lung_mask_from_ct(ct_image)
        lung_label = "CT-derived lung fallback (dry run)"
        input_path = staging_root / case_id / "ct_hu_clipped.nii.gz"
        totalseg_dir = staging_root / case_id / "totalsegmentator"
    else:
        lung_mask, input_path, totalseg_dir = segment_lung_with_totalsegmentator(
            hu_clipped,
            case_id,
            staging_root,
            task_id=totalsegmentator_task_id,
            trainer=totalsegmentator_trainer,
            resample_mm=totalsegmentator_resample_mm,
            device=totalsegmentator_device,
            overwrite=overwrite and not reuse_existing_lung_mask,
            lung_rois=totalsegmentator_lung_rois,
        )
        lung_label = f"TotalSegmentator:Task{totalsegmentator_task_id}:lung_lobes"
    try:
        seg_path = _find_seg(case_dir)
        tumor_mask, tumor_labels = _decode_segments_to_ct(
            seg_path, RADIOGENOMICS_TUMOR_SEGMENT_LABELS, ct_image
        )
    except ValueError:
        tumor_mask, tumor_labels = _decode_rtstruct_tumor_to_ct(
            _find_rtstruct(case_dir), ct_image
        )
    tumor_mask, tumor_cleanup = clean_tumor_mask(
        tumor_mask,
        closing_radius_mm=tumor_closing_radius_mm,
        fill_holes=tumor_fill_holes,
        keep_largest_component=tumor_keep_largest_component,
        minimum_component_voxels=tumor_min_component_voxels,
        dilation_radius_mm=tumor_dilation_radius_mm,
    )
    lung_mask, removed_lung_components = remove_small_lung_components(
        lung_mask, LUNG_MIN_COMPONENT_VOXELS
    )
    region = lung_crop_region(
        lung_mask, margin_mm=margin_mm, crop_z_to_lung_bbox=crop_z_to_lung_bbox
    )
    image_before_3d = crop_and_resize(
        _normalize_ct_image(ct_image, hu_window_low, hu_window_high),
        region,
        is_label=False,
    )
    mask_before_3d = crop_and_resize(tumor_mask, region, is_label=True)
    image_final = resample_image_to_fixed_size(
        image_before_3d, target_size_xyz, is_label=False
    )
    mask_final = sitk.Cast(
        resample_image_to_fixed_size(
            sitk.Cast(mask_before_3d > 0, sitk.sitkUInt8),
            target_size_xyz,
            is_label=True,
        )
        > 0,
        sitk.sitkUInt8,
    )
    result.update(
        ct_slices=str(len(ct_paths)),
        lung_source="TotalSegmentator",
        lung_labels=lung_label,
        tumor_source="DICOM SEG/RTSTRUCT",
        tumor_labels=";".join(tumor_labels),
        removed_lung_components=str(removed_lung_components),
        crop_index=str(region[0]),
        crop_size=str(region[1]),
        crop_z_to_lung_bbox=str(crop_z_to_lung_bbox),
        totalsegmentator_input=str(input_path),
        totalsegmentator_output=str(totalseg_dir),
    )
    result.update({key: str(value) for key, value in tumor_cleanup.items()})
    result.update(_three_d_geometry_report(image_before_3d, image_final))
    if not dry_run:
        write_3d_nifti_case(
            processed_3d_root, case_id, image_final, mask_final, overwrite=overwrite
        )
    return result
