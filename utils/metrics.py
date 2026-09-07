"""Binary 2D and spacing-aware 3D segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def _overlap_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    both_empty = not pred.any() and not truth.any()
    dice = (
        1.0
        if both_empty
        else (2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    )
    iou = 1.0 if both_empty else (tp / (tp + fp + fn) if tp + fp + fn else 0.0)
    recall = (
        1.0
        if not truth.any() and not pred.any()
        else (tp / (tp + fn) if tp + fn else 0.0)
    )
    precision = (
        1.0
        if not pred.any() and not truth.any()
        else (tp / (tp + fp) if tp + fp else 0.0)
    )
    return {
        "dice": float(dice),
        "iou": float(iou),
        "recall": float(recall),
        "precision": float(precision),
        "fp": fp,
        "fn": fn,
    }


def binary_slice_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float | int]:
    """Compute overlap, recall/precision, FP and FN for one binary slice."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    return _overlap_metrics(prediction, target)


def _surface_points(
    mask_zyx: np.ndarray, spacing_xyz: tuple[float, float, float]
) -> np.ndarray:
    eroded = binary_erosion(mask_zyx, border_value=0)
    surface = np.logical_and(mask_zyx, ~eroded)
    coordinates_zyx = np.argwhere(surface).astype(np.float64)
    return coordinates_zyx * np.asarray(spacing_xyz[::-1], dtype=np.float64)


def _directed_surface_distances(
    source: np.ndarray, destination: np.ndarray, spacing_xyz: tuple[float, float, float]
) -> np.ndarray:
    source_points = _surface_points(source, spacing_xyz)
    destination_points = _surface_points(destination, spacing_xyz)
    return cKDTree(destination_points).query(source_points, k=1)[0]


def binary_volume_metrics(
    prediction: np.ndarray, target: np.ndarray, spacing_xyz: tuple[float, float, float]
) -> dict[str, float | int]:
    """Compute binary overlap and physical HD95/ASSD for a complete ZYX volume."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    result = _overlap_metrics(prediction, target)
    if not np.any(prediction) or not np.any(target):
        return result | {"hd95": float("nan"), "assd": float("nan")}
    distances = np.concatenate(
        [
            _directed_surface_distances(prediction, target, spacing_xyz),
            _directed_surface_distances(target, prediction, spacing_xyz),
        ]
    )
    return result | {
        "hd95": float(np.percentile(distances, 95)),
        "assd": float(distances.mean()),
    }


@dataclass(frozen=True)
class SlicePrediction:
    """One thresholded output and target associated with a volume position."""

    case_id: str
    slice_index: int
    prediction: np.ndarray
    target: np.ndarray


def aggregate_case_predictions(
    records: Sequence[SlicePrediction],
) -> dict[str, dict[str, np.ndarray]]:
    """Sort slice records and stack them to ZYX case volumes."""
    grouped: dict[str, list[SlicePrediction]] = {}
    for record in records:
        grouped.setdefault(record.case_id, []).append(record)
    volumes: dict[str, dict[str, np.ndarray]] = {}
    for case_id, case_records in grouped.items():
        ordered = sorted(case_records, key=lambda record: record.slice_index)
        expected = list(range(len(ordered)))
        actual = [record.slice_index for record in ordered]
        if actual != expected:
            raise ValueError(
                f"case {case_id} has non-contiguous slice indices: {actual}"
            )
        volumes[case_id] = {
            "prediction": np.stack(
                [record.prediction.astype(bool) for record in ordered]
            ),
            "target": np.stack([record.target.astype(bool) for record in ordered]),
        }
    return volumes


def finite_mean(values: Sequence[float]) -> float:
    """Return the mean of finite values or NaN when no finite values exist."""
    finite = np.asarray(
        [value for value in values if np.isfinite(value)], dtype=np.float64
    )
    return float(finite.mean()) if finite.size else float("nan")
