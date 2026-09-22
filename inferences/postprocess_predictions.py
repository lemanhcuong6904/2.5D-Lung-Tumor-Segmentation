"""3D post-process saved prediction volumes and re-evaluate them.

Set the paths and morphology parameters in ``CFG`` then run this file directly.
The input directory must contain one ``<case_id>.nii.gz`` binary prediction per
case.  Cleaned masks and reports are written below ``OUTPUT_DIR``.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from tqdm.auto import tqdm

from utils.metrics import binary_slice_metrics, binary_volume_metrics, finite_mean


CFG: dict[str, Any] = {
    # Selects dataset-specific final lesion selection. Paths remain explicit
    # below so a particular experiment/run can be evaluated safely.
    "DATASET": "nsclc_radiomics",  # "nsclc_radiomics" or "nsclc_radiogenomics"
    # Directory containing <case_id>.nii.gz volumes produced by inference.
    "PREDICTIONS_DIR": str(
        ROOT_DIR
        / "output"
        / "transunet_2.5d-5_balanced_sampling"
        / "test"
        / "predictions"
    ),
    "NIFTI_ROOT": str(ROOT_DIR / "data" / "nsclc-radiomics" / "nifti"),
    # Outputs: predictions/ plus per_case_metrics.csv, slice_metrics.csv, and
    # summary_metrics.json.  Keep this outside the original inference results.
    "OUTPUT_DIR": str(
        ROOT_DIR
        / "output"
        / "transunet_2.5d-5_balanced_sampling"
        / "test"
        / "post_processed"
    ),
    # Physical radii for 3D ellipsoidal morphology.  They are converted using
    # each case's X/Y/Z spacing so anisotropic CT voxels retain physical scale.
    "CLOSING_RADIUS_MM": 3.0,
    # Opening is deliberately disabled: erosion can split thin, true tumour
    # regions before connected-component analysis.
    "OPENING_RADIUS_MM": 0.0,
    # Remove only unequivocally tiny components. Set to 0 to disable this
    # first-pass filter.
    "MIN_COMPONENT_VOLUME_MM3": 50.0,
    "CONNECTIVITY": 26,
    # Components within this physical surface distance are treated as one
    # lesion cluster. The link is used for selection only; it never dilates
    # the output mask or fills the gap between components.
    "COMPONENT_LINK_DISTANCE_MM": 9.0,
    # Also retain a separate cluster if it is substantial relative to the
    # dominant cluster. This is the binary-mask fallback for a high-confidence
    # secondary component; set to 1.0 to keep only the dominant cluster.
    "SECONDARY_CLUSTER_MIN_VOLUME_RATIO": 0.10,
    # Radiogenomics only: after the standard post-processing above, retain one
    # lesion using a weighted score across connected components. Scores are
    # normalized within each volume before applying these weights.
    "RADIOGENOMICS_VOLUME_WEIGHT": 0.20,
    "RADIOGENOMICS_Z_THICKNESS_WEIGHT": 0.10,
    "RADIOGENOMICS_SLICE_STABILITY_WEIGHT": 0.70,
}

# This is intentionally disabled by default.  It uses test-set ground truth to
# choose parameters and therefore gives an optimistically biased test score.
# Enable only for the requested oracle/sensitivity analysis; tune on validation
# data for a final paper setting.
TUNING_CFG: dict[str, Any] = {
    "RUN_TUNING": False,
    "CLOSING_RADIUS_MM": (2.0, 3.0, 4.0, 5.0),
    "OPENING_RADIUS_MM": (0.0,),
    "MIN_COMPONENT_VOLUME_MM3": (0.0, 10.0, 25.0, 50.0, 100.0, 250.0),
    "CONNECTIVITY": (6, 18, 26),
    "COMPONENT_LINK_DISTANCE_MM": (6.0, 9.0, 12.0, 15.0),
    "SECONDARY_CLUSTER_MIN_VOLUME_RATIO": (0.05, 0.10),
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite_std_iqr(values: list[float]) -> tuple[float, float]:
    """Return sample standard deviation and IQR after excluding non-finite values."""
    finite_values = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite_values.size == 0:
        return float("nan"), float("nan")
    standard_deviation = float(finite_values.std(ddof=1)) if finite_values.size > 1 else 0.0
    interquartile_range = float(np.percentile(finite_values, 75) - np.percentile(finite_values, 25))
    return standard_deviation, interquartile_range


def ellipsoidal_structure(radius_mm: float, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """Return a ZYX ellipsoid whose voxel radii follow round(radius / spacing)."""
    if radius_mm < 0:
        raise ValueError("morphology radius must be non-negative")
    if radius_mm == 0:
        return np.ones((1, 1, 1), dtype=bool)
    radii_xyz = np.rint(radius_mm / np.asarray(spacing_xyz, dtype=float)).astype(int)
    radii_xyz = np.maximum(radii_xyz, 0)
    # Mesh in the array's ZYX order.  A zero radius restricts that axis to its
    # centre plane, which is appropriate when the requested physical radius is
    # smaller than the slice spacing.
    radii_zyx = radii_xyz[::-1]
    z, y, x = np.ogrid[
        -radii_zyx[0] : radii_zyx[0] + 1,
        -radii_zyx[1] : radii_zyx[1] + 1,
        -radii_zyx[2] : radii_zyx[2] + 1,
    ]
    distance = np.zeros((2 * radii_zyx[0] + 1, 2 * radii_zyx[1] + 1, 2 * radii_zyx[2] + 1), dtype=float)
    for coordinates, radius in zip((z, y, x), radii_zyx, strict=True):
        if radius == 0:
            distance += np.where(coordinates == 0, 0.0, np.inf)
        else:
            distance += (coordinates / radius) ** 2
    return distance <= 1.0


def _connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity not in (6, 18, 26):
        raise ValueError("CONNECTIVITY must be 6, 18, or 26")
    return ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])


def _z_link_structure() -> np.ndarray:
    """Allow diagonal motion only when a connection advances through Z.

    The 3x3 neighbourhood in the preceding/following Z slice allows a contour
    to move diagonally as it is interpolated.  The centre Z plane contains
    ordinary face-connected neighbourhood is retained in the centre Z plane
    so a pre-existing 2D component stays intact.  This structure never fills
    or bridges a horizontal X/Y gap.
    """
    # Keep the ordinary face-connected neighbourhood in the current slice so
    # each original 2D component remains intact.  It does not fill or bridge
    # any horizontal gap; it only defines a component's existing topology.
    structure = ndimage.generate_binary_structure(3, 1)
    structure[0, :, :] = True
    structure[2, :, :] = True
    return structure


def _signed_distance_2d(mask_yx: np.ndarray, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """Signed 2D distance: positive inside a mask and negative outside it."""
    sampling_yx = (float(spacing_xyz[1]), float(spacing_xyz[0]))
    mask = np.asarray(mask_yx, dtype=bool)
    return (
        ndimage.distance_transform_edt(mask, sampling=sampling_yx)
        - ndimage.distance_transform_edt(~mask, sampling=sampling_yx)
    )


def _inplane_disk_structure(radius_mm: float, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """Physical-radius 2D disk in YX array order."""
    if radius_mm <= 0:
        return np.ones((1, 1), dtype=bool)
    radius_y = int(np.ceil(radius_mm / float(spacing_xyz[1])))
    radius_x = int(np.ceil(radius_mm / float(spacing_xyz[0])))
    y, x = np.ogrid[-radius_y : radius_y + 1, -radius_x : radius_x + 1]
    return (y * float(spacing_xyz[1])) ** 2 + (x * float(spacing_xyz[0])) ** 2 <= radius_mm**2


def _projection_distance_mm(
    first_projection_yx: np.ndarray,
    second_projection_yx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> float:
    """Shortest physical XY distance between two component projections."""
    if np.any(first_projection_yx & second_projection_yx):
        return 0.0
    sampling_yx = (float(spacing_xyz[1]), float(spacing_xyz[0]))
    distance_to_first = ndimage.distance_transform_edt(~first_projection_yx, sampling=sampling_yx)
    return float(np.min(distance_to_first[second_projection_yx]))


def _global_chain_interpolation(
    labels_zyx: np.ndarray,
    components: list[dict[str, Any]],
    chain_indices: list[int],
    spacing_xyz: tuple[float, float, float],
    min_reliable_anchor_volume_ratio: float,
    max_terminal_extrapolation_mm: float,
    terminal_inplane_margin_mm: float,
    max_terminal_anchor_volume_ratio: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Loft one component chain with robust, global shape anchors.

    Only components with enough volume contribute contours to the monotone
    cubic centre trajectory.  Low-confidence fragments are replaced within
    the anchor span rather than being allowed to create a narrow waist.
    """
    chain_labels = {int(components[index]["label"]) for index in chain_indices}
    chain_mask = np.isin(labels_zyx, list(chain_labels))
    strongest_component_voxels = max(int(components[index]["voxel_count"]) for index in chain_indices)
    reliable_indices = [
        index
        for index in chain_indices
        if int(components[index]["voxel_count"])
        >= strongest_component_voxels * min_reliable_anchor_volume_ratio
    ]
    # Never attempt a loft from a single anchor: in that situation the source
    # mask is safer than extrapolating an unsupported tumour shape.
    if len(reliable_indices) < 2:
        return np.zeros_like(chain_mask, dtype=bool), np.zeros(chain_mask.shape[0], dtype=bool), 0, 0
    reliable_labels = {int(components[index]["label"]) for index in reliable_indices}
    anchor_mask = np.isin(labels_zyx, list(reliable_labels))
    known_z = np.flatnonzero(anchor_mask.any(axis=(1, 2)))
    if known_z.size < 2:
        return np.zeros_like(chain_mask, dtype=bool), np.zeros(chain_mask.shape[0], dtype=bool), 0, 0

    centres_yx = np.empty((known_z.size, 2), dtype=float)
    for index, z in enumerate(known_z):
        coordinates_yx = np.argwhere(anchor_mask[z])
        centres_yx[index] = coordinates_yx.mean(axis=0)

    # PCHIP uses every contour in the chain while avoiding cubic overshoot,
    # which is important when a small imperfect component is off-centre.
    all_z = np.arange(int(known_z[0]), int(known_z[-1]) + 1)
    trajectory_y = PchipInterpolator(known_z, centres_yx[:, 0])(all_z)
    trajectory_x = PchipInterpolator(known_z, centres_yx[:, 1])(all_z)
    trajectory = {int(z): np.asarray((y, x), dtype=float) for z, y, x in zip(all_z, trajectory_y, trajectory_x, strict=True)}
    known_centres = {int(z): centre for z, centre in zip(known_z, centres_yx, strict=True)}

    lofted = np.zeros_like(chain_mask, dtype=bool)
    for lower_z, upper_z in zip(known_z[:-1], known_z[1:], strict=True):
        lower_z, upper_z = int(lower_z), int(upper_z)
        lofted[lower_z] |= anchor_mask[lower_z]
        if upper_z == lower_z + 1:
            continue
        lower_sdf = _signed_distance_2d(anchor_mask[lower_z], spacing_xyz)
        upper_sdf = _signed_distance_2d(anchor_mask[upper_z], spacing_xyz)
        outside_distance = -float(max(chain_mask.shape[1:]) * max(spacing_xyz[:2]))
        for z in range(lower_z + 1, upper_z):
            fraction = (z - lower_z) / (upper_z - lower_z)
            shifted_lower = ndimage.shift(
                lower_sdf,
                shift=trajectory[z] - known_centres[lower_z],
                order=1,
                mode="constant",
                cval=outside_distance,
                prefilter=False,
            )
            shifted_upper = ndimage.shift(
                upper_sdf,
                shift=trajectory[z] - known_centres[upper_z],
                order=1,
                mode="constant",
                cval=outside_distance,
                prefilter=False,
            )
            lofted[z] |= (1.0 - fraction) * shifted_lower + fraction * shifted_upper >= 0.0
    lofted[int(known_z[-1])] |= anchor_mask[int(known_z[-1])]
    replacement_z = np.zeros(chain_mask.shape[0], dtype=bool)
    replacement_z[int(known_z[0]) : int(known_z[-1]) + 1] = True
    low_confidence_voxels = int((chain_mask[replacement_z] & ~anchor_mask[replacement_z]).sum())
    terminal_voxels_added = 0
    max_terminal_slices = int(np.floor(max_terminal_extrapolation_mm / float(spacing_xyz[2])))
    if max_terminal_slices > 0 and terminal_inplane_margin_mm > 0:
        reliable_set = set(reliable_indices)
        for edge_z, direction in ((int(known_z[0]), -1), (int(known_z[-1]), 1)):
            # Complete only a real low-confidence terminal fragment.  We do
            # not invent a cap on a side with no prediction evidence at all.
            candidates = [
                index
                for index in chain_indices
                if index not in reliable_set
                and (int(components[index]["z_last"]) < edge_z if direction < 0 else int(components[index]["z_first"]) > edge_z)
            ]
            if not candidates:
                continue
            terminal_index = (
                max(candidates, key=lambda index: int(components[index]["z_last"]))
                if direction < 0
                else min(candidates, key=lambda index: int(components[index]["z_first"]))
            )
            terminal = components[terminal_index]
            if int(terminal["voxel_count"]) > strongest_component_voxels * max_terminal_anchor_volume_ratio:
                continue
            terminal_mask = labels_zyx == int(terminal["label"])
            terminal_z = np.flatnonzero(terminal_mask.any(axis=(1, 2)))
            join_z = int(terminal_z[-1] if direction < 0 else terminal_z[0])

            # Restore a physically plausible terminal fragment before joining
            # it to the main reliable shape.
            expanded_terminal = np.zeros_like(terminal_mask, dtype=bool)
            for z in terminal_z:
                before = int(lofted[z].sum())
                expanded_terminal[z] = ndimage.binary_dilation(
                    terminal_mask[z], structure=_inplane_disk_structure(terminal_inplane_margin_mm * 0.70, spacing_xyz)
                )
                lofted[z] |= expanded_terminal[z]
                terminal_voxels_added += int(lofted[z].sum()) - before

            if abs(join_z - edge_z) > 1:
                main_sdf = _signed_distance_2d(anchor_mask[edge_z], spacing_xyz)
                terminal_sdf = _signed_distance_2d(expanded_terminal[join_z], spacing_xyz)
                for z in range(min(edge_z, join_z) + 1, max(edge_z, join_z)):
                    fraction = abs(z - edge_z) / abs(join_z - edge_z)
                    lofted[z] |= (1.0 - fraction) * main_sdf + fraction * terminal_sdf >= 0.0

            end_z = int(terminal_z[0] if direction < 0 else terminal_z[-1])
            end_mask = terminal_mask[end_z]
            for distance in range(1, max_terminal_slices + 1):
                z = end_z + direction * distance
                if not 0 <= z < chain_mask.shape[0]:
                    break
                radius_mm = terminal_inplane_margin_mm * (1.0 - distance / (max_terminal_slices + 1))
                before = int(lofted[z].sum())
                lofted[z] |= ndimage.binary_dilation(end_mask, structure=_inplane_disk_structure(radius_mm, spacing_xyz))
                terminal_voxels_added += int(lofted[z].sum()) - before
    return lofted, replacement_z, low_confidence_voxels, terminal_voxels_added


def interpolate_vertical_component_gaps(
    mask_zyx: np.ndarray,
    max_gap_mm: float,
    max_lateral_match_distance_mm: float,
    spacing_xyz: tuple[float, float, float],
    connectivity: int,
    min_reliable_anchor_volume_ratio: float = 0.12,
    max_terminal_extrapolation_mm: float = 0.0,
    terminal_inplane_margin_mm: float = 0.0,
    max_terminal_anchor_volume_ratio: float = 0.25,
) -> tuple[np.ndarray, int, int, int, int]:
    """Bridge matched components by interpolating their 2D shapes through Z.

    Components are eligible only when their Z ranges are disjoint, their gap is
    within ``max_gap_mm``, and their XY projections are close enough.  The
    intermediate masks are interpolated from signed distance fields, so a
    drifting tumour contour evolves gradually from one slice to the next rather
    than becoming a straight same-(X,Y) column.  No morphology is applied in
    the X/Y plane.
    """
    if max_gap_mm < 0:
        raise ValueError("CLOSING_RADIUS_MM must be non-negative")
    if max_lateral_match_distance_mm < 0:
        raise ValueError("MAX_LATERAL_MATCH_DISTANCE_MM must be non-negative")
    if not 0.0 <= min_reliable_anchor_volume_ratio <= 1.0:
        raise ValueError("MIN_RELIABLE_ANCHOR_VOLUME_RATIO must be in [0, 1]")
    if max_terminal_extrapolation_mm < 0 or terminal_inplane_margin_mm < 0:
        raise ValueError("terminal extrapolation distances must be non-negative")
    if not 0.0 <= max_terminal_anchor_volume_ratio <= 1.0:
        raise ValueError("MAX_TERMINAL_ANCHOR_VOLUME_RATIO must be in [0, 1]")

    source = np.asarray(mask_zyx, dtype=bool)
    labels, component_count = ndimage.label(source, structure=_connectivity_structure(connectivity))
    if component_count < 2 or max_gap_mm == 0:
        return source.copy(), 0, 0, 0, 0

    max_gap_slices = int(np.floor(max_gap_mm / float(spacing_xyz[2])))
    if max_gap_slices < 1:
        return source.copy(), 0, 0, 0, 0

    components: list[dict[str, Any]] = []
    for label in range(1, component_count + 1):
        component = labels == label
        z_indices = np.flatnonzero(component.any(axis=(1, 2)))
        components.append(
            {
                "label": label,
                "z_first": int(z_indices[0]),
                "z_last": int(z_indices[-1]),
                "projection": component.any(axis=0),
                "voxel_count": int(component.sum()),
            }
        )

    # A component can receive one link from below and emit one link upward.
    # This permits a single tumour chain while preventing a small component
    # from joining several unrelated structures.
    candidates: list[tuple[float, int, int, int]] = []
    for first_index, first in enumerate(components):
        for second_index, second in enumerate(components):
            if first_index == second_index or first["z_last"] >= second["z_first"]:
                continue
            missing_slices = int(second["z_first"] - first["z_last"] - 1)
            if not 1 <= missing_slices <= max_gap_slices:
                continue
            lateral_distance_mm = _projection_distance_mm(first["projection"], second["projection"], spacing_xyz)
            if lateral_distance_mm > max_lateral_match_distance_mm:
                continue
            # Prefer the shortest Z gap, then the closest projected contour.
            score = missing_slices * float(spacing_xyz[2]) + lateral_distance_mm
            candidates.append((score, first_index, second_index, missing_slices))

    selected: list[tuple[int, int, int]] = []
    used_upper: set[int] = set()
    used_lower: set[int] = set()
    for _, first_index, second_index, missing_slices in sorted(candidates):
        if first_index in used_upper or second_index in used_lower:
            continue
        selected.append((first_index, second_index, missing_slices))
        used_upper.add(first_index)
        used_lower.add(second_index)

    # Convert the matched links to full chains.  The loft for each chain is
    # then informed by all of its observed contours rather than by an isolated
    # pair of component endpoints.
    successor = {first_index: second_index for first_index, second_index, _ in selected}
    predecessor = {second_index: first_index for first_index, second_index, _ in selected}
    chains: list[list[int]] = []
    for start in (index for index in successor if index not in predecessor):
        chain = [start]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        chains.append(chain)

    interpolated = source.copy()
    low_confidence_anchor_voxels_replaced = 0
    terminal_extrapolation_voxels_added = 0
    for chain in chains:
        lofted, replacement_z, low_confidence_voxels, terminal_voxels_added = _global_chain_interpolation(
            labels,
            components,
            chain,
            spacing_xyz,
            min_reliable_anchor_volume_ratio,
            max_terminal_extrapolation_mm,
            terminal_inplane_margin_mm,
            max_terminal_anchor_volume_ratio,
        )
        if replacement_z.any():
            chain_labels = [int(components[index]["label"]) for index in chain]
            chain_mask = np.isin(labels, chain_labels)
            interpolated[replacement_z] &= ~chain_mask[replacement_z]
            interpolated |= lofted
            low_confidence_anchor_voxels_replaced += low_confidence_voxels
            terminal_extrapolation_voxels_added += terminal_voxels_added

    return (
        interpolated,
        int(interpolated.sum() - source.sum()),
        len(selected),
        low_confidence_anchor_voxels_replaced,
        terminal_extrapolation_voxels_added,
    )


def z_only_structure(radius_mm: float, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """Return a morphology kernel with extent only along Z."""
    if radius_mm < 0:
        raise ValueError("OPENING_RADIUS_MM must be non-negative")
    radius_slices = int(np.rint(radius_mm / float(spacing_xyz[2])))
    return np.ones((2 * radius_slices + 1, 1, 1), dtype=bool)


def _component_clusters(
    labels_zyx: np.ndarray,
    component_count: int,
    spacing_xyz: tuple[float, float, float],
    link_distance_mm: float,
) -> list[list[int]]:
    """Group labels whose original surfaces are within ``link_distance_mm``.

    Distances use the image's physical spacing. This intentionally returns
    label groups only: the dilation/EDT is evidence of a relationship and is
    never written into the final segmentation.
    """
    if link_distance_mm < 0:
        raise ValueError("COMPONENT_LINK_DISTANCE_MM must be non-negative")
    parents = list(range(component_count + 1))

    def find(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    def union(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first != second:
            parents[second] = first

    if link_distance_mm > 0:
        # The EDT is in ZYX order, whereas SimpleITK reports spacing in XYZ.
        sampling_zyx = tuple(reversed(tuple(float(value) for value in spacing_xyz)))
        for label in range(1, component_count + 1):
            component = labels_zyx == label
            distances = ndimage.distance_transform_edt(~component, sampling=sampling_zyx)
            nearby_labels = np.unique(labels_zyx[(distances <= link_distance_mm) & (labels_zyx != 0)])
            for other_label in nearby_labels:
                other_label = int(other_label)
                if other_label > label:
                    union(label, other_label)

    groups: dict[int, list[int]] = {}
    for label in range(1, component_count + 1):
        groups.setdefault(find(label), []).append(label)
    return list(groups.values())


def retain_weighted_stable_lesion(
    mask_zyx: np.ndarray,
    connectivity: int,
    volume_weight: float,
    z_thickness_weight: float,
    slice_stability_weight: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Keep one component using normalized volume, Z extent, and slice stability.

    Stability is the mean IoU of masks in consecutive occupied Z slices.  It
    rewards a lesion whose contour changes smoothly through the volume, while
    component volume and Z extent prevent a tiny but coincidentally smooth
    fragment from winning.  This method never uses ground truth.
    """
    weights = np.asarray(
        (volume_weight, z_thickness_weight, slice_stability_weight), dtype=float
    )
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Radiogenomics lesion-selection weights must be non-negative and not all zero")
    weights /= weights.sum()
    labels, component_count = ndimage.label(
        np.asarray(mask_zyx, dtype=bool), structure=_connectivity_structure(connectivity)
    )
    if component_count == 0:
        return np.zeros_like(mask_zyx, dtype=bool), {
            "radiogenomics_components_before_selection": 0,
            "radiogenomics_components_removed": 0,
        }

    candidates: list[dict[str, float | int]] = []
    for label in range(1, component_count + 1):
        component = labels == label
        z_indices = np.flatnonzero(component.any(axis=(1, 2)))
        slice_masks = component[z_indices]
        if len(slice_masks) < 2:
            stability = 0.0
        else:
            overlaps = np.logical_and(slice_masks[:-1], slice_masks[1:]).sum(axis=(1, 2))
            unions = np.logical_or(slice_masks[:-1], slice_masks[1:]).sum(axis=(1, 2))
            stability = float(np.mean(np.divide(overlaps, unions, out=np.zeros_like(overlaps, dtype=float), where=unions > 0)))
        candidates.append({
            "label": label,
            "voxels": int(component.sum()),
            "z_span": int(z_indices[-1] - z_indices[0] + 1),
            "stability": stability,
        })

    maxima = np.asarray([
        max(float(candidate[key]) for candidate in candidates)
        for key in ("voxels", "z_span", "stability")
    ])
    for candidate in candidates:
        normalized = np.divide(
            np.asarray([candidate["voxels"], candidate["z_span"], candidate["stability"]], dtype=float),
            maxima,
            out=np.zeros(3, dtype=float),
            where=maxima > 0,
        )
        candidate["score"] = float(np.dot(weights, normalized))
    selected = max(candidates, key=lambda item: (float(item["score"]), float(item["stability"]), int(item["voxels"])))
    result = labels == int(selected["label"])
    return result, {
        "radiogenomics_components_before_selection": int(component_count),
        "radiogenomics_components_removed": int(component_count - 1),
        "radiogenomics_selected_component_voxels": int(selected["voxels"]),
        "radiogenomics_selected_component_z_span_slices": int(selected["z_span"]),
        "radiogenomics_selected_component_stability": float(selected["stability"]),
        "radiogenomics_selected_component_score": float(selected["score"]),
    }


def postprocess_volume(
    prediction_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    closing_radius_mm: float,
    opening_radius_mm: float,
    min_component_volume_mm3: float,
    connectivity: int,
    component_link_distance_mm: float,
    secondary_cluster_min_volume_ratio: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Conservatively clean a binary prediction without hard-LCC removal.

    The order is closing -> optional opening -> hole filling -> tiny-component
    removal -> physical-distance component clustering -> adaptive cluster
    retention. Crucially, linking only selects original components; it does
    not add a synthetic bridge to the output.
    """
    prediction = np.asarray(prediction_zyx, dtype=bool)
    if prediction.ndim != 3:
        raise ValueError(f"expected a ZYX 3D volume, got shape {prediction.shape}")
    if not 0.0 <= secondary_cluster_min_volume_ratio <= 1.0:
        raise ValueError("SECONDARY_CLUSTER_MIN_VOLUME_RATIO must be in [0, 1]")

    cleaned = prediction
    if closing_radius_mm > 0:
        cleaned = ndimage.binary_closing(
            cleaned, structure=ellipsoidal_structure(closing_radius_mm, spacing_xyz)
        )
    if opening_radius_mm > 0:
        cleaned = ndimage.binary_opening(
            cleaned, structure=ellipsoidal_structure(opening_radius_mm, spacing_xyz)
        )
    cleaned = ndimage.binary_fill_holes(cleaned)

    component_structure = _connectivity_structure(connectivity)
    labels, component_count_before = ndimage.label(cleaned, structure=component_structure)
    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    minimum_voxels = int(np.ceil(min_component_volume_mm3 / voxel_volume_mm3))
    if component_count_before == 0:
        return cleaned.astype(bool), {
            "components_before_filter": 0,
            "components_after_filter": 0,
            "spatial_cluster_count": 0,
            "retained_cluster_count": 0,
            "minimum_component_voxels": minimum_voxels,
        }
    component_sizes = np.bincount(labels.ravel())[1:]
    kept_labels = np.flatnonzero(component_sizes >= minimum_voxels) + 1
    filtered = np.isin(labels, kept_labels)
    if not filtered.any():
        return filtered, {
            "components_before_filter": int(component_count_before),
            "components_after_filter": 0,
            "spatial_cluster_count": 0,
            "retained_cluster_count": 0,
            "minimum_component_voxels": minimum_voxels,
        }
    filtered_labels, component_count_after = ndimage.label(filtered, structure=component_structure)
    filtered_sizes = np.bincount(filtered_labels.ravel())[1:]
    clusters = _component_clusters(
        filtered_labels, int(component_count_after), spacing_xyz, component_link_distance_mm
    )
    cluster_sizes = np.asarray(
        [sum(int(filtered_sizes[label - 1]) for label in cluster) for cluster in clusters], dtype=np.int64
    )
    dominant_index = int(np.argmax(cluster_sizes))
    dominant_size = int(cluster_sizes[dominant_index])
    # A sizeable but disconnected cluster can be a second part of the lesion.
    # For binary inputs we do not have probability confidence, so this is an
    # intentionally conservative proxy; tune the ratio on validation data.
    retained_indices = [
        index for index, size in enumerate(cluster_sizes)
        if index == dominant_index or size >= dominant_size * secondary_cluster_min_volume_ratio
    ]
    retained_labels = [label for index in retained_indices for label in clusters[index]]
    result = np.isin(filtered_labels, retained_labels)
    return result, {
        "components_before_filter": int(component_count_before),
        "components_after_filter": int(component_count_after),
        "spatial_cluster_count": len(clusters),
        "retained_cluster_count": len(retained_indices),
        "dominant_cluster_voxels": dominant_size,
        "dominant_cluster_volume_fraction": float(dominant_size / int(filtered_sizes.sum())),
        "minimum_component_voxels": minimum_voxels,
    }


def _write_volume(output_path: Path, mask_zyx: np.ndarray, reference: sitk.Image) -> None:
    result = sitk.GetImageFromArray(np.asarray(mask_zyx, dtype=np.uint8))
    result.CopyInformation(reference)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result, str(output_path))


def _slice_rows(case_id: str, prediction: np.ndarray, target: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slice_index, (pred_slice, target_slice) in enumerate(zip(prediction, target, strict=True)):
        pred_has_tumor, target_has_tumor = bool(pred_slice.any()), bool(target_slice.any())
        rows.append(
            {"case_id": case_id, "slice_index": slice_index}
            | binary_slice_metrics(pred_slice, target_slice)
            | {
                "pred_has_tumor": int(pred_has_tumor),
                "target_has_tumor": int(target_has_tumor),
                "slice_tp": int(pred_has_tumor and target_has_tumor),
                "slice_tn": int(not pred_has_tumor and not target_has_tumor),
                "slice_fp": int(pred_has_tumor and not target_has_tumor),
                "slice_fn": int(not pred_has_tumor and target_has_tumor),
            }
        )
    return rows


def _load_cases_for_tuning(cfg: dict[str, Any]) -> list[tuple[str, np.ndarray, np.ndarray, tuple[float, float, float]]]:
    """Read prediction/GT pairs once so a parameter search does not re-read disk."""
    prediction_paths = sorted(Path(cfg["PREDICTIONS_DIR"]).glob("*.nii.gz"))
    if not prediction_paths:
        raise FileNotFoundError(f"no .nii.gz predictions found in {cfg['PREDICTIONS_DIR']}")
    cases = []
    for prediction_path in tqdm(prediction_paths, desc="Loading tuning volumes", unit="case", dynamic_ncols=True):
        case_id = prediction_path.name.removesuffix(".nii.gz")
        target_path = Path(cfg["NIFTI_ROOT"]) / case_id / "mask.nii.gz"
        if not target_path.is_file():
            raise FileNotFoundError(f"missing reference mask for {case_id}: {target_path}")
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(str(prediction_path))) > 0
        target_image = sitk.ReadImage(str(target_path))
        target = sitk.GetArrayFromImage(target_image) > 0
        if prediction.shape != target.shape:
            raise ValueError(f"prediction and target shapes differ for {case_id}")
        cases.append((case_id, prediction, target, tuple(float(value) for value in target_image.GetSpacing())))
    return cases


def tune_postprocessing_parameters(
    cfg: dict[str, Any], tuning_cfg: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    """Grid-search post-processing settings against GT, ranking by mean 3D Dice.

    Results are persisted in ``OUTPUT_DIR/tuning_results.csv`` so the selected
    setting and its test-set leakage are transparent and reproducible.
    """
    cases = _load_cases_for_tuning(cfg)
    keys = (
        "CLOSING_RADIUS_MM",
        "OPENING_RADIUS_MM",
        "MIN_COMPONENT_VOLUME_MM3",
        "CONNECTIVITY",
        "COMPONENT_LINK_DISTANCE_MM",
        "SECONDARY_CLUSTER_MIN_VOLUME_RATIO",
    )
    combinations = list(itertools.product(*(tuning_cfg[key] for key in keys)))
    rows: list[dict[str, float | int]] = []
    for values in tqdm(combinations, desc="Tuning post-processing", unit="setting", dynamic_ncols=True):
        params = dict(zip(keys, values, strict=True))
        all_metrics = []
        for _, prediction, target, spacing_xyz in cases:
            cleaned, _ = postprocess_volume(
                prediction, spacing_xyz, float(params["CLOSING_RADIUS_MM"]),
                float(params["OPENING_RADIUS_MM"]), float(params["MIN_COMPONENT_VOLUME_MM3"]),
                int(params["CONNECTIVITY"]), float(params["COMPONENT_LINK_DISTANCE_MM"]),
                float(params["SECONDARY_CLUSTER_MIN_VOLUME_RATIO"]),
            )
            all_metrics.append(binary_volume_metrics(cleaned, target, spacing_xyz))
        row: dict[str, float | int] = {
            "closing_radius_mm": float(params["CLOSING_RADIUS_MM"]),
            "opening_radius_mm": float(params["OPENING_RADIUS_MM"]),
            "min_component_volume_mm3": float(params["MIN_COMPONENT_VOLUME_MM3"]),
            "connectivity": int(params["CONNECTIVITY"]),
            "component_link_distance_mm": float(params["COMPONENT_LINK_DISTANCE_MM"]),
            "secondary_cluster_min_volume_ratio": float(params["SECONDARY_CLUSTER_MIN_VOLUME_RATIO"]),
        }
        for metric in ("dice", "iou", "recall", "precision", "hd95", "assd"):
            row[f"{metric}_3d"] = finite_mean([float(item[metric]) for item in all_metrics])
        rows.append(row)
    # Prefer overlap; use physical boundary errors only to break an exact tie.
    rows.sort(key=lambda row: (-float(row["dice_3d"]), -float(row["iou_3d"]), float(row["hd95_3d"]), float(row["assd_3d"])))
    output_dir = Path(cfg["OUTPUT_DIR"])
    _write_csv(output_dir / "tuning_results.csv", rows)
    best = rows[0]
    (output_dir / "tuning_summary.json").write_text(
        json.dumps({"selection_metric": "mean_dice_3d", "best": best, "settings_tested": len(rows)}, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return dict(best), rows


def run_postprocessing(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Post-process every saved prediction and write masks plus metric reports."""
    predictions_dir = Path(cfg["PREDICTIONS_DIR"])
    nifti_root, output_dir = Path(cfg["NIFTI_ROOT"]), Path(cfg["OUTPUT_DIR"])
    dataset = str(cfg.get("DATASET", "")).casefold()
    if dataset not in {"nsclc_radiomics", "nsclc_radiogenomics"}:
        raise ValueError("DATASET must be 'nsclc_radiomics' or 'nsclc_radiogenomics'")
    prediction_paths = sorted(predictions_dir.glob("*.nii.gz"))
    if not prediction_paths:
        raise FileNotFoundError(f"no .nii.gz predictions found in {predictions_dir}")

    case_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    for prediction_path in tqdm(prediction_paths, desc="Post-processing volumes", unit="case", dynamic_ncols=True):
        case_id = prediction_path.name.removesuffix(".nii.gz")
        target_path = nifti_root / case_id / "mask.nii.gz"
        if not target_path.is_file():
            raise FileNotFoundError(f"missing reference mask for {case_id}: {target_path}")
        prediction_image, target_image = sitk.ReadImage(str(prediction_path)), sitk.ReadImage(str(target_path))
        prediction = sitk.GetArrayFromImage(prediction_image) > 0
        target = sitk.GetArrayFromImage(target_image) > 0
        if prediction.shape != target.shape:
            raise ValueError(f"prediction and target shapes differ for {case_id}: {prediction.shape} vs {target.shape}")
        spacing_xyz = tuple(float(value) for value in target_image.GetSpacing())
        postprocess_start = time.perf_counter()
        cleaned, components = postprocess_volume(
            prediction, spacing_xyz, float(cfg["CLOSING_RADIUS_MM"]), float(cfg["OPENING_RADIUS_MM"]),
            float(cfg["MIN_COMPONENT_VOLUME_MM3"]), int(cfg["CONNECTIVITY"]),
            float(cfg["COMPONENT_LINK_DISTANCE_MM"]), float(cfg["SECONDARY_CLUSTER_MIN_VOLUME_RATIO"]),
        )
        if dataset == "nsclc_radiogenomics":
            cleaned, lesion_selection = retain_weighted_stable_lesion(
                cleaned,
                int(cfg["CONNECTIVITY"]),
                float(cfg["RADIOGENOMICS_VOLUME_WEIGHT"]),
                float(cfg["RADIOGENOMICS_Z_THICKNESS_WEIGHT"]),
                float(cfg["RADIOGENOMICS_SLICE_STABILITY_WEIGHT"]),
            )
            components |= lesion_selection
        postprocess_seconds = time.perf_counter() - postprocess_start
        metrics = binary_volume_metrics(cleaned, target, spacing_xyz)
        _write_volume(output_dir / "predictions" / prediction_path.name, cleaned, target_image)
        case_rows.append(
            {
                "case_id": case_id,
                "original_pred_foreground_voxels": int(prediction.sum()),
                "pred_foreground_voxels": int(cleaned.sum()),
                "target_foreground_voxels": int(target.sum()),
                "pred_has_foreground": bool(cleaned.any()),
                "target_has_foreground": bool(target.any()),
                "spacing_x_mm": spacing_xyz[0], "spacing_y_mm": spacing_xyz[1], "spacing_z_mm": spacing_xyz[2],
                # Morphology and component-cluster selection only. NIfTI I/O
                # and metrics are excluded.
                "postprocess_seconds": postprocess_seconds,
            }
            | components
            | metrics
        )
        slice_rows.extend(_slice_rows(case_id, cleaned, target))

    slices_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in slice_rows:
        slices_by_case.setdefault(str(row["case_id"]), []).append(row)
    for case_row in case_rows:
        rows = slices_by_case[str(case_row["case_id"])]
        for metric in ("dice", "iou", "recall", "precision", "fp", "fn", "hd95", "assd"):
            case_row[f"{metric}_3d"] = case_row[metric]
        for metric in ("dice", "iou", "recall", "precision"):
            case_row[f"{metric}_2d"] = finite_mean([float(row[metric]) for row in rows])
        case_row.update({
            "slice_count": len(rows),
            "slice_tp_2d": sum(int(row["slice_tp"]) for row in rows),
            "slice_tn_2d": sum(int(row["slice_tn"]) for row in rows),
            "fp_2d": sum(int(row["slice_fp"]) for row in rows),
            "fn_2d": sum(int(row["slice_fn"]) for row in rows),
            "hd95_valid": int(np.isfinite(float(case_row["hd95"]))),
            "assd_valid": int(np.isfinite(float(case_row["assd"]))),
        })

    summary: dict[str, Any] = {"case_count": len(case_rows)}
    for key in ("dice", "iou", "recall", "precision", "hd95", "assd"):
        values = [float(row[key]) for row in case_rows]
        summary[f"{key}_3d"] = finite_mean(values)
        standard_deviation, interquartile_range = _finite_std_iqr(values)
        summary[f"{key}_3d_std"] = standard_deviation
        summary[f"{key}_3d_iqr"] = interquartile_range
        if key in {"hd95", "assd"}:
            summary[f"{key}_valid_cases"] = int(sum(np.isfinite(values)))
    for key in ("dice", "iou", "recall", "precision"):
        values = [float(row[key]) for row in slice_rows]
        summary[f"{key}_2d"] = finite_mean(values)
        standard_deviation, interquartile_range = _finite_std_iqr(values)
        summary[f"{key}_2d_std"] = standard_deviation
        summary[f"{key}_2d_iqr"] = interquartile_range
    summary.update({
        "fp_2d": sum(int(row["slice_fp"]) for row in slice_rows),
        "fn_2d": sum(int(row["slice_fn"]) for row in slice_rows),
        "pred_positive_cases": int(sum(bool(row["pred_has_foreground"]) for row in case_rows)),
        "target_positive_cases": int(sum(bool(row["target_has_foreground"]) for row in case_rows)),
        "pred_foreground_voxels": int(sum(int(row["pred_foreground_voxels"]) for row in case_rows)),
        "target_foreground_voxels": int(sum(int(row["target_foreground_voxels"]) for row in case_rows)),
        "pipeline": "binary 3D closing -> optional opening -> hole filling -> tiny-component filtering -> physical component clustering -> adaptive cluster retention",
        "closing_radius_mm": float(cfg["CLOSING_RADIUS_MM"]),
        "opening_radius_mm": float(cfg["OPENING_RADIUS_MM"]),
        "min_component_volume_mm3": float(cfg["MIN_COMPONENT_VOLUME_MM3"]),
        "connectivity": int(cfg["CONNECTIVITY"]),
        "component_link_distance_mm": float(cfg["COMPONENT_LINK_DISTANCE_MM"]),
        "secondary_cluster_min_volume_ratio": float(cfg["SECONDARY_CLUSTER_MIN_VOLUME_RATIO"]),
        "dataset": dataset,
    })
    if dataset == "nsclc_radiogenomics":
        summary.update({
            "radiogenomics_volume_weight": float(cfg["RADIOGENOMICS_VOLUME_WEIGHT"]),
            "radiogenomics_z_thickness_weight": float(cfg["RADIOGENOMICS_Z_THICKNESS_WEIGHT"]),
            "radiogenomics_slice_stability_weight": float(cfg["RADIOGENOMICS_SLICE_STABILITY_WEIGHT"]),
        })
    postprocess_times = np.asarray([float(row["postprocess_seconds"]) for row in case_rows], dtype=np.float64)
    summary.update({
        "postprocess_total_seconds": float(postprocess_times.sum()),
        "postprocess_mean_seconds_per_case": float(postprocess_times.mean()),
        "postprocess_std_seconds_per_case": float(postprocess_times.std(ddof=1)) if postprocess_times.size > 1 else 0.0,
        "postprocess_iqr_seconds_per_case": float(np.percentile(postprocess_times, 75) - np.percentile(postprocess_times, 25)),
        "postprocess_timing_scope": "3D morphology, hole filling, connected-component filtering, and adaptive component-cluster selection; excludes NIfTI I/O, metric calculation, and report writing",
    })
    _write_csv(output_dir / "per_case_metrics.csv", case_rows)
    _write_csv(output_dir / "slice_metrics.csv", slice_rows)
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return case_rows, summary


def main() -> None:
    run_cfg = dict(CFG)
    if bool(TUNING_CFG["RUN_TUNING"]):
        best, _ = tune_postprocessing_parameters(run_cfg, TUNING_CFG)
        run_cfg.update({
            "CLOSING_RADIUS_MM": best["closing_radius_mm"],
            "OPENING_RADIUS_MM": best["opening_radius_mm"],
            "MIN_COMPONENT_VOLUME_MM3": best["min_component_volume_mm3"],
            "CONNECTIVITY": best["connectivity"],
            "COMPONENT_LINK_DISTANCE_MM": best["component_link_distance_mm"],
            "SECONDARY_CLUSTER_MIN_VOLUME_RATIO": best["secondary_cluster_min_volume_ratio"],
        })
    _, summary = run_postprocessing(run_cfg)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
