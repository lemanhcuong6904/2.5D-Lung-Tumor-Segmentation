"""Create reproducible patient-level train/validation/test split files.

The fixed test cohort is LUNG-001 through LUNG-042. Ten cases without a
native lung SEG are fixed in the unused cohort. From the other 370 patients,
the script stratifies by 3 tumor-volume groups and 3 normalized cranio-caudal
groups, then assigns 260 patients to train and 60 to validation. The remaining
50 cases plus the 10 excluded cases are deliberately left unlisted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

ALL_CASES = tuple(f"LUNG-{index:03d}" for index in range(1, 423))
TEST_CASES = tuple(f"LUNG-{index:03d}" for index in range(1, 43))
POOL_CASES = tuple(case_id for case_id in ALL_CASES if case_id not in TEST_CASES)
LUNG_SEGMENT_MISSING_CASES = (
    "LUNG-050",
    "LUNG-067",
    "LUNG-115",
    "LUNG-119",
    "LUNG-121",
    "LUNG-149",
    "LUNG-158",
    "LUNG-210",
    "LUNG-219",
    "LUNG-273",
)
STRATIFIED_POOL_CASES = tuple(
    case_id for case_id in POOL_CASES if case_id not in LUNG_SEGMENT_MISSING_CASES
)

TRAIN_COUNT = 260
VAL_COUNT = 60
SEED = 42
# NSCLC-Radiomics only.  Change these values here instead of passing CLI args.
DATASET_ROOT = Path("data/nsclc-radiomics")
NIFTI_ROOT = DATASET_ROOT / "nifti"
OUTPUT_DIR = DATASET_ROOT / "config"
OVERWRITE = False


def largest_remainder_allocation(total: int, group_sizes: dict[str, int]) -> dict[str, int]:
    """Allocate ``total`` proportionally without exceeding any group size."""
    available = sum(group_sizes.values())
    if total < 0 or total > available:
        raise ValueError(f"cannot allocate {total} patients from {available}")
    if not available:
        return {group: 0 for group in group_sizes}

    quotas = {group: total * size / available for group, size in group_sizes.items()}
    allocation = {
        group: min(size, int(np.floor(quotas[group])))
        for group, size in group_sizes.items()
    }
    remaining = total - sum(allocation.values())
    for group in sorted(
        group_sizes,
        key=lambda item: (quotas[item] - allocation[item], group_sizes[item], item),
        reverse=True,
    ):
        if not remaining:
            break
        if allocation[group] < group_sizes[group]:
            allocation[group] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("allocation did not reach requested total")
    return allocation


def tumor_features(mask_path: Path) -> tuple[float, float]:
    """Return tumor volume (mm3) and normalized superior-inferior centroid."""
    image = sitk.ReadImage(str(mask_path))
    mask = sitk.GetArrayFromImage(image) > 0  # NumPy order: z, y, x
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        raise ValueError("tumor mask is empty")

    volume_mm3 = float(len(coordinates) * np.prod(image.GetSpacing()))
    depth = mask.shape[0]
    z_normalized = 0.5 if depth == 1 else float(coordinates[:, 0].mean() / (depth - 1))
    return volume_mm3, z_normalized


def volume_group(volume: float, quantile_33: float, quantile_67: float) -> str:
    if volume < quantile_33:
        return "small"
    if volume > quantile_67:
        return "large"
    return "medium"


def location_group(z_normalized: float) -> str:
    """Classify normalized DICOM-z: high z is cranial (upper lung)."""
    if z_normalized < 1 / 3:
        return "lower"
    if z_normalized < 2 / 3:
        return "middle"
    return "upper"


def write_case_list(path: Path, case_ids: list[str], overwrite: bool) -> None:
    if path.exists() and path.stat().st_size and not overwrite:
        raise FileExistsError(f"refusing to replace non-empty file: {path}")
    path.write_text("\n".join(case_ids) + "\n", encoding="utf-8")


def create_split(nifti_root: Path, seed: int) -> tuple[list[str], list[str], list[str], list[str], dict[str, int]]:
    missing = [case_id for case_id in ALL_CASES if not (nifti_root / case_id / "mask.nii.gz").is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} tumor-mask NIfTI files, e.g. {', '.join(missing[:10])}"
        )

    features = {
        case_id: tumor_features(nifti_root / case_id / "mask.nii.gz")
        for case_id in STRATIFIED_POOL_CASES
    }
    volumes = np.array([features[case_id][0] for case_id in STRATIFIED_POOL_CASES])
    quantile_33, quantile_67 = np.quantile(volumes, [0.33, 0.67])

    strata: dict[str, list[str]] = {f"{volume}_{location}": [] for volume in ("small", "medium", "large") for location in ("upper", "middle", "lower")}
    for case_id in STRATIFIED_POOL_CASES:
        volume, z_normalized = features[case_id]
        strata[f"{volume_group(volume, quantile_33, quantile_67)}_{location_group(z_normalized)}"].append(case_id)

    rng = np.random.default_rng(seed)
    for case_ids in strata.values():
        rng.shuffle(case_ids)

    selected_counts = largest_remainder_allocation(
        TRAIN_COUNT + VAL_COUNT, {name: len(case_ids) for name, case_ids in strata.items()}
    )
    selected_by_stratum = {
        name: case_ids[: selected_counts[name]] for name, case_ids in strata.items()
    }
    val_counts = largest_remainder_allocation(
        VAL_COUNT, {name: len(case_ids) for name, case_ids in selected_by_stratum.items()}
    )

    train: list[str] = []
    val: list[str] = []
    unused: list[str] = list(LUNG_SEGMENT_MISSING_CASES)
    for name, case_ids in strata.items():
        selected = selected_by_stratum[name]
        val.extend(selected[: val_counts[name]])
        train.extend(selected[val_counts[name] :])
        unused.extend(case_ids[selected_counts[name] :])

    if len(train) != TRAIN_COUNT or len(val) != VAL_COUNT or len(unused) != 60:
        raise RuntimeError("split sizes do not match the requested 260/60/60 allocation")
    return sorted(train), sorted(val), list(TEST_CASES), sorted(unused), {
        name: len(case_ids) for name, case_ids in strata.items()
    }


def main() -> int:
    train, val, test, unused, strata = create_split(NIFTI_ROOT, SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_case_list(OUTPUT_DIR / "train.txt", train, OVERWRITE)
    write_case_list(OUTPUT_DIR / "val.txt", val, OVERWRITE)
    write_case_list(OUTPUT_DIR / "test.txt", test, OVERWRITE)

    print(
        f"train={len(train)}, val={len(val)}, test={len(test)}, unused={len(unused)} "
        f"(native-lung-SEG excluded={len(LUNG_SEGMENT_MISSING_CASES)})"
    )
    print("strata in 370-case pool:", dict(sorted(strata.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
