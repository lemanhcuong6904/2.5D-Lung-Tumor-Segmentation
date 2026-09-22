"""Create reproducible stratified train/validation/test splits for NSCLC-Radiogenomics.

The 144 cases are stratified by tumour volume tertile and normalized
cranio-caudal tumour location (lower/middle/upper).  The script assigns 15
patients to test, 15 to validation, and every remaining patient to training.
All configuration is kept below; no command-line arguments are used.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET_ROOT = Path("data/nsclc_radiogenomics")
NIFTI_ROOT = DATASET_ROOT / "nifti"
OUTPUT_DIR = DATASET_ROOT / "config"
FIGURE_DIR = OUTPUT_DIR / "figures"
# Set to 144 to refuse a split until every collection case has been processed.
# ``None`` uses every case that currently has a valid mask NIfTI.
EXPECTED_CASE_COUNT: int | None = None
TEST_COUNT = 15
VAL_COUNT = 15
SEED = 42
OVERWRITE = False
SAVE_FIGURES = True
FIGURE_FORMATS = ("png", "pdf")
FIGURE_DPI = 300

VOLUME_LEVELS = ("small", "medium", "large")
LOCATION_LEVELS = ("lower", "middle", "upper")
SPLIT_LEVELS = ("train", "val", "test")
SPLIT_COLORS = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
PLOT_SPLIT_ORDER = ("Train", "Validation", "Test")
PLOT_VOLUME_ORDER = ("Small", "Medium", "Large")
PLOT_LOCATION_ORDER = ("Lower", "Middle", "Upper")
PLOT_PALETTE = {"Train": "#0072B2", "Validation": "#D55E00", "Test": "#009E73"}


def largest_remainder_allocation(total: int, group_sizes: dict[str, int]) -> dict[str, int]:
    """Allocate a fixed total proportionally without exceeding group sizes."""
    available = sum(group_sizes.values())
    if not 0 <= total <= available:
        raise ValueError(f"cannot allocate {total} cases from {available}")
    quotas = {name: total * size / available for name, size in group_sizes.items()}
    allocation = {name: min(size, int(np.floor(quotas[name]))) for name, size in group_sizes.items()}
    remaining = total - sum(allocation.values())
    for name in sorted(
        group_sizes,
        key=lambda item: (quotas[item] - allocation[item], group_sizes[item], item),
        reverse=True,
    ):
        if remaining == 0:
            break
        if allocation[name] < group_sizes[name]:
            allocation[name] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("stratified allocation did not reach the requested size")
    return allocation


def tumour_features(mask_path: Path) -> tuple[float, float]:
    """Return tumour volume in mm³ and its normalized Z-centroid."""
    image = sitk.ReadImage(str(mask_path))
    mask = sitk.GetArrayFromImage(image) > 0  # NumPy order: z, y, x
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        raise ValueError("tumour mask is empty")
    volume_mm3 = float(len(coordinates) * np.prod(image.GetSpacing()))
    depth = mask.shape[0]
    z_normalized = 0.5 if depth == 1 else float(coordinates[:, 0].mean() / (depth - 1))
    return volume_mm3, z_normalized


def volume_group(volume_mm3: float, q33: float, q67: float) -> str:
    if volume_mm3 < q33:
        return "small"
    if volume_mm3 > q67:
        return "large"
    return "medium"


def location_group(z_normalized: float) -> str:
    if z_normalized < 1 / 3:
        return "lower"
    if z_normalized < 2 / 3:
        return "middle"
    return "upper"


def write_case_list(path: Path, case_ids: list[str]) -> None:
    if path.exists() and path.stat().st_size and not OVERWRITE:
        raise FileExistsError(f"refusing to replace non-empty file: {path}")
    path.write_text("\n".join(case_ids) + "\n", encoding="utf-8")


def discover_cases() -> list[str]:
    case_ids = sorted(
        path.name for path in NIFTI_ROOT.iterdir() if (path / "mask.nii.gz").is_file()
    )
    if EXPECTED_CASE_COUNT is not None and len(case_ids) != EXPECTED_CASE_COUNT:
        raise FileNotFoundError(
            f"expected {EXPECTED_CASE_COUNT} mask NIfTI files under {NIFTI_ROOT}, "
            f"but found {len(case_ids)}. Finish preprocessing before creating the split."
        )
    return case_ids


def create_split(case_ids: list[str]) -> tuple[dict[str, list[str]], list[dict[str, object]]]:
    features = {case_id: tumour_features(NIFTI_ROOT / case_id / "mask.nii.gz") for case_id in case_ids}
    volumes = np.array([features[case_id][0] for case_id in case_ids])
    q33, q67 = np.quantile(volumes, (0.33, 0.67))
    strata = {f"{volume}_{location}": [] for volume in VOLUME_LEVELS for location in LOCATION_LEVELS}
    rows: list[dict[str, object]] = []
    for case_id in case_ids:
        volume_mm3, z_normalized = features[case_id]
        volume_bin = volume_group(volume_mm3, q33, q67)
        location_bin = location_group(z_normalized)
        stratum = f"{volume_bin}_{location_bin}"
        strata[stratum].append(case_id)
        rows.append({
            "case_id": case_id,
            "volume_mm3": volume_mm3,
            "volume_ml": volume_mm3 / 1000.0,
            "z_normalized": z_normalized,
            "volume_group": volume_bin,
            "location_group": location_bin,
            "stratum": stratum,
        })

    rng = np.random.default_rng(SEED)
    for members in strata.values():
        rng.shuffle(members)

    test_counts = largest_remainder_allocation(TEST_COUNT, {name: len(members) for name, members in strata.items()})
    remaining = {name: members[test_counts[name] :] for name, members in strata.items()}
    val_counts = largest_remainder_allocation(VAL_COUNT, {name: len(members) for name, members in remaining.items()})

    splits = {"train": [], "val": [], "test": []}
    for name, members in strata.items():
        test = members[: test_counts[name]]
        val = remaining[name][: val_counts[name]]
        train = remaining[name][val_counts[name] :]
        splits["test"].extend(test)
        splits["val"].extend(val)
        splits["train"].extend(train)

    for split, members in splits.items():
        if split == "test" and len(members) != TEST_COUNT:
            raise RuntimeError("test size does not match TEST_COUNT")
        if split == "val" and len(members) != VAL_COUNT:
            raise RuntimeError("validation size does not match VAL_COUNT")
        if split == "train" and len(members) != len(case_ids) - TEST_COUNT - VAL_COUNT:
            raise RuntimeError("training size is inconsistent")
        for row in rows:
            if row["case_id"] in members:
                row["split"] = split
    return {name: sorted(members) for name, members in splits.items()}, rows


def write_metadata(rows: list[dict[str, object]]) -> None:
    fields = ("case_id", "split", "volume_mm3", "volume_ml", "z_normalized", "volume_group", "location_group", "stratum")
    with (OUTPUT_DIR / "split_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["case_id"])))


def configure_figure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Serif")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": FIGURE_DPI, "axes.titleweight": "bold"})


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in FIGURE_FORMATS:
        figure.savefig(FIGURE_DIR / f"{name}.{extension}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def figure_table(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Format split metadata for figures using the Radiomics paper convention."""
    table = pd.DataFrame.from_records(rows)
    table["volume_ml"] = pd.to_numeric(table["volume_ml"])
    table["z_normalized"] = pd.to_numeric(table["z_normalized"])
    table["split"] = table["split"].map({"train": "Train", "val": "Validation", "test": "Test"})
    table["volume_group"] = table["volume_group"].str.capitalize()
    table["location_group"] = table["location_group"].str.capitalize()
    quantiles = tuple(np.quantile(table["volume_ml"], (0.33, 0.67)))
    return table, (float(quantiles[0]), float(quantiles[1]))


def plot_distributions(table: pd.DataFrame, quantiles: tuple[float, float]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    volume_bins = np.geomspace(table.volume_ml.min(), table.volume_ml.max(), 30)
    for split in PLOT_SPLIT_ORDER:
        subset = table.loc[table.split == split]
        axes[0].hist(subset.volume_ml, bins=volume_bins, density=True, histtype="step", linewidth=1.8, color=PLOT_PALETTE[split], label=f"{split} (n={len(subset)})")
        axes[1].hist(subset.z_normalized, bins=np.linspace(0, 1, 21), density=True, histtype="step", linewidth=1.8, color=PLOT_PALETTE[split], label=f"{split} (n={len(subset)})")
    for value, label in zip(quantiles, ("33rd percentile", "67th percentile")):
        axes[0].axvline(value, color="black", linestyle="--", linewidth=1, label=label)
    for value in (1 / 3, 2 / 3):
        axes[1].axvline(value, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Tumor volume (mL)", ylabel="Density", title="A. Tumor-volume distribution")
    axes[1].set(xlabel="Normalized cranio-caudal centroid, $z_n$", ylabel="Density", title="B. Tumor-centroid distribution")
    axes[0].set_xscale("log")
    axes[1].set_xlim(0, 1)
    axes[0].legend(fontsize=8, frameon=True)
    axes[1].legend(fontsize=8, frameon=True)
    save_figure(figure, "stratification_distributions")


def plot_strata_heatmaps(table: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True, sharey=True)
    matrices: dict[str, pd.DataFrame] = {}
    maximum = 0
    for split in PLOT_SPLIT_ORDER:
        matrix = pd.crosstab(
            pd.Categorical(table.loc[table.split == split, "volume_group"], categories=PLOT_VOLUME_ORDER, ordered=True),
            pd.Categorical(table.loc[table.split == split, "location_group"], categories=PLOT_LOCATION_ORDER, ordered=True),
            dropna=False,
        ).reindex(index=PLOT_VOLUME_ORDER, columns=PLOT_LOCATION_ORDER, fill_value=0)
        matrices[split] = matrix
        maximum = max(maximum, int(matrix.to_numpy().max()))
    for axis, split in zip(axes, PLOT_SPLIT_ORDER):
        sns.heatmap(matrices[split], annot=True, fmt="d", cmap="Blues", vmin=0, vmax=maximum, square=True, cbar=axis is axes[-1], linewidths=0.5, linecolor="white", ax=axis)
        axis.set(title=f"{split} (n={(table.split == split).sum()})", xlabel="Cranio-caudal location", ylabel="Tumor volume" if axis is axes[0] else "")
    save_figure(figure, "stratification_9_strata")


def plot_joint_distribution(table: pd.DataFrame, quantiles: tuple[float, float]) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    for split in PLOT_SPLIT_ORDER:
        subset = table[table.split == split]
        axis.scatter(subset.z_normalized, subset.volume_ml, s=18, alpha=0.65, color=PLOT_PALETTE[split], label=f"{split} (n={len(subset)})", edgecolors="none")
    for value in quantiles:
        axis.axhline(value, color="black", linestyle="--", linewidth=1)
    for value in (1 / 3, 2 / 3):
        axis.axvline(value, color="black", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_yscale("log")
    axis.set(xlabel="Normalized cranio-caudal centroid, $z_n$", ylabel="Tumor volume (mL)", title="Joint distribution used for stratification")
    axis.legend(fontsize=8, frameon=True)
    save_figure(figure, "stratification_joint_distribution")


def create_figures(rows: list[dict[str, object]]) -> None:
    configure_figure_style()
    table, quantiles = figure_table(rows)
    plot_distributions(table, quantiles)
    plot_strata_heatmaps(table)
    plot_joint_distribution(table, quantiles)
    table.to_csv(FIGURE_DIR / "stratification_summary.csv", index=False)


def main() -> int:
    case_ids = discover_cases()
    splits, rows = create_split(case_ids)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_case_list(OUTPUT_DIR / "train.txt", splits["train"])
    write_case_list(OUTPUT_DIR / "val.txt", splits["val"])
    write_case_list(OUTPUT_DIR / "test.txt", splits["test"])
    write_metadata(rows)
    if SAVE_FIGURES:
        create_figures(rows)

    print(f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print("volume tertiles (mm3):", np.quantile([row["volume_mm3"] for row in rows], (0.33, 0.67)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
