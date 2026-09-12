"""Create publication-ready figures describing the patient-level data split."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk


ALL_CASES = [f"LUNG-{index:03d}" for index in range(1, 423)]
TEST_CASES = set(ALL_CASES[:42])
LUNG_SEGMENT_MISSING_CASES = {
    "LUNG-050", "LUNG-067", "LUNG-115", "LUNG-119", "LUNG-121",
    "LUNG-149", "LUNG-158", "LUNG-210", "LUNG-219", "LUNG-273",
}
VOLUME_ORDER = ["Small", "Medium", "Large"]
LOCATION_ORDER = ["Lower", "Middle", "Upper"]
SPLIT_ORDER = ["Train", "Validation", "Test"]
PALETTE = {"Train": "#0072B2", "Validation": "#D55E00", "Test": "#009E73"}


def read_case_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def tumor_features(mask_path: Path) -> tuple[float, float]:
    image = sitk.ReadImage(str(mask_path))
    mask = sitk.GetArrayFromImage(image) > 0
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        raise ValueError(f"empty tumor mask: {mask_path}")
    volume_ml = float(len(coordinates) * np.prod(image.GetSpacing()) / 1000.0)
    depth = mask.shape[0]
    z_normalized = 0.5 if depth == 1 else float(coordinates[:, 0].mean() / (depth - 1))
    return volume_ml, z_normalized


def build_table(nifti_root: Path, config_root: Path) -> tuple[pd.DataFrame, tuple[float, float]]:
    train = read_case_ids(config_root / "train.txt")
    validation = read_case_ids(config_root / "val.txt")
    test = read_case_ids(config_root / "test.txt")
    assigned = train | validation | test
    records = []
    for case_id in ALL_CASES:
        volume_ml, z_normalized = tumor_features(nifti_root / case_id / "mask.nii.gz")
        split = "Train" if case_id in train else "Validation" if case_id in validation else "Test" if case_id in test else "Unused"
        records.append({"case_id": case_id, "split": split, "volume_ml": volume_ml, "z_normalized": z_normalized})

    table = pd.DataFrame.from_records(records)
    pool = table[~table.case_id.isin(TEST_CASES | LUNG_SEGMENT_MISSING_CASES)]
    q33, q67 = np.quantile(pool.volume_ml, [0.33, 0.67])
    table["volume_group"] = pd.cut(
        table.volume_ml,
        bins=[-np.inf, q33, q67, np.inf],
        labels=VOLUME_ORDER,
        include_lowest=True,
    )
    table["location_group"] = pd.cut(
        table.z_normalized,
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=LOCATION_ORDER,
        include_lowest=True,
    )
    if len(assigned) != 362:
        raise ValueError("split files must contain 362 unique assigned cases")
    return table, (float(q33), float(q67))


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Serif")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titleweight": "bold"})


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_distributions(table: pd.DataFrame, quantiles: tuple[float, float], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    volume_bins = np.geomspace(table.volume_ml.min(), table.volume_ml.max(), 30)
    for split in SPLIT_ORDER:
        data = table.loc[table.split == split, "volume_ml"]
        axes[0].hist(data, bins=volume_bins, density=True, histtype="step", linewidth=1.8, color=PALETTE[split], label=f"{split} (n={len(data)})")
        axes[1].hist(table.loc[table.split == split, "z_normalized"], bins=np.linspace(0, 1, 21), density=True, histtype="step", linewidth=1.8, color=PALETTE[split], label=f"{split} (n={len(data)})")
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
    save_figure(figure, output_dir, "stratification_distributions")


def plot_strata_heatmaps(table: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True, sharey=True)
    maximum = 0
    matrices = {}
    for split in ("Train", "Validation", "Test"):
        matrix = pd.crosstab(
            pd.Categorical(table.loc[table.split == split, "volume_group"], categories=VOLUME_ORDER, ordered=True),
            pd.Categorical(table.loc[table.split == split, "location_group"], categories=LOCATION_ORDER, ordered=True),
            dropna=False,
        ).reindex(index=VOLUME_ORDER, columns=LOCATION_ORDER, fill_value=0)
        matrices[split] = matrix
        maximum = max(maximum, int(matrix.to_numpy().max()))
    for axis, split in zip(axes, ("Train", "Validation", "Test")):
        sns.heatmap(matrices[split], annot=True, fmt="d", cmap="Blues", vmin=0, vmax=maximum, square=True, cbar=axis is axes[-1], linewidths=0.5, linecolor="white", ax=axis)
        axis.set(title=f"{split} (n={(table.split == split).sum()})", xlabel="Cranio-caudal location", ylabel="Tumor volume" if axis is axes[0] else "")
    save_figure(figure, output_dir, "stratification_9_strata")


def plot_joint_distribution(table: pd.DataFrame, quantiles: tuple[float, float], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    for split in SPLIT_ORDER:
        subset = table[table.split == split]
        axis.scatter(subset.z_normalized, subset.volume_ml, s=18, alpha=0.65, color=PALETTE[split], label=f"{split} (n={len(subset)})", edgecolors="none")
    for value in quantiles:
        axis.axhline(value, color="black", linestyle="--", linewidth=1)
    for value in (1 / 3, 2 / 3):
        axis.axvline(value, color="black", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_yscale("log")
    axis.set(
        xlabel="Normalized cranio-caudal centroid, $z_n$",
        ylabel="Tumor volume (mL)",
        title="Joint distribution used for stratification",
    )
    axis.legend(fontsize=8, frameon=True)
    save_figure(figure, output_dir, "stratification_joint_distribution")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nifti-root", type=Path, default=Path("data/nsclc-radiomics/nifti"))
    parser.add_argument("--config-root", type=Path, default=Path("data/nsclc-radiomics/config"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    table, quantiles = build_table(args.nifti_root, args.config_root)
    plot_distributions(table, quantiles, args.output_dir)
    plot_strata_heatmaps(table, args.output_dir)
    plot_joint_distribution(table, quantiles, args.output_dir)
    table.to_csv(args.output_dir / "stratification_summary.csv", index=False)
    print(f"Saved 3 figures and stratification_summary.csv to {args.output_dir}")
    print(f"Volume thresholds: q33={quantiles[0]:.3f} mL, q67={quantiles[1]:.3f} mL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
