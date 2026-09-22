"""Save one augmented, balanced batch from the TransUNet train loader.

Configuration is deliberately declared below rather than read from CLI.
Each sample is rendered as its central input slice followed by its target mask.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt

from data.transunet_dataset import (
    LungTumorSliceDataset,
    PatientAwareBalancedBatchSampler,
    build_loader,
    build_train_transform,
    read_case_ids,
)
from trains.train_nsclc_radiomics import CFG as TRAIN_CFG

CFG = {
    "OUTPUT_PATH": str(ROOT_DIR / "output" / "debug" / "train_loader_batch_preview.png"),
    "COLUMNS": 4,
}


def main() -> None:
    cfg = dict(TRAIN_CFG)
    dataset = LungTumorSliceDataset(
        Path(cfg["PROCESSED_ROOT"]),
        read_case_ids(Path(cfg["TRAIN_SPLIT"])),
        int(cfg["NUM_SLICES"]),
        build_train_transform(),
        hard_negative_radius=int(cfg["HARD_NEGATIVE_RADIUS"]),
    )
    sampler = PatientAwareBalancedBatchSampler(
        dataset,
        batch_size=int(cfg["BATCH_TRAIN"]),
        batches_per_epoch=1,
        seed=int(cfg["SAMPLER_SEED"]),
        positive_fraction=float(cfg["POSITIVE_FRACTION"]),
        hard_negative_fraction=float(cfg["HARD_NEGATIVE_FRACTION"]),
        easy_negative_fraction=float(cfg["EASY_NEGATIVE_FRACTION"]),
    )
    loader = build_loader(
        dataset,
        int(cfg["BATCH_TRAIN"]),
        False,
        int(cfg["NUM_WORKERS"]),
        bool(cfg["PIN_MEMORY"]),
        batch_sampler=sampler,
    )
    batch = next(iter(loader))
    item_groups = {
        (item.case_id, item.slice_index): item.sample_group for item in dataset.items
    }
    batch_size = int(batch["image"].shape[0])
    columns = int(CFG["COLUMNS"])
    rows = math.ceil(batch_size / columns)
    figure, axes = plt.subplots(rows, columns * 2, figsize=(columns * 5, rows * 3.2))
    axes = axes.reshape(rows, columns * 2)
    centre_channel = int(batch["image"].shape[1]) // 2

    for sample_idx in range(rows * columns):
        image_axis = axes.flat[sample_idx * 2]
        mask_axis = axes.flat[sample_idx * 2 + 1]
        if sample_idx >= batch_size:
            image_axis.axis("off")
            mask_axis.axis("off")
            continue
        case_id = str(batch["case_id"][sample_idx])
        slice_index = int(batch["slice_index"][sample_idx])
        group = item_groups[(case_id, slice_index)]
        image_axis.imshow(batch["image"][sample_idx, centre_channel].numpy(), cmap="gray", vmin=0, vmax=1)
        image_axis.set_title(f"{case_id} | z={slice_index}\n{group}", fontsize=8)
        mask_axis.imshow(batch["mask"][sample_idx, 0].numpy(), cmap="magma", vmin=0, vmax=1)
        mask_axis.set_title("label", fontsize=8)
        image_axis.axis("off")
        mask_axis.axis("off")

    figure.suptitle(
        f"Augmented train-loader batch | {batch_size} samples | "
        f"context={cfg['NUM_SLICES']} slice(s)",
        fontsize=14,
    )
    figure.tight_layout()
    output_path = Path(CFG["OUTPUT_PATH"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
