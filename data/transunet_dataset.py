"""Slice-indexed 2D and 2.5D PNG datasets for binary lung tumor segmentation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset


@dataclass(frozen=True)
class SliceItem:
    """One centre slice and the complete ordered image stack for its case."""

    case_id: str
    slice_index: int
    image_paths: tuple[Path, ...]
    label_path: Path
    sample_group: str


def read_case_ids(split_path: Path) -> list[str]:
    """Return non-empty case identifiers from a one-ID-per-line split file."""
    return [
        line.strip()
        for line in Path(split_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_train_transform() -> A.Compose:
    """Build paired spatial and intensity augmentation for multi-slice CT input."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.3),
            A.Affine(
                rotate=(-25, 25),
                scale=(0.85, 1.15),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                interpolation=1,
                mask_interpolation=0,
                fill=0.0,
                fill_mask=0,
                p=0.7,
            ),
            A.ElasticTransform(
                alpha=1.0,
                sigma=30.0,
                interpolation=1,
                mask_interpolation=0,
                fill=0.0,
                fill_mask=0,
                p=0.2,
            ),
            # A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # A.RandomBrightnessContrast(
            #     brightness_limit=0.0, contrast_limit=(-0.15, 0.15), p=0.3
            # ),
        ]
    )


def build_eval_transform() -> None:
    """Validation and inference intentionally do not alter image geometry or values."""
    return None


def _read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _read_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0).astype(
        np.float32
    )


def _build_slice_items(
    processed_root: Path, case_ids: Sequence[str]
) -> list[SliceItem]:
    items: list[SliceItem] = []
    for case_id in case_ids:
        image_dir = processed_root / case_id / "images"
        label_dir = processed_root / case_id / "labels"
        image_paths = tuple(sorted(image_dir.glob("*.png")))
        label_paths = tuple(sorted(label_dir.glob("*.png")))
        if (
            not image_paths
            or len(image_paths) != len(label_paths)
            or [path.name for path in image_paths]
            != [path.name for path in label_paths]
        ):
            raise FileNotFoundError(
                f"missing paired PNG image/label files for case {case_id}"
            )
        positive_indices = [
            index
            for index, label_path in enumerate(label_paths)
            if _read_mask(label_path).any()
        ]
        for slice_index, label_path in enumerate(label_paths):
            if slice_index in positive_indices:
                group = "positive"
            elif (
                positive_indices
                and min(
                    abs(slice_index - positive_index)
                    for positive_index in positive_indices
                )
                <= 2
            ):
                group = "hard_negative"
            else:
                group = "easy_negative"
            items.append(
                SliceItem(case_id, slice_index, image_paths, label_path, group)
            )
    if not items:
        raise ValueError("no paired slices found for the requested cases")
    return items


class LungTumorSliceDataset(Dataset[dict[str, object]]):
    """Return a centre label and one, three, or five clamped axial CT slices."""

    def __init__(
        self,
        processed_root: Path,
        case_ids: Sequence[str],
        num_slices: int,
        transform: A.Compose | None = None,
        hard_negative_radius: int = 2,
    ) -> None:
        if num_slices < 1 or num_slices % 2 == 0:
            raise ValueError("num_slices must be a positive odd integer")
        if hard_negative_radius < 0:
            raise ValueError("hard_negative_radius must be non-negative")
        self.items = _build_slice_items(Path(processed_root), case_ids)
        if hard_negative_radius != 2:
            self._reclassify_hard_negatives(hard_negative_radius)
        self.num_slices = num_slices
        self.transform = transform

    def _reclassify_hard_negatives(self, radius: int) -> None:
        """Reclassify negatives with a configurable central-slice distance radius."""
        by_case: dict[str, list[SliceItem]] = {}
        for item in self.items:
            by_case.setdefault(item.case_id, []).append(item)
        rebuilt: list[SliceItem] = []
        for case_items in by_case.values():
            positive_indices = [
                item.slice_index
                for item in case_items
                if item.sample_group == "positive"
            ]
            for item in case_items:
                group = item.sample_group
                if group != "positive":
                    group = (
                        "hard_negative"
                        if positive_indices
                        and min(
                            abs(item.slice_index - index) for index in positive_indices
                        )
                        <= radius
                        else "easy_negative"
                    )
                rebuilt.append(
                    SliceItem(
                        item.case_id,
                        item.slice_index,
                        item.image_paths,
                        item.label_path,
                        group,
                    )
                )
        self.items = rebuilt

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        half = self.num_slices // 2
        context_indices = [
            min(max(item.slice_index + offset, 0), len(item.image_paths) - 1)
            for offset in range(-half, half + 1)
        ]
        image_chw = np.stack(
            [_read_gray(item.image_paths[i]) for i in context_indices], axis=0
        )
        mask = _read_mask(item.label_path)
        if self.transform is not None:
            transformed = self.transform(image=np.moveaxis(image_chw, 0, -1), mask=mask)
            image_chw = np.moveaxis(
                np.asarray(transformed["image"], dtype=np.float32), -1, 0
            )
            mask = np.asarray(transformed["mask"], dtype=np.float32)
        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(image_chw, dtype=np.float32)
            ),
            "mask": torch.from_numpy(
                np.ascontiguousarray(mask[None], dtype=np.float32)
            ),
            "case_id": item.case_id,
            "slice_index": item.slice_index,
        }


class PatientAwareBalancedBatchSampler(BatchSampler):
    """Sample central slices with a fixed positive/hard/easy mix and uniform patients."""

    def __init__(
        self,
        dataset: LungTumorSliceDataset,
        batch_size: int,
        batches_per_epoch: int | None = None,
        seed: int = 42,
        positive_fraction: float = 0.50,
        hard_negative_fraction: float = 0.25,
        easy_negative_fraction: float = 0.25,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch or math.ceil(
            len(dataset) / batch_size
        )
        self.seed = seed
        self.epoch = 0
        self.group_weights = {
            "positive": positive_fraction,
            "hard_negative": hard_negative_fraction,
            "easy_negative": easy_negative_fraction,
        }
        if any(
            weight < 0 for weight in self.group_weights.values()
        ) or not math.isclose(
            sum(self.group_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(
                "positive, hard-negative, and easy-negative fractions must be non-negative and sum to 1"
            )
        self.indices: dict[str, dict[str, list[int]]] = {
            group: {} for group in self.group_weights
        }
        for index, item in enumerate(dataset.items):
            self.indices[item.sample_group].setdefault(item.case_id, []).append(index)
        missing = [group for group, patients in self.indices.items() if not patients]
        if missing:
            raise ValueError(
                f"cannot create balanced sampler; missing groups: {', '.join(missing)}"
            )

    def _counts(self) -> dict[str, int]:
        raw = {
            group: self.batch_size * weight
            for group, weight in self.group_weights.items()
        }
        counts = {group: int(value) for group, value in raw.items()}
        for group, _ in sorted(
            raw.items(), key=lambda pair: pair[1] - int(pair[1]), reverse=True
        )[: self.batch_size - sum(counts.values())]:
            counts[group] += 1
        return counts

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        counts = self._counts()
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for group, count in counts.items():
                patients = list(self.indices[group])
                for _ in range(count):
                    case_id = rng.choice(patients)
                    batch.append(rng.choice(self.indices[group][case_id]))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch


class TumorCoveringBatchSampler(BatchSampler):
    """Limit an epoch while including every positive slice exactly once.

    The remaining slots are filled with randomly selected negative slices without
    replacement. This is not class-balanced sampling: it only guarantees that
    the limited epoch does not discard any tumor-containing slice.
    """

    def __init__(
        self,
        dataset: LungTumorSliceDataset,
        batch_size: int,
        batches_per_epoch: int,
        seed: int = 42,
    ) -> None:
        if batch_size < 1 or batches_per_epoch < 1:
            raise ValueError("batch_size and batches_per_epoch must be positive")
        normal_batches = math.ceil(len(dataset) / batch_size)
        if batches_per_epoch > normal_batches:
            raise ValueError(
                "TRAIN_BATCHES_PER_EPOCH cannot exceed the normal number of batches"
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        self.positive_indices = [
            index
            for index, item in enumerate(dataset.items)
            if item.sample_group == "positive"
        ]
        self.negative_indices = [
            index
            for index, item in enumerate(dataset.items)
            if item.sample_group != "positive"
        ]
        self.items_per_epoch = min(len(dataset), batch_size * batches_per_epoch)
        if len(self.positive_indices) > self.items_per_epoch:
            raise ValueError(
                "TRAIN_BATCHES_PER_EPOCH is too small to include every positive slice"
            )

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        positives = list(self.positive_indices)
        negatives = list(self.negative_indices)
        rng.shuffle(positives)
        rng.shuffle(negatives)

        selected: list[int | None] = [None] * self.items_per_epoch
        positions = list(range(self.items_per_epoch))
        rng.shuffle(positions)
        for position, index in zip(positions, positives):
            selected[position] = index
        remaining_positions = positions[len(positives) :]
        used = {index for index in selected if index is not None}
        remaining = iter([index for index in negatives if index not in used])
        for position in remaining_positions:
            selected[position] = next(remaining)

        indices = [index for index in selected if index is not None]
        for start in range(0, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(self.items_per_epoch / self.batch_size)


def build_loader(
    dataset: Dataset[dict[str, object]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    batch_sampler: BatchSampler | None = None,
) -> DataLoader:
    """Create a loader with worker-only settings enabled when workers are used."""
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
