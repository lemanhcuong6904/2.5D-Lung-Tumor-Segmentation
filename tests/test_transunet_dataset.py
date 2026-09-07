from pathlib import Path

import pytest
import torch
from PIL import Image

from data.transunet_dataset import LungTumorSliceDataset, PatientAwareBalancedBatchSampler


def _write_case(root: Path, case_id: str, image_values: list[int], label_values: list[int]) -> None:
    image_dir = root / case_id / "images"
    label_dir = root / case_id / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index, (image_value, label_value) in enumerate(zip(image_values, label_values, strict=True)):
        Image.new("L", (2, 2), color=image_value).save(image_dir / f"{index:04d}.png")
        Image.new("L", (2, 2), color=label_value).save(label_dir / f"{index:04d}.png")


def test_dataset_clamps_context_at_volume_edges(tmp_path: Path) -> None:
    _write_case(tmp_path, "LUNG-001", [10, 20, 30], [0, 255, 0])
    dataset = LungTumorSliceDataset(tmp_path, ["LUNG-001"], num_slices=3)

    first = dataset[0]
    last = dataset[2]

    assert torch.allclose(first["image"][:, 0, 0], torch.tensor([10, 10, 20]) / 255.0)
    assert torch.allclose(last["image"][:, 0, 0], torch.tensor([20, 30, 30]) / 255.0)
    assert first["mask"].dtype == torch.float32
    assert first["mask"].shape == (1, 2, 2)
    assert first["case_id"] == "LUNG-001"
    assert first["slice_index"] == 0


def test_dataset_rejects_even_context_and_mismatched_pairs(tmp_path: Path) -> None:
    _write_case(tmp_path, "LUNG-001", [10], [0])

    with pytest.raises(ValueError, match="positive odd"):
        LungTumorSliceDataset(tmp_path, ["LUNG-001"], num_slices=2)

    (tmp_path / "LUNG-001" / "labels" / "0000.png").unlink()
    with pytest.raises(FileNotFoundError, match="paired"):
        LungTumorSliceDataset(tmp_path, ["LUNG-001"], num_slices=1)


def test_dataset_assigns_positive_hard_and_easy_groups_from_central_slice(tmp_path: Path) -> None:
    _write_case(tmp_path, "LUNG-001", [10] * 6, [0, 0, 255, 0, 0, 0])
    dataset = LungTumorSliceDataset(tmp_path, ["LUNG-001"], num_slices=1, hard_negative_radius=2)

    assert [item.sample_group for item in dataset.items] == ["hard_negative", "hard_negative", "positive", "hard_negative", "hard_negative", "easy_negative"]


def test_patient_aware_sampler_creates_requested_group_mix(tmp_path: Path) -> None:
    _write_case(tmp_path, "LUNG-001", [10] * 6, [0, 0, 255, 0, 0, 0])
    _write_case(tmp_path, "LUNG-002", [10] * 6, [255, 0, 0, 0, 0, 0])
    dataset = LungTumorSliceDataset(tmp_path, ["LUNG-001", "LUNG-002"], num_slices=1, hard_negative_radius=1)
    sampler = PatientAwareBalancedBatchSampler(dataset, batch_size=4, batches_per_epoch=1, seed=7)

    batch = next(iter(sampler))
    groups = [dataset.items[index].sample_group for index in batch]

    assert len(batch) == 4
    assert groups.count("positive") == 2
    assert groups.count("hard_negative") == 1
    assert groups.count("easy_negative") == 1


def test_patient_aware_sampler_accepts_configurable_group_weights(tmp_path: Path) -> None:
    _write_case(tmp_path, "LUNG-001", [10] * 6, [0, 0, 255, 0, 0, 0])
    _write_case(tmp_path, "LUNG-002", [10] * 6, [255, 0, 0, 0, 0, 0])
    dataset = LungTumorSliceDataset(tmp_path, ["LUNG-001", "LUNG-002"], num_slices=1, hard_negative_radius=1)
    sampler = PatientAwareBalancedBatchSampler(
        dataset, batch_size=10, batches_per_epoch=1, seed=7,
        positive_fraction=0.6, hard_negative_fraction=0.2, easy_negative_fraction=0.2,
    )

    groups = [dataset.items[index].sample_group for index in next(iter(sampler))]

    assert groups.count("positive") == 6
    assert groups.count("hard_negative") == 2
    assert groups.count("easy_negative") == 2
