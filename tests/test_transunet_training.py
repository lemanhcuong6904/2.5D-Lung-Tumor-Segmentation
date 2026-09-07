import pytest
import torch
import numpy as np

from models.transunet import TransUNet
from trains.train_transunet import (
    _summarize_predictions,
    build_weighted_bce_loss,
    restore_training_state,
    run_epoch,
)
from utils.metrics import SlicePrediction
from utils.training import load_checkpoint, save_checkpoint, warmup_cosine_lambda


def test_warmup_cosine_reaches_base_then_decays_to_minimum() -> None:
    schedule = warmup_cosine_lambda(total_epochs=10, warmup_epochs=2, lr=1e-3, lr_min=1e-5)

    assert schedule(0) == pytest.approx(0.505)
    assert schedule(1) == pytest.approx(1.0)
    assert schedule(9) == pytest.approx(0.01)


def test_weighted_bce_uses_configured_foreground_weight() -> None:
    criterion = build_weighted_bce_loss({"BCE_POS_WEIGHT": 7.0}, torch.device("cpu"))
    logits = torch.zeros(1, 1, 1, 2)
    target = torch.tensor([[[[1.0, 0.0]]]])

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(7.0)
    )

    assert torch.allclose(criterion(logits, target), expected)


def test_checkpoint_round_trip(tmp_path) -> None:
    path = tmp_path / "state.pt"
    save_checkpoint(path, {"epoch": 3, "tensor": torch.tensor([2])})

    state = load_checkpoint(path, torch.device("cpu"))

    assert state["epoch"] == 3
    assert state["tensor"].item() == 2


def test_run_epoch_returns_requested_metrics() -> None:
    batch = {
        "image": torch.randn(2, 1, 32, 32),
        "mask": torch.zeros(2, 1, 32, 32),
        "case_id": ["LUNG-001", "LUNG-001"],
        "slice_index": torch.tensor([0, 1]),
    }
    model = TransUNet(1, base_channels=8, embed_dim=64, transformer_depth=1, transformer_heads=4, mlp_ratio=2.0, dropout=0.0, img_dim=32, backbone_pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = run_epoch(model, [batch], torch.nn.BCEWithLogitsLoss(), torch.device("cpu"), optimizer, None, 0.5)

    assert {"loss", "dice_2d", "dice_tumor", "dice_3d"} <= metrics.keys()
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_run_epoch_emits_per_batch_loss_components_to_step_logger() -> None:
    batch = {
        "image": torch.randn(1, 1, 32, 32),
        "mask": torch.zeros(1, 1, 32, 32),
        "case_id": ["LUNG-001"],
        "slice_index": torch.tensor([0]),
    }
    events: list[dict[str, float]] = []
    model = TransUNet(1, base_channels=8, embed_dim=64, transformer_depth=1, transformer_heads=4, mlp_ratio=2.0, dropout=0.0, img_dim=32, backbone_pretrained=False)

    run_epoch(model, [batch], torch.nn.BCEWithLogitsLoss(), torch.device("cpu"), None, None, 0.5, step_logger=events.append)

    assert len(events) == 1
    assert {
        "loss", "bce_loss", "dice", "dice_tumor",
    } <= events[0].keys()


def test_run_epoch_skips_3d_reconstruction_for_repeated_sampled_slices() -> None:
    batch = {
        "image": torch.randn(2, 1, 32, 32),
        "mask": torch.zeros(2, 1, 32, 32),
        "case_id": ["LUNG-001", "LUNG-001"],
        "slice_index": torch.tensor([3, 3]),
    }
    model = TransUNet(1, base_channels=8, embed_dim=64, transformer_depth=1, transformer_heads=4, mlp_ratio=2.0, dropout=0.0, img_dim=32, backbone_pretrained=False)

    metrics = run_epoch(model, [batch], torch.nn.BCEWithLogitsLoss(), torch.device("cpu"), None, None, 0.5, compute_volume_metrics=False)

    assert np.isnan(metrics["dice_3d"])


def test_dice_tumor_excludes_empty_ground_truth_slices() -> None:
    tumor = np.array([[1, 0]], dtype=bool)
    summary = _summarize_predictions(
        [
            SlicePrediction("LUNG-001", 0, np.zeros_like(tumor), tumor),
            SlicePrediction("LUNG-001", 1, np.zeros_like(tumor), np.zeros_like(tumor)),
        ]
    )

    assert summary["dice_2d"] == pytest.approx(0.5)
    assert summary["dice_tumor"] == pytest.approx(0.0)


def test_restore_training_state_resumes_epoch_optimizer_and_early_stop(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint_path = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "epoch": 4,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": None,
            "best_val_dice_tumor": 0.25,
            "stale_epochs": 2,
        },
    )
    for parameter in model.parameters():
        parameter.data.zero_()

    start_epoch, best_val_dice_tumor, stale_epochs = restore_training_state(
        checkpoint_path, model, optimizer, scheduler, None, torch.device("cpu")
    )

    assert start_epoch == 5
    assert best_val_dice_tumor == pytest.approx(0.25)
    assert stale_epochs == 2
    assert any(torch.count_nonzero(parameter).item() for parameter in model.parameters())
