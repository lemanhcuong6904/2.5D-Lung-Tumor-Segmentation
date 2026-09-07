"""Config-only supervised training for 2D and 2.5D TransUNet experiments."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from data.transunet_dataset import (
    LungTumorSliceDataset,
    PatientAwareBalancedBatchSampler,
    TumorCoveringBatchSampler,
    build_eval_transform,
    build_loader,
    build_train_transform,
    read_case_ids,
)
from models.transunet import TransUNet
from utils.metrics import (
    SlicePrediction,
    aggregate_case_predictions,
    binary_slice_metrics,
    binary_volume_metrics,
    finite_mean,
)
from utils.training import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    warmup_cosine_lambda,
)

CFG: dict[str, Any] = {
    "EXP_NAME": "transunet_2d_BCE_Dice_balanced_sampling",
    "PROCESSED_ROOT": str(ROOT_DIR / "data" / "processed"),
    "NIFTI_ROOT": str(ROOT_DIR / "data" / "nifti"),
    "TRAIN_SPLIT": str(ROOT_DIR / "data" / "config" / "train.txt"),
    "VAL_SPLIT": str(ROOT_DIR / "data" / "config" / "val.txt"),
    "SAVE_ROOT": str(ROOT_DIR / "experiments"),
    "NUM_SLICES": 1,
    "IMAGE_SIZE": 256,
    "PATCH_DIM": 16,
    "BASE_CHANNELS": 128,
    "EMBED_DIM": 512,  #
    "TRANSFORMER_DEPTH": 8,
    "TRANSFORMER_HEADS": 4,
    "MLP_DIM": 512,
    "MLP_RATIO": 0.5,  #
    "DROPOUT": 0.05,
    "BACKBONE_PRETRAINED": True,  #
    "BATCH_TRAIN": 16,
    "BATCH_VAL": 16,
    "HARD_NEGATIVE_RADIUS": 4,
    # True: patient-aware positive/hard/easy-negative sampling. False: use
    # the natural slice distribution (optionally with the epoch limit below).
    "BALANCED_TRAIN_SAMPLING": True,
    "POSITIVE_FRACTION": 0.3,
    "HARD_NEGATIVE_FRACTION": 0.35,
    "EASY_NEGATIVE_FRACTION": 0.35,
    # None: use every training slice once per epoch. Set an integer limit to
    # include every tumor slice while randomly subsampling negative slices.
    "TRAIN_BATCHES_PER_EPOCH": 800,
    "SAMPLER_SEED": 42,
    "NUM_WORKERS": 0,
    "PIN_MEMORY": True,
    "EPOCHS": 20,
    "LR": 2e-4,
    "LR_MIN": 1e-6,
    "WEIGHT_DECAY": 1e-4,
    # Multiplier for target=1 pixels in BCE. Tune jointly with threshold.
    "BCE_POS_WEIGHT": 1.0,
    # Total loss = BCEWithLogitsLoss + DICE_LOSS_WEIGHT * weighted Dice loss.
    "DICE_LOSS_WEIGHT": 1.2,
    "DICE_BACKGROUND_WEIGHT": 1.0,
    "DICE_FOREGROUND_WEIGHT": 1.2,
    "DICE_SMOOTH": 1e-6,
    "AMP": True,
    "THRESHOLD": 0.5,
    "EARLY_STOPPING_PATIENCE": 5,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WANDB_ENABLED": True,
    "WANDB_PROJECT": "lung-tumor-transunet",
    "RESUME_PATH": "",  # Old ResNet-50 checkpoints are incompatible; train this architecture from scratch.
    "SEED": 42,
}


def _model_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "in_channels": int(cfg["NUM_SLICES"]),
        "base_channels": int(cfg["BASE_CHANNELS"]),
        "embed_dim": int(cfg["EMBED_DIM"]),
        "transformer_depth": int(cfg["TRANSFORMER_DEPTH"]),
        "transformer_heads": int(cfg["TRANSFORMER_HEADS"]),
        "mlp_ratio": float(cfg["MLP_RATIO"]),
        "dropout": float(cfg["DROPOUT"]),
        "img_dim": int(cfg["IMAGE_SIZE"]),
        "patch_dim": int(cfg["PATCH_DIM"]),
        "backbone_pretrained": bool(cfg["BACKBONE_PRETRAINED"]),
        "mlp_dim": int(cfg["MLP_DIM"]),
    }


class BCEWeightedDiceLoss(torch.nn.Module):
    """Weighted BCE plus a soft Dice loss over background and tumor classes."""

    def __init__(
        self,
        pos_weight: float,
        dice_weight: float,
        background_weight: float,
        foreground_weight: float,
        smooth: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        if pos_weight <= 0:
            raise ValueError("BCE_POS_WEIGHT must be positive")
        if dice_weight < 0:
            raise ValueError("DICE_LOSS_WEIGHT must be non-negative")
        if background_weight <= 0 or foreground_weight <= 0:
            raise ValueError("Dice class weights must be positive")
        if smooth <= 0:
            raise ValueError("DICE_SMOOTH must be positive")
        self.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=device)
        )
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.register_buffer(
            "dice_class_weights",
            torch.tensor([background_weight, foreground_weight], device=device),
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce_loss = self.bce(logits, targets)
        targets = targets.float()
        foreground_probabilities = logits.sigmoid()
        probabilities = torch.cat(
            (1.0 - foreground_probabilities, foreground_probabilities), dim=1
        )
        class_targets = torch.cat((1.0 - targets, targets), dim=1)
        reduce_dims = tuple(range(2, probabilities.ndim))
        intersection = (probabilities * class_targets).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + class_targets.sum(
            dim=reduce_dims
        )
        class_dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        # Do not score a class absent from a particular target. Background is
        # still scored on all ordinary slices and penalizes false positives.
        valid_classes = class_targets.sum(dim=reduce_dims) > 0
        weights = self.dice_class_weights.to(dtype=class_dice.dtype)
        weighted_dice = (class_dice * valid_classes * weights).sum(dim=1) / (
            (valid_classes * weights)
            .sum(dim=1)
            .clamp_min(torch.finfo(class_dice.dtype).eps)
        )
        dice_loss = 1.0 - weighted_dice.mean()
        total_loss = bce_loss + self.dice_weight * dice_loss
        return total_loss, bce_loss, dice_loss


def build_loss(cfg: dict[str, Any], device: torch.device) -> BCEWeightedDiceLoss:
    """Build the configured BCE + class-weighted Dice objective."""
    return BCEWeightedDiceLoss(
        pos_weight=float(cfg["BCE_POS_WEIGHT"]),
        dice_weight=float(cfg["DICE_LOSS_WEIGHT"]),
        background_weight=float(cfg["DICE_BACKGROUND_WEIGHT"]),
        foreground_weight=float(cfg["DICE_FOREGROUND_WEIGHT"]),
        smooth=float(cfg["DICE_SMOOTH"]),
        device=device,
    )


def _summarize_predictions(
    records: list[SlicePrediction], compute_volume_metrics: bool = True
) -> dict[str, float]:
    """Aggregate slice metrics without letting one FP collapse a whole slice."""
    if not records:
        raise ValueError("cannot summarize an empty prediction collection")
    slice_rows = [
        binary_slice_metrics(record.prediction, record.target) for record in records
    ]
    true_positive = sum(
        int(np.logical_and(record.prediction, record.target).sum())
        for record in records
    )
    false_positive = sum(
        int(np.logical_and(record.prediction, ~record.target).sum())
        for record in records
    )
    false_negative = sum(
        int(np.logical_and(~record.prediction, record.target).sum())
        for record in records
    )
    micro_denominator = 2 * true_positive + false_positive + false_negative
    negative_records = [record for record in records if not record.target.any()]
    summary = {
        "dice_micro": (
            1.0 if micro_denominator == 0 else 2.0 * true_positive / micro_denominator
        ),
        "iou_2d": finite_mean([float(row["iou"]) for row in slice_rows]),
        "recall_2d": finite_mean([float(row["recall"]) for row in slice_rows]),
        "precision_2d": finite_mean([float(row["precision"]) for row in slice_rows]),
        "fp_2d": finite_mean([float(row["fp"]) for row in slice_rows]),
        "fn_2d": finite_mean([float(row["fn"]) for row in slice_rows]),
        # Unlike dice_micro, this excludes background-only targets. It is the
        # useful score for knowing whether the model actually segments tumors.
        "dice_tumor": finite_mean(
            [
                float(row["dice"])
                for record, row in zip(records, slice_rows, strict=True)
                if record.target.any()
            ]
        ),
        "negative_slice_specificity": (
            float(sum(not record.prediction.any() for record in negative_records))
            / len(negative_records)
            if negative_records
            else float("nan")
        ),
    }
    if not compute_volume_metrics:
        return summary | {"dice_3d": float("nan"), "iou_3d": float("nan")}
    volumes = aggregate_case_predictions(records)
    volume_rows = [
        binary_volume_metrics(value["prediction"], value["target"], (1.0, 1.0, 1.0))
        for value in volumes.values()
    ]
    return summary | {
        "dice_3d": finite_mean([float(row["dice"]) for row in volume_rows]),
        "iou_3d": finite_mean([float(row["iou"]) for row in volume_rows]),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: Any,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    threshold: float,
    progress_desc: str | None = None,
    step_logger: Callable[[dict[str, float]], None] | None = None,
    compute_volume_metrics: bool = True,
) -> dict[str, float]:
    """Run one train or validation epoch and aggregate slice/case predictions."""
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None and device.type == "cuda"
    total_loss, total_bce_loss, total_dice_loss, total_images = 0.0, 0.0, 0.0, 0
    records: list[SlicePrediction] = []
    iterator = tqdm(
        loader,
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
        disable=progress_desc is None,
    )
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=use_amp
        ):
            logits = model(images)
            loss, bce_loss, dice_loss = criterion(logits, masks)
        if training:
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        predictions = (logits.detach().sigmoid() >= threshold).cpu().numpy()[:, 0]
        targets = masks.detach().cpu().numpy()[:, 0] > 0.5
        case_ids = list(batch["case_id"])
        indices = batch["slice_index"].detach().cpu().tolist()
        records.extend(
            SlicePrediction(str(case_id), int(slice_index), prediction, target)
            for case_id, slice_index, prediction, target in zip(
                case_ids, indices, predictions, targets, strict=True
            )
        )
        batch_size = int(images.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_bce_loss += float(bce_loss.detach()) * batch_size
        total_dice_loss += float(dice_loss.detach()) * batch_size
        total_images += batch_size
        batch_records = [
            SlicePrediction(str(case_id), int(slice_index), prediction, target)
            for case_id, slice_index, prediction, target in zip(
                case_ids, indices, predictions, targets, strict=True
            )
        ]
        batch_summary = _summarize_predictions(
            batch_records, compute_volume_metrics=False
        )
        step_values = {
            "loss": float(loss.detach()),
            "bce_loss": float(bce_loss.detach()),
            "dice_loss": float(dice_loss.detach()),
            "dice_micro": batch_summary["dice_micro"],
            "dice_tumor": batch_summary["dice_tumor"],
            "negative_slice_specificity": batch_summary["negative_slice_specificity"],
        }
        iterator.set_postfix(
            loss=f"{step_values['loss']:.4f}",
            bce=f"{step_values['bce_loss']:.4f}",
            dice_loss=f"{step_values['dice_loss']:.4f}",
        )
        if step_logger is not None:
            step_logger(step_values)
    if total_images == 0:
        raise ValueError("loader yielded no batches")
    return {
        "loss": total_loss / total_images,
        "bce_loss": total_bce_loss / total_images,
        "dice_loss": total_dice_loss / total_images,
    } | _summarize_predictions(records, compute_volume_metrics)


def _wandb_run(cfg: dict[str, Any]) -> Any | None:
    if not bool(cfg["WANDB_ENABLED"]):
        return None
    try:
        import wandb

        return wandb.init(
            project=str(cfg["WANDB_PROJECT"]), name=str(cfg["EXP_NAME"]), config=cfg
        )
    except Exception as error:
        print(f"[WARN] W&B disabled: {error}")
        return None


def _wandb_finite_values(values: dict[str, float | int]) -> dict[str, float | int]:
    """Drop NaN/Inf metrics so W&B charts contain only valid observations."""
    return {
        key: value
        for key, value in values.items()
        if not isinstance(value, (float, np.floating)) or np.isfinite(value)
    }


def _wandb_step_logger(
    wandb_run: Any | None, split: str
) -> Callable[[dict[str, float]], None] | None:
    """Create an independent W&B step stream for training or validation batches."""
    if wandb_run is None:
        return None
    step_key = f"{split}_step"
    metric_keys = [
        "loss",
        "bce_loss",
        "dice_loss",
        "dice_micro",
        "dice_tumor",
        "negative_slice_specificity",
    ]
    wandb_run.define_metric(step_key)
    for metric_key in metric_keys:
        wandb_run.define_metric(f"{split}_step_{metric_key}", step_metric=step_key)
    step = 0

    def log(values: dict[str, float]) -> None:
        nonlocal step
        step += 1
        finite_values = _wandb_finite_values(values)
        wandb_run.log(
            {step_key: step}
            | {f"{split}_step_{key}": value for key, value in finite_values.items()}
        )

    return log


def restore_training_state(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, float, int]:
    """Restore model and optimizer state, returning next epoch and stop state."""
    checkpoint = load_checkpoint(Path(checkpoint_path), device)
    required = {
        "epoch",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "best_val_loss",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise KeyError(f"resume checkpoint is missing keys: {', '.join(missing)}")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_val_loss"]),
        int(checkpoint.get("stale_epochs", 0)),
    )


def main() -> None:
    """Train with the constants declared in CFG; no command-line interface is used."""
    cfg = dict(CFG)
    seed_everything(int(cfg["SEED"]))
    device = torch.device(str(cfg["DEVICE"]))
    experiment_dir = Path(cfg["SAVE_ROOT"]) / str(cfg["EXP_NAME"])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "config.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
    )
    train_dataset = LungTumorSliceDataset(
        Path(cfg["PROCESSED_ROOT"]),
        read_case_ids(Path(cfg["TRAIN_SPLIT"])),
        int(cfg["NUM_SLICES"]),
        build_train_transform(),
        hard_negative_radius=int(cfg["HARD_NEGATIVE_RADIUS"]),
    )
    val_dataset = LungTumorSliceDataset(
        Path(cfg["PROCESSED_ROOT"]),
        read_case_ids(Path(cfg["VAL_SPLIT"])),
        int(cfg["NUM_SLICES"]),
        build_eval_transform(),
    )
    train_batch_sampler = (
        PatientAwareBalancedBatchSampler(
            train_dataset,
            batch_size=int(cfg["BATCH_TRAIN"]),
            batches_per_epoch=(
                None
                if cfg["TRAIN_BATCHES_PER_EPOCH"] is None
                else int(cfg["TRAIN_BATCHES_PER_EPOCH"])
            ),
            seed=int(cfg["SAMPLER_SEED"]),
            positive_fraction=float(cfg["POSITIVE_FRACTION"]),
            hard_negative_fraction=float(cfg["HARD_NEGATIVE_FRACTION"]),
            easy_negative_fraction=float(cfg["EASY_NEGATIVE_FRACTION"]),
        )
        if bool(cfg["BALANCED_TRAIN_SAMPLING"])
        else (
            TumorCoveringBatchSampler(
                train_dataset,
                batch_size=int(cfg["BATCH_TRAIN"]),
                batches_per_epoch=int(cfg["TRAIN_BATCHES_PER_EPOCH"]),
                seed=int(cfg["SAMPLER_SEED"]),
            )
            if cfg["TRAIN_BATCHES_PER_EPOCH"] is not None
            else None
        )
    )
    train_loader = build_loader(
        train_dataset,
        int(cfg["BATCH_TRAIN"]),
        not bool(cfg["BALANCED_TRAIN_SAMPLING"]),
        int(cfg["NUM_WORKERS"]),
        bool(cfg["PIN_MEMORY"]),
        batch_sampler=train_batch_sampler,
    )
    val_loader = build_loader(
        val_dataset,
        int(cfg["BATCH_VAL"]),
        False,
        int(cfg["NUM_WORKERS"]),
        bool(cfg["PIN_MEMORY"]),
    )
    model_config = _model_config(cfg)
    model = TransUNet(**model_config).to(device)
    criterion = build_loss(cfg, device)
    optimizer = AdamW(
        model.parameters(), lr=float(cfg["LR"]), weight_decay=float(cfg["WEIGHT_DECAY"])
    )
    warmup_epochs = max(1, math.ceil(0.1 * int(cfg["EPOCHS"])))
    scheduler = LambdaLR(
        optimizer,
        warmup_cosine_lambda(
            int(cfg["EPOCHS"]), warmup_epochs, float(cfg["LR"]), float(cfg["LR_MIN"])
        ),
    )
    use_amp = bool(cfg["AMP"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    start_epoch, best_val_loss, stale_epochs = 1, float("inf"), 0
    resume_path = str(cfg["RESUME_PATH"]).strip()
    if resume_path:
        start_epoch, best_val_loss, stale_epochs = restore_training_state(
            Path(resume_path), model, optimizer, scheduler, scaler, device
        )
        print(
            f"[RESUME] checkpoint={resume_path} start_epoch={start_epoch} "
            f"best_val_loss={best_val_loss:.6f} stale_epochs={stale_epochs}"
        )
    wandb_run = _wandb_run(cfg)
    train_step_logger = _wandb_step_logger(wandb_run, "train")
    val_step_logger = _wandb_step_logger(wandb_run, "val")
    try:
        for epoch in range(start_epoch, int(cfg["EPOCHS"]) + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                scaler,
                float(cfg["THRESHOLD"]),
                f"Epoch {epoch}/{cfg['EPOCHS']} | train",
                step_logger=train_step_logger,
                compute_volume_metrics=False,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                None,
                None,
                float(cfg["THRESHOLD"]),
                f"Epoch {epoch}/{cfg['EPOCHS']} | val",
                step_logger=val_step_logger,
                compute_volume_metrics=False,
            )
            log = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_bce_loss": train_metrics["bce_loss"],
                "train_dice_loss": train_metrics["dice_loss"],
                "train_dice_micro": train_metrics["dice_micro"],
                "train_dice_tumor": train_metrics["dice_tumor"],
                "train_negative_specificity": train_metrics[
                    "negative_slice_specificity"
                ],
                "val_loss": val_metrics["loss"],
                "val_bce_loss": val_metrics["bce_loss"],
                "val_dice_loss": val_metrics["dice_loss"],
                "val_dice_micro": val_metrics["dice_micro"],
                "val_dice_tumor": val_metrics["dice_tumor"],
                "val_negative_specificity": val_metrics["negative_slice_specificity"],
            }
            print(log)
            if wandb_run is not None:
                wandb_run.log(_wandb_finite_values(log))
            should_stop = False
            val_loss = val_metrics["loss"]
            if np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss, stale_epochs = val_loss, 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(cfg["EARLY_STOPPING_PATIENCE"]):
                    should_stop = True
                    print(
                        f"Early stopping after {stale_epochs} epochs without val_loss improvement."
                    )
            scheduler.step()
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_val_loss": best_val_loss,
                "stale_epochs": stale_epochs,
                "config": cfg,
                "model_config": model_config,
            }
            save_checkpoint(experiment_dir / "last.pt", state)
            if stale_epochs == 0:
                save_checkpoint(experiment_dir / "best.pt", state)
            if should_stop:
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
