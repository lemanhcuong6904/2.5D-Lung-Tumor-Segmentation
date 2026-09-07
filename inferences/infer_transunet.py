"""Config-only full-volume inference and evaluation for TransUNet."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import SimpleITK as sitk
import torch
from tqdm.auto import tqdm

from data.transunet_dataset import LungTumorSliceDataset, build_eval_transform, build_loader, read_case_ids
from models.transunet import TransUNet
from utils.metrics import SlicePrediction, aggregate_case_predictions, binary_slice_metrics, binary_volume_metrics, finite_mean
from utils.training import load_checkpoint

CFG: dict[str, Any] = {
    # Set this after training the notebook TransUNet. The previous ResNet-50
    # checkpoint is incompatible with this architecture.
    "CHECKPOINT_PATH": r"D:\2.5D Lung Tumor Segmentation\experiments\transunet_2d\best.pt",
    "SPLIT_PATH": str(ROOT_DIR / "data" / "config" / "test.txt"),
    "PROCESSED_ROOT": str(ROOT_DIR / "data" / "processed"),
    "NIFTI_ROOT": str(ROOT_DIR / "data" / "nifti"),
    "OUTPUT_DIR": str(ROOT_DIR / "output" / "transunet_2d_test"),
    "NUM_SLICES": 1,
    "BATCH_SIZE": 8,
    "NUM_WORKERS": 0,
    "PIN_MEMORY": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "THRESHOLD": 0.5,
}


def write_prediction_volume(output_path: Path, prediction_zyx: np.ndarray, reference_path: Path) -> Path:
    """Write a binary ZYX prediction using exact image geometry from a reference NIfTI."""
    reference = sitk.ReadImage(str(reference_path))
    prediction = np.asarray(prediction_zyx, dtype=np.uint8)
    expected_shape = tuple(reversed(reference.GetSize()))
    if prediction.shape != expected_shape:
        raise ValueError(f"prediction shape {prediction.shape} does not match reference shape {expected_shape}")
    image = sitk.GetImageFromArray(prediction)
    image.CopyInformation(reference)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))
    return output_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _foreground_diagnostics(case_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize whether 3D surface metrics were computable for this run."""
    return {
        "pred_positive_cases": float(sum(bool(row["pred_has_foreground"]) for row in case_rows)),
        "target_positive_cases": float(sum(bool(row["target_has_foreground"]) for row in case_rows)),
        "pred_foreground_voxels": float(sum(int(row["pred_foreground_voxels"]) for row in case_rows)),
        "target_foreground_voxels": float(sum(int(row["target_foreground_voxels"]) for row in case_rows)),
    }


def _slice_classification_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, int]:
    """Classify each slice by whether it contains any tumor pixel.

    Dice/IoU/etc. remain pixel-wise segmentation metrics.  In contrast,
    ``slice_fp`` and ``slice_fn`` are binary, per-slice classification errors
    and can therefore be summed into integral counts over a split.
    """
    pred_has_tumor = bool(np.any(prediction))
    target_has_tumor = bool(np.any(target))
    return {
        "pred_has_tumor": int(pred_has_tumor),
        "target_has_tumor": int(target_has_tumor),
        "slice_tp": int(pred_has_tumor and target_has_tumor),
        "slice_tn": int(not pred_has_tumor and not target_has_tumor),
        "slice_fp": int(pred_has_tumor and not target_has_tumor),
        "slice_fn": int(not pred_has_tumor and target_has_tumor),
    }


def run_inference(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Run one checkpoint over a split, save volumes, and return case/report summaries."""
    device = torch.device(str(cfg["DEVICE"]))
    checkpoint_path = Path(str(cfg["CHECKPOINT_PATH"]).strip())
    if not str(cfg["CHECKPOINT_PATH"]).strip():
        raise ValueError("Set CFG['CHECKPOINT_PATH'] to a checkpoint trained with the current TransUNet architecture.")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = dict(checkpoint["model_config"])
    if int(model_config["in_channels"]) != int(cfg["NUM_SLICES"]):
        raise ValueError("checkpoint in_channels and CFG NUM_SLICES differ")
    model = TransUNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    case_ids = read_case_ids(Path(cfg["SPLIT_PATH"]))
    dataset = LungTumorSliceDataset(
        Path(cfg["PROCESSED_ROOT"]),
        case_ids,
        int(cfg["NUM_SLICES"]),
        build_eval_transform(),
    )
    loader = build_loader(dataset, int(cfg["BATCH_SIZE"]), False, int(cfg["NUM_WORKERS"]), bool(cfg["PIN_MEMORY"]))
    records: list[SlicePrediction] = []
    slice_rows: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    with torch.inference_mode():
        progress = tqdm(loader, desc="Inference slices", unit="batch", dynamic_ncols=True)
        for batch in progress:
            logits = model(batch["image"].to(device, non_blocking=True))
            predictions = (logits.sigmoid() >= float(cfg["THRESHOLD"])).cpu().numpy()[:, 0]
            targets = batch["mask"].numpy()[:, 0] > 0.5
            for case_id, slice_index, prediction, target in zip(batch["case_id"], batch["slice_index"].tolist(), predictions, targets, strict=True):
                case_id = str(case_id)
                seen_cases.add(case_id)
                record = SlicePrediction(case_id, int(slice_index), prediction, target)
                records.append(record)
                pixel_metrics = binary_slice_metrics(prediction, target)
                # Do not expose pixel-level FP/FN here: reported FP/FN are
                # defined by the requested per-slice tumor-presence task.
                slice_rows.append(
                    {"case_id": case_id, "slice_index": int(slice_index)}
                    | {
                        key: value
                        for key, value in pixel_metrics.items()
                        if key not in {"fp", "fn"}
                    }
                    | _slice_classification_metrics(prediction, target)
                )
            progress.set_postfix(cases=f"{len(seen_cases)}/{len(case_ids)}")
    output_dir = Path(cfg["OUTPUT_DIR"])
    volumes = aggregate_case_predictions(records)
    case_rows: list[dict[str, Any]] = []
    for case_id, volume in tqdm(volumes.items(), total=len(volumes), desc="Writing 3D volumes", unit="case", dynamic_ncols=True):
        reference_image = Path(cfg["NIFTI_ROOT"]) / case_id / "image.nii.gz"
        reference_mask = Path(cfg["NIFTI_ROOT"]) / case_id / "mask.nii.gz"
        label = sitk.ReadImage(str(reference_mask))
        target = sitk.GetArrayFromImage(label) > 0
        prediction = volume["prediction"]
        if prediction.shape != target.shape:
            raise ValueError(f"PNG and NIfTI slice shapes differ for {case_id}: {prediction.shape} vs {target.shape}")
        metrics = binary_volume_metrics(prediction, target, tuple(float(value) for value in label.GetSpacing()))
        write_prediction_volume(output_dir / "predictions" / f"{case_id}.nii.gz", prediction, reference_image)
        case_rows.append(
            {
                "case_id": case_id,
                "pred_foreground_voxels": int(prediction.sum()),
                "target_foreground_voxels": int(target.sum()),
                "pred_has_foreground": bool(prediction.any()),
                "target_has_foreground": bool(target.any()),
            }
            | metrics
        )
    summary: dict[str, float | int] = {}
    for key in ("dice", "iou", "hd95", "assd"):
        values = [float(row[key]) for row in case_rows]
        summary[f"{key}_3d"] = finite_mean(values)
        if key in {"hd95", "assd"}:
            summary[f"{key}_valid_cases"] = float(sum(np.isfinite(values)))
    for key in ("dice", "iou", "recall", "precision"):
        summary[f"{key}_2d"] = finite_mean([float(row[key]) for row in slice_rows])
    summary["fp_2d"] = sum(int(row["slice_fp"]) for row in slice_rows)
    summary["fn_2d"] = sum(int(row["slice_fn"]) for row in slice_rows)
    summary.update(_foreground_diagnostics(case_rows))
    summary["threshold"] = float(cfg["THRESHOLD"])
    _write_csv(output_dir / "per_case_metrics.csv", case_rows)
    _write_csv(output_dir / "slice_metrics.csv", slice_rows)
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return case_rows, summary


def main() -> None:
    """Run inference with settings declared in CFG; no command-line interface is used."""
    _, summary = run_inference(dict(CFG))
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
