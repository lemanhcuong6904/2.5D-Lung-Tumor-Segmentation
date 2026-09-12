"""Config-only full-volume inference and evaluation for TransUNet."""

from __future__ import annotations

import csv
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
import psutil
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
    "CHECKPOINT_PATH": r"experiments\transunet_2d_micro_dice_loss_balanced_sampling\best.pt",
    "SPLIT_PATH": str(ROOT_DIR / "data" / "nsclc-radiomics" / "config" / "test.txt"),
    "PROCESSED_ROOT": str(ROOT_DIR / "data" / "nsclc-radiomics" / "processed"),
    "NIFTI_ROOT": str(ROOT_DIR / "data" / "nsclc-radiomics" / "nifti"),
    "OUTPUT_DIR": str(ROOT_DIR / "output" / "transunet_2d_micro_dice_loss_balanced_sampling/test"),
    "NUM_SLICES": 1,
    "BATCH_SIZE": 8,
    "NUM_WORKERS": 0,
    "PIN_MEMORY": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "THRESHOLD": 0.5,
    # Case IDs omitted from inference and from every report/summary metric.
    "EXCLUDE_CASE": ["LUNG-013", "LUNG-014", "LUNG-027", "LUNG-037"],
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


def _add_distribution_summary(summary: dict[str, Any], key: str, values: list[float]) -> None:
    """Add finite mean, sample standard deviation, and IQR for one metric."""
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    summary[key] = float(finite.mean()) if finite.size else float("nan")
    summary[f"{key}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0 if finite.size else float("nan")
    summary[f"{key}_iqr"] = float(np.percentile(finite, 75) - np.percentile(finite, 25)) if finite.size else float("nan")
    summary[f"{key}_valid_cases"] = int(finite.size)


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


def _add_per_case_slice_metrics(
    case_rows: list[dict[str, Any]], slice_rows: list[dict[str, Any]], threshold: float
) -> None:
    """Add per-case 2D aggregates and clearly named 3D metric aliases."""
    slices_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in slice_rows:
        slices_by_case.setdefault(str(row["case_id"]), []).append(row)

    for case_row in case_rows:
        case_id = str(case_row["case_id"])
        rows = slices_by_case.get(case_id, [])
        if not rows:
            raise ValueError(f"no slice metrics found for {case_id}")
        for metric in ("dice", "iou", "recall", "precision", "fp", "fn", "hd95", "assd"):
            case_row[f"{metric}_3d"] = case_row[metric]
        for metric in ("dice", "iou", "recall", "precision"):
            case_row[f"{metric}_2d"] = finite_mean(
                [float(row[metric]) for row in rows]
            )
        case_row.update(
            {
                "slice_count": len(rows),
                # These are slice-level tumor-presence classification counts,
                # matching the fp_2d/fn_2d definitions in summary_metrics.json.
                "slice_tp_2d": sum(int(row["slice_tp"]) for row in rows),
                "slice_tn_2d": sum(int(row["slice_tn"]) for row in rows),
                "fp_2d": sum(int(row["slice_fp"]) for row in rows),
                "fn_2d": sum(int(row["slice_fn"]) for row in rows),
                "hd95_valid": int(np.isfinite(float(case_row["hd95"]))),
                "assd_valid": int(np.isfinite(float(case_row["assd"]))),
                "threshold": threshold,
            }
        )


def run_inference(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    all_case_ids = read_case_ids(Path(cfg["SPLIT_PATH"]))
    excluded_cases = {str(case_id) for case_id in cfg.get("EXCLUDE_CASE", [])}
    case_ids = [case_id for case_id in all_case_ids if case_id not in excluded_cases]
    if not case_ids:
        raise ValueError("EXCLUDE_CASE removes every case from the selected split")
    dataset = LungTumorSliceDataset(
        Path(cfg["PROCESSED_ROOT"]),
        case_ids,
        int(cfg["NUM_SLICES"]),
        build_eval_transform(),
    )
    loader = build_loader(dataset, int(cfg["BATCH_SIZE"]), False, int(cfg["NUM_WORKERS"]), bool(cfg["PIN_MEMORY"]))
    output_dir = Path(cfg["OUTPUT_DIR"])
    records: list[SlicePrediction] = []
    slice_rows: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    process = psutil.Process()
    cpu_rss_peak_bytes = process.memory_info().rss
    cuda_enabled = device.type == "cuda" and torch.cuda.is_available()
    if cuda_enabled:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        vram_start_allocated_bytes = int(torch.cuda.memory_allocated(device))
        vram_start_reserved_bytes = int(torch.cuda.memory_reserved(device))
    else:
        vram_start_allocated_bytes = vram_start_reserved_bytes = 0
    model_forward_seconds = 0.0
    threshold_transfer_seconds = 0.0
    inferred_slice_count = 0
    with torch.inference_mode():
        progress = tqdm(loader, desc="Inference slices", unit="batch", dynamic_ncols=True)
        for batch in progress:
            inputs = batch["image"].to(device, non_blocking=True)
            if cuda_enabled:
                torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            logits = model(inputs)
            if cuda_enabled:
                torch.cuda.synchronize(device)
            model_forward_seconds += time.perf_counter() - forward_start
            transfer_start = time.perf_counter()
            predictions = (logits.sigmoid() >= float(cfg["THRESHOLD"])).cpu().numpy()[:, 0]
            threshold_transfer_seconds += time.perf_counter() - transfer_start
            inferred_slice_count += len(predictions)
            cpu_rss_peak_bytes = max(cpu_rss_peak_bytes, process.memory_info().rss)
            targets = batch["mask"].numpy()[:, 0] > 0.5
            for case_id, slice_index, prediction, target in zip(
                batch["case_id"], batch["slice_index"].tolist(), predictions, targets, strict=True
            ):
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
    if cuda_enabled:
        torch.cuda.synchronize(device)
        vram_peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        vram_peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    else:
        vram_peak_allocated_bytes = vram_peak_reserved_bytes = 0
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
    _add_per_case_slice_metrics(
        case_rows, slice_rows, float(cfg["THRESHOLD"])
    )
    summary: dict[str, Any] = {
        "case_count": len(case_rows),
        "split_case_count": len(all_case_ids),
        "excluded_case_count": len(all_case_ids) - len(case_ids),
        "excluded_cases": sorted(excluded_cases & set(all_case_ids)),
        "unknown_excluded_cases": sorted(excluded_cases - set(all_case_ids)),
        "metric_std_definition": "sample standard deviation (ddof=1)",
        "metric_iqr_definition": "75th percentile minus 25th percentile",
    }
    for key in ("dice", "iou", "recall", "precision", "hd95", "assd"):
        values = [float(row[key]) for row in case_rows]
        _add_distribution_summary(summary, f"{key}_3d", values)
    # Preserve the historic report keys for existing comparison scripts.
    summary["hd95_valid_cases"] = summary["hd95_3d_valid_cases"]
    summary["assd_valid_cases"] = summary["assd_3d_valid_cases"]
    for key in ("dice", "iou", "recall", "precision"):
        _add_distribution_summary(summary, f"{key}_2d", [float(row[key]) for row in slice_rows])
    summary["fp_2d"] = sum(int(row["slice_fp"]) for row in slice_rows)
    summary["fn_2d"] = sum(int(row["slice_fn"]) for row in slice_rows)
    summary.update(_foreground_diagnostics(case_rows))
    summary["threshold"] = float(cfg["THRESHOLD"])
    summary["prediction_volume_directory"] = "predictions"
    summary["prediction_volume_dtype"] = "uint8"
    summary.update(
        {
            # Measured only around model forward passes and threshold/CPU
            # transfer. Metric calculation, 3D aggregation, NIfTI I/O, and
            # report writing are deliberately excluded.
            "inference_slices": inferred_slice_count,
            "model_forward_seconds": model_forward_seconds,
            "threshold_transfer_seconds": threshold_transfer_seconds,
            "inference_seconds_excluding_metrics_and_3d": model_forward_seconds + threshold_transfer_seconds,
            "model_forward_slices_per_second": inferred_slice_count / model_forward_seconds if model_forward_seconds else float("nan"),
            "inference_slices_per_second": inferred_slice_count / (model_forward_seconds + threshold_transfer_seconds) if model_forward_seconds + threshold_transfer_seconds else float("nan"),
            "cpu_rss_peak_mb": cpu_rss_peak_bytes / (1024**2),
            "cuda_enabled": cuda_enabled,
            "vram_start_allocated_mb": vram_start_allocated_bytes / (1024**2) if cuda_enabled else None,
            "vram_start_reserved_mb": vram_start_reserved_bytes / (1024**2) if cuda_enabled else None,
            "peak_vram_allocated_mb": vram_peak_allocated_bytes / (1024**2) if cuda_enabled else None,
            "peak_vram_reserved_mb": vram_peak_reserved_bytes / (1024**2) if cuda_enabled else None,
            "peak_vram_allocated_delta_mb": (vram_peak_allocated_bytes - vram_start_allocated_bytes) / (1024**2) if cuda_enabled else None,
            "peak_vram_reserved_delta_mb": (vram_peak_reserved_bytes - vram_start_reserved_bytes) / (1024**2) if cuda_enabled else None,
        }
    )
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
