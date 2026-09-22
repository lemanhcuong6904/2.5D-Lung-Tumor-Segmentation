"""Run TransUNet inference on the NSCLC-Radiomics test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inferences.infer_transunet import CFG as BASE_CFG, run_inference

CFG = dict(BASE_CFG) | {
    "CHECKPOINT_PATH": r"experiments\transunet_2d_micro_dice_loss_balanced_sampling\best.pt",
    "SPLIT_PATH": str(ROOT_DIR / "data" / "nsclc-radiomics" / "config" / "test.txt"),
    "PROCESSED_ROOT": str(ROOT_DIR / "data" / "nsclc-radiomics" / "processed"),
    "NIFTI_ROOT": str(ROOT_DIR / "data" / "nsclc-radiomics" / "nifti"),
    "OUTPUT_DIR": str(
        ROOT_DIR / "output" / "transunet_2d_micro_dice_loss_balanced_sampling" / "test"
    ),
    "NUM_SLICES": 1,
    # Set to "cpu" when benchmarking CPU inference.
    "DEVICE": "cpu",
    "PIN_MEMORY": False,
    "EXCLUDE_CASE": [],
}


def main() -> None:
    _, summary = run_inference(dict(CFG))
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
