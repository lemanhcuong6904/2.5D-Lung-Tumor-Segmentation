"""Reproduce the paper-compatible 2.5D-3 inference and 3D post-processing.

This intentionally writes to a new directory: the existing predictions were
thresholded at 0.50 and cannot be repaired into 0.30-threshold masks later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infer_transunet import CFG as BASE_INFERENCE_CFG, run_inference
from postprocess_predictions import CFG as BASE_POSTPROCESS_CFG, run_postprocessing


# The paper uses three adjacent slices and selects 0.30 on validation.  Use the
# best checkpoint, not the final (last-epoch) checkpoint, for evaluation.
INFERENCE_CFG = dict(BASE_INFERENCE_CFG) | {
    "CHECKPOINT_PATH": str(ROOT_DIR / "experiments" / "transunet_2.5d-3_balanced_sampling" / "best.pt"),
    "OUTPUT_DIR": str(ROOT_DIR / "output" / "transunet_2.5d-3_balanced_sampling" / "test_threshold_0.30"),
    "NUM_SLICES": 3,
    "THRESHOLD": 0.30,
}

# Use the currently GT-guided morphology choice.  This should subsequently be
# selected on validation (rather than test) before reporting a final test score.
POSTPROCESS_CFG = dict(BASE_POSTPROCESS_CFG) | {
    "PREDICTIONS_DIR": str(Path(INFERENCE_CFG["OUTPUT_DIR"]) / "predictions"),
    "OUTPUT_DIR": str(Path(INFERENCE_CFG["OUTPUT_DIR"]) / "post_processed"),
    "CLOSING_RADIUS_MM": 5.0,
    "OPENING_RADIUS_MM": 1.0,
    "MIN_COMPONENT_VOLUME_MM3": 50.0,
    "CONNECTIVITY": 26,
}


def main() -> None:
    _, inference_summary = run_inference(dict(INFERENCE_CFG))
    _, postprocess_summary = run_postprocessing(dict(POSTPROCESS_CFG))
    print(json.dumps({"inference": inference_summary, "post_processed": postprocess_summary}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
