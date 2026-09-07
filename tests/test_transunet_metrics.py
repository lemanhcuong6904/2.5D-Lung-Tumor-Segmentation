import numpy as np
import SimpleITK as sitk

from inferences.infer_transunet import write_prediction_volume
from utils.metrics import binary_slice_metrics, binary_volume_metrics


def test_slice_metrics_empty_convention_and_error_counts() -> None:
    empty = np.zeros((2, 2), dtype=bool)
    assert binary_slice_metrics(empty, empty) == {
        "dice": 1.0,
        "iou": 1.0,
        "recall": 1.0,
        "precision": 1.0,
        "fp": 0,
        "fn": 0,
    }

    result = binary_slice_metrics(
        np.array([[1, 1]], dtype=bool), np.array([[1, 0]], dtype=bool)
    )
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["precision"] == 0.5


def test_volume_metrics_uses_spacing_and_empty_surface_nan() -> None:
    volume = np.zeros((3, 5, 5), dtype=bool)
    volume[1, 2, 2] = True

    equal = binary_volume_metrics(volume, volume, (2.0, 1.0, 1.0))
    empty = binary_volume_metrics(np.zeros_like(volume), volume, (2.0, 1.0, 1.0))

    assert equal["dice"] == 1.0
    assert equal["iou"] == 1.0
    assert equal["hd95"] == 0.0
    assert equal["assd"] == 0.0
    assert np.isnan(empty["hd95"])
    assert np.isnan(empty["assd"])


def test_write_prediction_preserves_reference_geometry(tmp_path) -> None:
    reference = sitk.GetImageFromArray(np.zeros((2, 4, 4), dtype=np.uint8))
    reference.SetSpacing((1.5, 1.5, 3.0))
    reference.SetOrigin((4.0, 5.0, 6.0))
    reference_path = tmp_path / "reference.nii.gz"
    sitk.WriteImage(reference, str(reference_path))

    output_path = write_prediction_volume(
        tmp_path / "prediction.nii.gz", np.zeros((2, 4, 4), dtype=np.uint8), reference_path
    )
    written = sitk.ReadImage(str(output_path))

    assert written.GetSpacing() == (1.5, 1.5, 3.0)
    assert written.GetOrigin() == (4.0, 5.0, 6.0)
    assert written.GetSize() == (4, 4, 2)
