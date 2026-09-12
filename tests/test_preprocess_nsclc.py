import numpy as np
import SimpleITK as sitk
from PIL import Image
from pathlib import Path

from scripts.preprocess_nsclc import (
    crop_and_resize,
    derive_lung_mask_from_ct,
    lung_crop_region,
    normalize_hu_to_uint8,
    output_case_id,
    sort_slice_positions,
    write_case_outputs,
)


def test_output_case_id_removes_only_lung1_prefix():
    """Catches a broken raw-to-output case naming rule."""
    assert output_case_id("LUNG1-001") == "LUNG-001"


def test_normalize_hu_clips_and_scales_to_uint8():
    """Catches missing CT clipping or incorrect linear normalization."""
    result = normalize_hu_to_uint8(np.array([-1000, -910, -160, 590, 700]))
    np.testing.assert_array_equal(
        result, np.array([0, 0, 128, 255, 255], dtype=np.uint8)
    )


def test_lung_crop_region_squares_xy_before_applying_margin():
    """Catches applying margin before the per-case in-plane square expansion."""
    array = np.zeros((4, 12, 12), dtype=np.uint8)
    array[1:3, 3:8, 2:10] = 1
    lung = sitk.GetImageFromArray(array)
    lung.SetSpacing((2.0, 2.0, 3.0))

    assert lung_crop_region(lung, margin_mm=2.0) == ((1, 1, 0), (10, 10, 4))


def test_crop_and_resize_preserves_z_and_physical_extent():
    """Catches a resize that changes depth or fails to update in-plane spacing."""
    image = sitk.GetImageFromArray(
        np.arange(4 * 10 * 12, dtype=np.uint8).reshape(4, 10, 12)
    )
    image.SetSpacing((1.0, 2.0, 3.0))

    cropped = crop_and_resize(image, ((2, 3, 1), (8, 5, 2)), is_label=False)

    assert cropped.GetSize() == (256, 256, 2)
    assert cropped.GetSpacing() == (8 / 256, 10 / 256, 3.0)


def test_write_case_outputs_matching_png_pairs_and_nifti(tmp_path):
    """Catches mismatched PNG pairs, non-binary labels, or missing NIfTI."""
    image = sitk.GetImageFromArray(np.full((2, 3, 4), 30, dtype=np.uint8))
    mask = sitk.GetImageFromArray(
        np.array([[[0, 1, 0, 0]] * 3] * 2, dtype=np.uint8)
    )
    image.SetSpacing((1.0, 1.0, 3.0))
    mask.CopyInformation(image)

    write_case_outputs(tmp_path / "processed", tmp_path / "nifti", "LUNG-001", image, mask)

    images = tmp_path / "processed/LUNG-001/images"
    labels = tmp_path / "processed/LUNG-001/labels"
    assert sorted(path.name for path in images.glob("*.png")) == ["0000.png", "0001.png"]
    assert sorted(path.name for path in labels.glob("*.png")) == ["0000.png", "0001.png"]
    assert np.unique(np.asarray(Image.open(labels / "0000.png"))).tolist() == [0, 255]
    assert sitk.ReadImage(str(tmp_path / "nifti/LUNG-001/image.nii.gz")).GetSize() == (4, 3, 2)


def test_write_case_outputs_replaces_an_existing_case_only_when_requested(tmp_path):
    """Catches stale PNG/NIfTI output after an explicit regeneration request."""
    old_image = sitk.GetImageFromArray(np.full((1, 2, 2), 10, dtype=np.uint8))
    new_image = sitk.GetImageFromArray(np.full((1, 2, 2), 20, dtype=np.uint8))
    mask = sitk.GetImageFromArray(np.ones((1, 2, 2), dtype=np.uint8))
    old_image.SetSpacing((1.0, 1.0, 3.0))
    new_image.CopyInformation(old_image)
    mask.CopyInformation(old_image)
    processed_root, nifti_root = tmp_path / "processed", tmp_path / "nifti"

    write_case_outputs(processed_root, nifti_root, "LUNG-001", old_image, mask)
    write_case_outputs(processed_root, nifti_root, "LUNG-001", new_image, mask, overwrite=True)

    output = np.asarray(Image.open(processed_root / "LUNG-001/images/0000.png"))
    assert output.tolist() == [[20, 20], [20, 20]]


def test_sort_slice_positions_uses_image_orientation_not_filename_order():
    """Catches CT ordering that reverses the SEG-to-CT Z mapping."""
    positions = [(-249.5, -460.5, -282.5), (-249.5, -460.5, -681.5), (-249.5, -460.5, -432.5)]
    ordered = sort_slice_positions(positions, (1, 0, 0, 0, 1, 0))
    assert ordered == [1, 2, 0]


def test_derive_lung_mask_excludes_border_air_and_keeps_two_lung_components():
    """Catches a fallback that uses external air instead of the lungs for bbox."""
    array = np.full((4, 32, 32), -1000, dtype=np.int16)
    array[:, 2:30, 2:30] = 0
    array[:, 8:20, 5:12] = -800
    array[:, 8:20, 20:27] = -800
    ct = sitk.GetImageFromArray(array)

    lung = sitk.GetArrayFromImage(derive_lung_mask_from_ct(ct))

    assert lung.sum() == 4 * 12 * 7 * 2
    assert lung[:, 0, 0].sum() == 0
