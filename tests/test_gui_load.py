import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from source.export_gui_volume import (
    GuiGrid,
    LabelPainter,
    export_ct_image,
    to_gui_orientation,
)
from source.gui_load import (
    multilabel_masks,
    parse_label_names,
    seg_nrrd_masks,
    structure_name_from_filename,
)

IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
GRID = GuiGrid(size=(8, 8, 4), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=IDENTITY)


def _image(array_zyx: np.ndarray, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=IDENTITY) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection([float(d) for d in direction])
    return image


def test_ct_export_reorients_flipped_float_volume(tmp_path: Path):
    array = np.random.default_rng(0).uniform(-1000, 1000, size=(4, 5, 6)).astype(np.float32)
    image = _image(array, spacing=(0.5, 0.75, 2.0), origin=(10.0, 20.0, 30.0), direction=(-1, 0, 0, 0, -1, 0, 0, 0, 1))

    export_ct_image(to_gui_orientation(image), tmp_path, "ct")

    meta = json.loads((tmp_path / "ct.json").read_text())
    assert meta["dtype"] == "int16"
    assert np.allclose(meta["direction"], IDENTITY)
    exported = np.fromfile(tmp_path / "ct.bin", dtype="<i2").reshape(meta["shape"])

    # The same physical point holds the same (rounded) value before and after.
    point = image.TransformIndexToPhysicalPoint((1, 2, 3))  # (i, j, k)
    i, j, k = (round((point[a] - meta["origin"][a]) / meta["spacing"][a]) for a in range(3))
    assert exported[k, j, i] == int(np.rint(array[3, 2, 1]))


def test_multilabel_resampled_onto_ct_grid_with_dataset_json_names(tmp_path: Path):
    labels = np.zeros((2, 4, 4), np.uint8)
    labels[0, :2, :2] = 1
    labels[1, 2:, 2:] = 3
    path = tmp_path / "pred.nii.gz"
    sitk.WriteImage(_image(labels, spacing=(2.0, 2.0, 2.0), origin=(0.5, 0.5, 0.5)), str(path))
    (tmp_path / "dataset.json").write_text(json.dumps({"labels": {"background": 0, "liver": 1, "spleen": 3}}))

    array, names = multilabel_masks(path, GRID)

    assert array.shape == GRID.array_shape
    assert array[0, 0, 0] == 1 and array[3, 7, 7] == 3 and array[0, 7, 7] == 0
    assert names == ["liver", "label_2", "spleen"]


def test_multilabel_names_from_nnunet_dataset_root(tmp_path: Path):
    # nnU-Net raw layout: dataset.json sits in the dataset root, one folder above labelsTr/.
    dataset_dir = tmp_path / "Dataset999_Test"
    (dataset_dir / "labelsTr").mkdir(parents=True)
    labels = np.zeros(GRID.array_shape, np.uint8)
    labels[0, :2, :2] = 1
    labels[1, 2:, 2:] = 2
    path = dataset_dir / "labelsTr" / "case_01.nii.gz"
    sitk.WriteImage(_image(labels), str(path))
    (dataset_dir / "dataset.json").write_text(json.dumps({"labels": {"background": 0, "brainstem": 1, "optic_chiasm": 2}}))

    _, names = multilabel_masks(path, GRID)

    assert names == ["brainstem", "optic_chiasm"]


def test_multilabel_name_search_stays_near_the_file(tmp_path: Path):
    # A dataset.json more than SIDECAR_PARENT_LEVELS folders up belongs to something else.
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    labels = np.zeros(GRID.array_shape, np.uint8)
    labels[0, :2, :2] = 1
    path = deep / "seg.nii.gz"
    sitk.WriteImage(_image(labels), str(path))
    (tmp_path / "dataset.json").write_text(json.dumps({"labels": {"background": 0, "liver": 1}}))

    _, names = multilabel_masks(path, GRID)

    assert names == ["label_1"]


def test_ct_dropped_as_mask_is_rejected(tmp_path: Path):
    ct = np.full(GRID.array_shape, -1000, np.int16)
    ct[1:3, 2:6, 2:6] = 40
    path = tmp_path / "ct.nii.gz"
    sitk.WriteImage(_image(ct), str(path))

    with pytest.raises(ValueError, match="did you mean to load it as the CT"):
        multilabel_masks(path, GRID)


def test_seg_nrrd_smaller_structure_wins_overlap_regardless_of_order(tmp_path: Path):
    big = np.zeros(GRID.array_shape, np.uint8)
    big[:, 1:7, 1:7] = 1
    small = np.zeros_like(big)
    small[1:3, 3:5, 3:5] = 1
    small_path = tmp_path / "case_01_OAR_A_Lens.seg.nrrd"
    big_path = tmp_path / "case_01_OAR_Eye.seg.nrrd"
    sitk.WriteImage(_image(small), str(small_path))
    sitk.WriteImage(_image(big), str(big_path))

    for order in ([small_path, big_path], [big_path, small_path]):
        array, names = seg_nrrd_masks(order, GRID)
        lens_id = names.index("A_Lens") + 1
        eye_id = names.index("Eye") + 1
        assert array[2, 4, 4] == lens_id
        assert array[0, 2, 2] == eye_id


def test_label_painter_skips_empty_masks():
    painter = LabelPainter((2, 2, 2))
    assert not painter.add("empty", np.zeros((2, 2, 2), bool))
    assert painter.add("one", np.ones((2, 2, 2), bool))
    assert painter.names == ["one"]


@pytest.mark.parametrize(("file_name", "expected"), [
    ("case_03_OAR_OpticNrv_L.seg.nrrd", "OpticNrv_L"),
    ("brainstem.seg.nrrd", "brainstem"),
    ("Lens.nii.gz", "Lens"),
])
def test_structure_name_from_filename(file_name: str, expected: str):
    assert structure_name_from_filename(Path(file_name)) == expected


def test_parse_label_names_handles_common_json_shapes():
    assert parse_label_names({"labels": {"background": 0, "liver": 1, "region": [1, 2]}}) == {1: "liver"}
    assert parse_label_names({"labels": {"0": "background", "2": "kidney"}}) == {2: "kidney"}
    assert parse_label_names({"labels": ["brainstem", "optic_chiasm"]}) == {1: "brainstem", 2: "optic_chiasm"}
