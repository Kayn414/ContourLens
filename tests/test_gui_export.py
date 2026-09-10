import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from pydicom.uid import generate_uid

from source.export_gui_volume import GuiGrid, read_dicom_series, resolve_dicom_series
from source.gui_export import export_nifti, export_rtstruct
from source.gui_load import rasterize_rtstruct, sidecar_label_names


def _write_ct_series(directory: Path, shape_zyx=(6, 64, 64), spacing=(1.0, 1.0, 2.0)) -> None:
    """A minimal axial CT series with the tags rt-utils needs."""
    image = sitk.GetImageFromArray(np.full(shape_zyx, -500, np.int16))
    image.SetSpacing(spacing)
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for k in range(shape_zyx[0]):
        slice_image = image[:, :, k]
        tags = {
            "0008|0016": "1.2.840.10008.5.1.4.1.1.2",
            "0008|0018": generate_uid(),
            "0008|0020": "20260101",
            "0008|0030": "120000",
            "0008|0060": "CT",
            "0010|0010": "Test^Phantom",
            "0010|0020": "TEST",
            "0018|0050": str(spacing[2]),
            "0020|000d": study,
            "0020|000e": series,
            "0020|0010": "1",
            "0020|0013": str(k + 1),
            "0020|0032": "\\".join(str(v) for v in image.TransformIndexToPhysicalPoint((0, 0, k))),
            "0020|0037": "1\\0\\0\\0\\1\\0",
            "0020|0052": frame,
        }
        for key, value in tags.items():
            slice_image.SetMetaData(key, value)
        writer.SetFileName(str(directory / f"CT{k:03d}.dcm"))
        writer.Execute(slice_image)


def test_prediction_rtstruct_round_trips_through_the_rtstruct_loader(tmp_path: Path):
    dicom_dir = tmp_path / "ct"
    dicom_dir.mkdir()
    _write_ct_series(dicom_dir)
    series = resolve_dicom_series(dicom_dir)
    assert series is not None
    ct = read_dicom_series(series)
    grid = GuiGrid.of(ct)

    labels = np.zeros(grid.array_shape, np.uint8)
    labels[1:4, 8:30, 8:28] = 1
    labels[2:5, 36:58, 34:56] = 2
    prediction = sitk.GetImageFromArray(labels)
    prediction.CopyInformation(ct)

    out_dir = tmp_path / "export"
    out_dir.mkdir()
    path = export_rtstruct(prediction, ["brainstem", "optic_chiasm"], series, out_dir, {"brainstem": [230, 60, 60]})
    array, names = rasterize_rtstruct(path, series, grid)

    assert names == ["brainstem", "optic_chiasm"]
    for label_id in (1, 2):
        exported, original = array == label_id, labels == label_id
        assert 2 * (exported & original).sum() / (exported.sum() + original.sum()) > 0.9


def test_nifti_export_keeps_names_when_dropped_back_in(tmp_path: Path):
    image = sitk.GetImageFromArray(np.zeros((2, 4, 4), np.uint8))
    export_nifti(image, ["liver", "", "spleen"], tmp_path)

    assert json.loads((tmp_path / "dataset.json").read_text())["labels"] == {"background": 0, "liver": 1, "spleen": 3}
    assert sidecar_label_names(tmp_path / "prediction.nii.gz") == {1: "liver", 3: "spleen"}
