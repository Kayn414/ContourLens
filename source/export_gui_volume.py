"""Dump a NIfTI volume as raw little-endian binary + a JSON sidecar, so the
C++ GUI (gui/) can load it without linking a NIfTI library (see the GUI
planning doc's "Path A" loader decision: raw+JSON for milestone 1, migrate to
nifti_clib later if arbitrary .nii.gz files need to be opened directly).

Conventions (write these once, don't rediscover them in C++):
- The array is dumped exactly as SimpleITK returns it: index order (k, j, i)
  i.e. (z, y, x), C-contiguous. Byte offset of voxel (i, j, k) is
  ((k * ny + j) * nx + i) * itemsize.
- spacing / origin / direction are in ITK's (x, y, z) axis order (NOT the
  array's (z, y, x) order) and describe physical space in **LPS**
  millimeters (SimpleITK/DICOM convention -- see the GUI planning doc's
  coordinate-system section).
- world = origin + direction_matrix @ (spacing * (i, j, k)), where
  direction_matrix is the 3x3 from `direction`, row-major.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

DEFAULT_CT = Path("data/phantom/nnunet_input/phantom_ent_0000.nii.gz")
DEFAULT_OUT_DIR = Path("data/phantom/gui_export")
DEFAULT_MASKS_DIR = Path("data/phantom/masks")


def load_volume_image(path: Path) -> sitk.Image:
    """Reads a CT (or any single-channel) volume from any of:
    - a directory of DICOM slices (one series -- if more than one, the
      largest by file count is used; DICOM is natively LPS, matching the
      convention this whole module writes),
    - a single-file NIfTI (.nii/.nii.gz) or NRRD (.nrrd/.nhdr) volume --
      SimpleITK's format sniffing handles both via plain ReadImage(), no
      special-casing needed.
    Either way, the result carries its own real spacing/origin/direction, so
    the rest of the pipeline (export_volume et al.) doesn't need to know
    which format/loader produced it.
    """
    if path.is_dir():
        series_ids = sitk.ImageSeriesReader_GetGDCMSeriesIDs(str(path))
        if not series_ids:
            raise FileNotFoundError(f"No DICOM series found in {path}")
        # Multiple series (e.g. scout + real acquisition) can share a directory; take the largest.
        series_file_lists = [sitk.ImageSeriesReader_GetGDCMSeriesFileNames(str(path), sid) for sid in series_ids]
        file_names = max(series_file_lists, key=len)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_names)
        return reader.Execute()
    return sitk.ReadImage(str(path))


def export_volume(source_path: Path, out_dir: Path, name: str) -> None:
    image = load_volume_image(source_path)
    array = sitk.GetArrayFromImage(image)  # (z, y, x)

    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{source_path}: expected an integer-valued volume, got {array.dtype}")
    if array.min() < np.iinfo(np.int16).min or array.max() > np.iinfo(np.int16).max:
        raise ValueError(f"{source_path}: value range [{array.min()}, {array.max()}] doesn't fit int16")
    array = array.astype("<i2")  # little-endian int16

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.bin"
    json_path = out_dir / f"{name}.json"

    array.tofile(bin_path)
    json_path.write_text(json.dumps({
        "shape": list(array.shape),  # (nz, ny, nx)
        "dtype": "int16",
        "spacing": list(image.GetSpacing()),  # (sx, sy, sz) mm
        "origin": list(image.GetOrigin()),    # (ox, oy, oz) mm, LPS
        "direction": list(image.GetDirection()),  # 3x3 row-major
    }, indent=2))

    print(f"{source_path} -> {bin_path} ({array.nbytes:,} bytes), {json_path}")


def export_label_volume(mask_dir: Path, reference_nifti: Path, out_dir: Path, name: str) -> None:
    """Combine every *.nii.gz binary mask in `mask_dir` into one uint8
    multi-label volume on the reference CT's grid (id = sorted-filename index
    + 1, 0 = background), + a JSON sidecar carrying id -> name via `labels`
    (labels[id - 1] == name). Masks are assumed already on that same grid
    (true for data/phantom/masks/, all built from the same CT read) --
    mismatched shapes raise rather than silently misalign.
    """
    reference = sitk.ReadImage(str(reference_nifti))
    ref_shape = sitk.GetArrayFromImage(reference).shape  # (z, y, x)

    mask_paths = sorted(mask_dir.glob("*.nii.gz"))
    if not mask_paths:
        raise FileNotFoundError(f"No *.nii.gz masks found in {mask_dir}")

    label_array = np.zeros(ref_shape, dtype=np.uint8)
    labels: list[str] = []
    for mask_path in mask_paths:
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
        if mask_array.shape != ref_shape:
            raise ValueError(f"{mask_path.name}: shape {mask_array.shape} != reference shape {ref_shape}")
        labels.append(mask_path.name.removesuffix(".nii.gz"))
        label_array[mask_array > 0] = len(labels)  # id = 1-based index just appended

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.bin"
    json_path = out_dir / f"{name}.json"

    label_array.tofile(bin_path)
    json_path.write_text(json.dumps({
        "shape": list(label_array.shape),
        "dtype": "uint8",
        "spacing": list(reference.GetSpacing()),
        "origin": list(reference.GetOrigin()),
        "direction": list(reference.GetDirection()),
        "labels": labels,
    }, indent=2))

    print(f"{mask_dir} ({len(labels)} structures: {', '.join(labels)}) -> {bin_path}, {json_path}")


def export_prediction_volume(prediction_nifti: Path, out_dir: Path, name: str = "prediction") -> None:
    """Export an nnU-Net multi-label prediction (source/predict.py's output)
    as a GUI overlay, same on-disk shape as export_label_volume's but built
    from source.data.hanseg.LABEL_IDS instead of a directory of binary masks
    -- nnU-Net already writes one file with the canonical ids baked in.
    """
    from source.data.hanseg import LABEL_IDS

    image = sitk.ReadImage(str(prediction_nifti))
    array = sitk.GetArrayFromImage(image)  # (z, y, x), uint8 label ids

    labels: list[str] = [""] * len(LABEL_IDS)
    for structure_name, label_id in LABEL_IDS.items():
        labels[label_id - 1] = structure_name

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.bin"
    json_path = out_dir / f"{name}.json"

    array.astype(np.uint8).tofile(bin_path)
    json_path.write_text(json.dumps({
        "shape": list(array.shape),
        "dtype": "uint8",
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "labels": labels,
    }, indent=2))

    print(f"{prediction_nifti} ({len(labels)} structures: {', '.join(labels)}) -> {bin_path}, {json_path}")


def export_dose(dose_dicom_path: Path, reference_nifti: Path, out_dir: Path, name: str = "dose") -> None:
    """Read an RTDOSE DICOM file, apply DoseGridScaling (SimpleITK's DICOM
    reader does NOT do this automatically -- it's an RT-specific tag, not the
    standard RescaleSlope/Intercept), resample onto the reference CT's grid
    (RTDOSE is usually coarser/offset from the CT), and dump as float32 Gy.
    """
    reference = sitk.ReadImage(str(reference_nifti))

    dose_image = sitk.Cast(sitk.ReadImage(str(dose_dicom_path)), sitk.sitkFloat32)
    scaling = float(pydicom.dcmread(str(dose_dicom_path), stop_before_pixels=True).DoseGridScaling)
    dose_image *= scaling

    if (dose_image.GetSize() != reference.GetSize()
            or dose_image.GetSpacing() != reference.GetSpacing()
            or dose_image.GetOrigin() != reference.GetOrigin()):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        dose_image = resampler.Execute(dose_image)

    array = sitk.GetArrayFromImage(dose_image).astype("<f4")  # (z, y, x) float32 Gy

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.bin"
    json_path = out_dir / f"{name}.json"

    array.tofile(bin_path)
    json_path.write_text(json.dumps({
        "shape": list(array.shape),
        "dtype": "float32",
        "spacing": list(reference.GetSpacing()),
        "origin": list(reference.GetOrigin()),
        "direction": list(reference.GetDirection()),
    }, indent=2))

    print(f"{dose_dicom_path} (x{scaling} Gy/unit) -> {bin_path} ({array.nbytes:,} bytes), {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nifti", type=Path, default=DEFAULT_CT,
        help="CT source: a NIfTI/NRRD file, or a directory of DICOM slices (one series).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", default="ct")
    parser.add_argument("--export-masks", action="store_true", help="Export the combined OAR label volume instead of the CT.")
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--dose", type=Path, help="Path to an RTDOSE DICOM file; exports it instead of the CT/masks.")
    parser.add_argument("--prediction", type=Path, help="Path to an nnU-Net prediction .nii.gz; exports it as a label-volume overlay.")
    args = parser.parse_args()

    if args.prediction:
        export_prediction_volume(args.prediction, args.out_dir, "prediction" if args.name == "ct" else args.name)
    elif args.dose:
        export_dose(args.dose, args.nifti, args.out_dir, "dose" if args.name == "ct" else args.name)
    elif args.export_masks:
        export_label_volume(args.masks_dir, args.nifti, args.out_dir, "masks" if args.name == "ct" else args.name)
    else:
        export_volume(args.nifti, args.out_dir, args.name)


if __name__ == "__main__":
    main()
