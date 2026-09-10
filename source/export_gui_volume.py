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
- The GUI's MPR slicing ignores `direction`, so CTs are reoriented to
  identity-direction LPS on export where that's a lossless axis
  permutation/flip (to_gui_orientation), and every overlay (masks,
  prediction, dose) is resampled onto the exported CT's grid (GuiGrid)
  rather than assumed to already share it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.misc import is_dicom

DEFAULT_CT = Path("data/phantom/nnunet_input/phantom_ent_0000.nii.gz")
DEFAULT_OUT_DIR = Path("data/phantom/gui_export")
DEFAULT_MASKS_DIR = Path("data/phantom/masks")

# DICOM objects that reference an image series but aren't one themselves.
_NON_IMAGE_MODALITIES = {"RTSTRUCT", "RTPLAN", "RTRECORD", "SEG", "REG", "SR"}

_GUI_DTYPES = {"int16": "<i2", "uint8": "u1", "float32": "<f4"}


@dataclass(frozen=True)
class DicomSeries:
    directory: Path
    series_uid: str
    file_names: list[str]


def is_dicom_file(path: Path) -> bool:
    return path.is_file() and (path.suffix.lower() == ".dcm" or is_dicom(str(path)))


def resolve_dicom_series(path: Path) -> DicomSeries | None:
    """The DICOM image series `path` refers to, or None if it isn't DICOM:
    - a directory: the largest series in it (e.g. a real acquisition next to
      a scout, or a CT next to its RTDOSE/RTIMAGE files),
    - a single .dcm file: the whole series that slice belongs to, from its
      directory (matched by SeriesInstanceUID, never by filename).
    """
    if path.is_dir():
        series_ids = sitk.ImageSeriesReader_GetGDCMSeriesIDs(str(path))
        if not series_ids:
            raise FileNotFoundError(f"No DICOM series found in {path}")
        candidates = [(sid, sitk.ImageSeriesReader_GetGDCMSeriesFileNames(str(path), sid)) for sid in series_ids]
        series_uid, file_names = max(candidates, key=lambda c: len(c[1]))
        return DicomSeries(path, series_uid, list(file_names))

    if not is_dicom_file(path):
        return None

    ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=["Modality", "SeriesInstanceUID"])
    modality = str(ds.get("Modality", ""))
    if modality in _NON_IMAGE_MODALITIES:
        hint = " -- load it as ground truth or prediction instead" if modality == "RTSTRUCT" else ""
        raise ValueError(f"{path.name} is a DICOM {modality}, not an image series{hint}")
    series_uid = str(ds.SeriesInstanceUID)
    file_names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(str(path.parent), series_uid)
    return DicomSeries(path.parent, series_uid, list(file_names) or [str(path)])


def read_dicom_series(series: DicomSeries) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series.file_names)
    image = reader.Execute()
    return sitk.JoinSeries(image) if image.GetDimension() == 2 else image  # single-slice series


def load_volume_image(path: Path) -> sitk.Image:
    """Reads a CT (or any single-channel) volume from any of:
    - a directory of DICOM slices, or one .dcm slice of a series (see
      resolve_dicom_series; DICOM is natively LPS, matching the convention
      this whole module writes),
    - a single-file NIfTI (.nii/.nii.gz) or NRRD (.nrrd/.nhdr) volume --
      SimpleITK's format sniffing handles both via plain ReadImage(), no
      special-casing needed.
    Either way, the result carries its own real spacing/origin/direction, so
    the rest of the pipeline (export_ct_image et al.) doesn't need to know
    which format/loader produced it.
    """
    series = resolve_dicom_series(path)
    if series is not None:
        return read_dicom_series(series)
    return sitk.ReadImage(str(path))


def to_gui_orientation(image: sitk.Image) -> sitk.Image:
    """Reorients an axis-aligned volume to identity-direction LPS (lossless:
    a permutation/flip of the voxel array), since the GUI's MPR panels slice
    straight along array axes. Common for NIfTI written by RAS-convention
    tools. Oblique volumes are returned unchanged (with a warning) -- they'd
    need an interpolating resample, which the GUI doesn't do yet either.
    """
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D volume, got a {image.GetDimension()}D image")
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    if np.allclose(direction, np.eye(3), atol=1e-5):
        return image
    if np.allclose(np.abs(direction), np.round(np.abs(direction)), atol=1e-5):
        print(f"Reorienting from direction {direction.round(3).tolist()} to identity (LPS) for the GUI")
        return sitk.DICOMOrient(image, "LPS")
    print(f"WARNING: oblique direction {direction.round(3).tolist()}; the GUI will show it unrotated")
    return image


@dataclass(frozen=True)
class GuiGrid:
    """The voxel grid of an exported volume -- in practice the case's CT,
    which every overlay gets resampled onto."""

    size: tuple[int, int, int]  # (nx, ny, nz), ITK order
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]  # 3x3 row-major

    @property
    def array_shape(self) -> tuple[int, int, int]:
        return (self.size[2], self.size[1], self.size[0])  # (nz, ny, nx)

    @classmethod
    def of(cls, image: sitk.Image) -> GuiGrid:
        if image.GetDimension() != 3:
            raise ValueError(f"Expected a 3D volume, got a {image.GetDimension()}D image")
        nx, ny, nz = image.GetSize()
        sx, sy, sz = image.GetSpacing()
        ox, oy, oz = image.GetOrigin()
        return cls((nx, ny, nz), (sx, sy, sz), (ox, oy, oz), tuple(image.GetDirection()))

    @classmethod
    def from_gui_json(cls, json_path: Path) -> GuiGrid:
        meta = json.loads(json_path.read_text())
        nz, ny, nx = meta["shape"]
        sx, sy, sz = meta["spacing"]
        ox, oy, oz = meta["origin"]
        return cls((nx, ny, nz), (sx, sy, sz), (ox, oy, oz), tuple(meta["direction"]))

    def matches(self, image: sitk.Image) -> bool:
        return (
            tuple(image.GetSize()) == self.size
            and np.allclose(image.GetSpacing(), self.spacing, rtol=0, atol=1e-4)
            and np.allclose(image.GetOrigin(), self.origin, rtol=0, atol=1e-3)
            and np.allclose(image.GetDirection(), self.direction, rtol=0, atol=1e-5)
        )

    def resample(self, image: sitk.Image, interpolator: int = sitk.sitkNearestNeighbor, default_value: float = 0.0) -> sitk.Image:
        """`image` on this grid, matched in physical space (so differing
        spacing/origin/orientation all line up); a no-op if it already is."""
        if self.matches(image):
            return image
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(list(self.size))
        resampler.SetOutputSpacing(list(self.spacing))
        resampler.SetOutputOrigin(list(self.origin))
        resampler.SetOutputDirection(list(self.direction))
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(default_value)
        return resampler.Execute(image)


def write_gui_volume(array: np.ndarray, grid: GuiGrid, out_dir: Path, name: str, dtype: str, labels: list[str] | None = None) -> Path:
    """Writes <out_dir>/<name>.bin + .json in the format documented above."""
    if tuple(array.shape) != grid.array_shape:
        raise ValueError(f"{name}: array shape {array.shape} != grid shape {grid.array_shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.bin"
    json_path = out_dir / f"{name}.json"

    np.ascontiguousarray(array, dtype=_GUI_DTYPES[dtype]).tofile(bin_path)
    meta: dict[str, object] = {
        "shape": list(array.shape),  # (nz, ny, nx)
        "dtype": dtype,
        "spacing": list(grid.spacing),  # (sx, sy, sz) mm
        "origin": list(grid.origin),  # (ox, oy, oz) mm, LPS
        "direction": list(grid.direction),  # 3x3 row-major
    }
    if labels is not None:
        meta["labels"] = labels  # labels[id - 1] == structure name
    json_path.write_text(json.dumps(meta, indent=2))
    return bin_path


def label_names_list(names_by_id: dict[int, str], max_id: int) -> list[str]:
    """The sidecar's `labels` list, sized to cover every id actually present
    (the GUI indexes labels[id - 1]); ids without a known name get `label_<id>`."""
    count = max([max_id, *names_by_id.keys()], default=0)
    return [names_by_id.get(i, f"label_{i}") for i in range(1, count + 1)]


# More distinct values than this in a supposed label map almost certainly
# means an intensity image (e.g. a CT) was dropped as a mask by mistake.
MAX_DISTINCT_LABELS = 100


def label_array_on_grid(image: sitk.Image, grid: GuiGrid, source_name: str) -> np.ndarray:
    """A multi-label image resampled (nearest-neighbor) onto `grid`, as a
    validated uint8 id array."""
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"{source_name}: expected a single-channel label map, got {image.GetNumberOfComponentsPerPixel()} channels")

    array = sitk.GetArrayFromImage(grid.resample(image, sitk.sitkNearestNeighbor))
    lo, hi = array.min(), array.max()
    if lo < 0 or hi > 255:
        raise ValueError(
            f"{source_name}: values span [{lo}, {hi}], which doesn't look like a label map "
            "(expected ids 0-255) -- did you mean to load it as the CT?"
        )
    if not np.issubdtype(array.dtype, np.integer) and not np.array_equal(array, np.rint(array)):
        raise ValueError(f"{source_name}: non-integer values, which doesn't look like a label map -- did you mean to load it as the CT?")

    array = array.astype(np.uint8)
    distinct = int(np.count_nonzero(np.bincount(array.reshape(-1), minlength=256)))
    if distinct > MAX_DISTINCT_LABELS:
        raise ValueError(
            f"{source_name}: {distinct} distinct values, which doesn't look like a label map -- "
            "did you mean to load it as the CT?"
        )
    return array


class LabelPainter:
    """Combines per-structure binary masks into one uint8 label volume.

    Where structures overlap, the SMALLER one keeps the voxel regardless of
    paint order -- so e.g. a lens stays visible inside an eyeball, or an
    organ inside a BODY/External contour, instead of whichever happened to
    be listed last winning.
    """

    def __init__(self, shape: tuple[int, int, int]):
        self.array = np.zeros(shape, dtype=np.uint8)
        self.names: list[str] = []
        self.overlap_voxels = 0
        self._sizes = np.full(256, np.inf)  # voxel count per id; background (0) never wins

    def add(self, name: str, mask: np.ndarray) -> bool:
        if mask.shape != self.array.shape:
            raise ValueError(f"{name}: mask shape {mask.shape} != label volume shape {self.array.shape}")
        voxels = np.flatnonzero(mask)
        if voxels.size == 0:
            print(f"  {name}: empty mask, skipping")
            return False
        if len(self.names) >= 255:
            raise ValueError("More than 255 structures don't fit in a uint8 label volume")

        label_id = len(self.names) + 1
        flat = self.array.reshape(-1)  # view: the array is C-contiguous
        current = flat[voxels]
        claimed = current != 0
        self.overlap_voxels += int(claimed.sum())
        keep = ~claimed | (self._sizes[current] > voxels.size)
        flat[voxels[keep]] = label_id

        self.names.append(name)
        self._sizes[label_id] = voxels.size
        return True

    def report_overlaps(self, source_name: str) -> None:
        if self.overlap_voxels:
            print(f"  {source_name}: {self.overlap_voxels:,} voxels claimed by >1 structure (smaller structure kept)")


def export_ct_image(image: sitk.Image, out_dir: Path, name: str = "ct", source_name: str = "") -> GuiGrid:
    """Writes `image` (already oriented -- see to_gui_orientation) as the
    GUI's int16 HU volume. Float volumes (common after resampling) are
    rounded and out-of-range values clipped, with a warning, rather than refused."""
    source_name = source_name or name
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"{source_name}: expected a single-channel volume, got {image.GetNumberOfComponentsPerPixel()} channels")

    grid = GuiGrid.of(image)
    array = sitk.GetArrayFromImage(image)  # (z, y, x)
    if np.issubdtype(array.dtype, np.floating):
        print(f"{source_name}: {array.dtype} voxels rounded to integer HU")
        np.rint(array, out=array)
    elif not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{source_name}: unsupported voxel type {array.dtype}")

    lo, hi = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    vmin, vmax = array.min(), array.max()
    if vmin < lo or vmax > hi:
        print(f"WARNING: {source_name}: value range [{vmin}, {vmax}] clipped to int16")
        array = np.clip(array, lo, hi)

    bin_path = write_gui_volume(array, grid, out_dir, name, "int16")
    print(f"{source_name} -> {bin_path} ({array.size * 2:,} bytes), {bin_path.with_suffix('.json')}")
    return grid


def export_volume(source_path: Path, out_dir: Path, name: str) -> GuiGrid:
    return export_ct_image(to_gui_orientation(load_volume_image(source_path)), out_dir, name, str(source_path))


def export_label_volume(mask_dir: Path, reference_nifti: Path, out_dir: Path, name: str) -> None:
    """Combine every *.nii.gz binary mask in `mask_dir` into one uint8
    multi-label volume on the reference CT's grid (id = sorted-filename index
    + 1, 0 = background, empty masks skipped), + a JSON sidecar carrying
    id -> name via `labels` (labels[id - 1] == name). Masks on a different
    grid are resampled onto it (nearest-neighbor) rather than refused.
    """
    grid = GuiGrid.of(to_gui_orientation(load_volume_image(reference_nifti)))

    mask_paths = sorted(mask_dir.glob("*.nii.gz"))
    if not mask_paths:
        raise FileNotFoundError(f"No *.nii.gz masks found in {mask_dir}")

    painter = LabelPainter(grid.array_shape)
    for mask_path in mask_paths:
        mask_image = grid.resample(sitk.ReadImage(str(mask_path)), sitk.sitkNearestNeighbor)
        painter.add(mask_path.name.removesuffix(".nii.gz"), sitk.GetArrayViewFromImage(mask_image) > 0)
    painter.report_overlaps(str(mask_dir))

    bin_path = write_gui_volume(painter.array, grid, out_dir, name, "uint8", painter.names)
    print(f"{mask_dir} ({len(painter.names)} structures: {', '.join(painter.names)}) -> {bin_path}")


def export_prediction_volume(
    prediction_nifti: Path,
    out_dir: Path,
    name: str = "prediction",
    label_ids: dict[str, int] | None = None,
    grid: GuiGrid | None = None,
) -> None:
    """Export an nnU-Net multi-label prediction (source/inference_backends.py's
    output) as a GUI overlay, same on-disk shape as export_label_volume's but
    built from a {structure_name: label_id} map instead of a directory of
    binary masks -- nnU-Net already writes one file with the ids baked in.

    Resampled onto `grid`, defaulting to <out_dir>/ct.json's (the CT the GUI
    shows) when that exists -- nnU-Net predicts on the source CT's native
    geometry, which differs from the export whenever it was reoriented.

    `label_ids` should come from whatever model actually produced
    `prediction_nifti` (source.inference_config.resolve_label_ids for the
    local backend, or the dict source/modal_nnunet.py's predict_nifti_bytes
    returns for the Modal backend) so this isn't tied to one model's
    structure set. Defaults to source.data.hanseg.LABEL_IDS only for this
    module's standalone `--prediction` CLI debugging tool.
    """
    if label_ids is None:
        from source.data.hanseg import LABEL_IDS
        label_ids = LABEL_IDS

    image = sitk.ReadImage(str(prediction_nifti))
    if grid is None:
        ct_json = out_dir / "ct.json"
        grid = GuiGrid.from_gui_json(ct_json) if ct_json.exists() else GuiGrid.of(to_gui_orientation(image))

    array = label_array_on_grid(image, grid, str(prediction_nifti))
    labels = label_names_list({label_id: structure_name for structure_name, label_id in label_ids.items()}, int(array.max()))

    bin_path = write_gui_volume(array, grid, out_dir, name, "uint8", labels)
    print(f"{prediction_nifti} ({len(labels)} structures: {', '.join(labels)}) -> {bin_path}")


def export_dose(dose_dicom_path: Path, reference_nifti: Path, out_dir: Path, name: str = "dose") -> None:
    """Read an RTDOSE DICOM file, apply DoseGridScaling (SimpleITK's DICOM
    reader does NOT do this automatically -- it's an RT-specific tag, not the
    standard RescaleSlope/Intercept), resample onto the reference CT's grid
    (RTDOSE is usually coarser/offset from the CT), and dump as float32 Gy.
    """
    export_dose_on_grid(dose_dicom_path, GuiGrid.of(to_gui_orientation(load_volume_image(reference_nifti))), out_dir, name)


def export_dose_on_grid(dose_dicom_path: Path, grid: GuiGrid, out_dir: Path, name: str = "dose") -> None:
    """export_dose onto an already-known grid, e.g. a case's exported CT
    (GuiGrid.from_gui_json) -- what the GUI's dose drop uses."""
    dose_image = sitk.Cast(sitk.ReadImage(str(dose_dicom_path)), sitk.sitkFloat32)
    scaling = float(pydicom.dcmread(str(dose_dicom_path), stop_before_pixels=True).DoseGridScaling)
    dose_image *= scaling
    dose_image = grid.resample(dose_image, sitk.sitkLinear)

    array = sitk.GetArrayFromImage(dose_image)  # (z, y, x) float32 Gy
    bin_path = write_gui_volume(array, grid, out_dir, name, "float32")
    print(f"{dose_dicom_path} (x{scaling} Gy/unit) -> {bin_path} ({array.size * 4:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nifti", type=Path, default=DEFAULT_CT,
        help="CT source: a NIfTI/NRRD file, a directory of DICOM slices (one series), or one .dcm slice.")
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
