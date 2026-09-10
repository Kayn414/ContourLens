"""Converts data dragged into the GUI (gui/src/main.cpp's drop handler) into
the raw+JSON files it loads (format: source/export_gui_volume.py's
docstring). Progress/errors are reported via source/gui_progress.py.

  uv run python -m source.gui_load ct <src> --case <case_name>
      <src>: a NIfTI/NRRD file, a DICOM series directory, or one .dcm slice
      of a series. Writes data/<case>/gui_export/ct.* (removing stale
      masks/prediction/dose overlays there). If <src> is DICOM and its
      directory also holds an RTSTRUCT referencing that same series, it is
      exported as ground truth (masks.*) in the same run.

  uv run python -m source.gui_load masks <src> [<src> ...] --case <case_name> --slot gt|prediction
      <src>: one multi-label NIfTI/NRRD file, one or more .seg.nrrd files
      (or a directory of them), or one DICOM RTSTRUCT file. Resampled
      (nearest-neighbor) onto the case's exported CT grid, so it needn't
      share the CT's exact geometry -- only its physical space.

  uv run python -m source.gui_load dose <RTDOSE.dcm> --case <case_name>
      Applies DoseGridScaling and resamples the dose onto the case's CT grid
      (data/<case>/gui_export/dose.*), for the GUI's DVH.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError

from source.export_gui_volume import (
    DicomSeries,
    GuiGrid,
    LabelPainter,
    export_ct_image,
    export_dose_on_grid,
    label_array_on_grid,
    label_names_list,
    load_volume_image,
    read_dicom_series,
    resolve_dicom_series,
    to_gui_orientation,
    write_gui_volume,
)
from source.gui_progress import configure_stdio, emit, emit_error, emit_notice

DATA_ROOT = Path("data")
SLOT_FILE_NAMES = {"gt": "masks", "prediction": "prediction"}
_VOLUME_SUFFIXES = (".seg.nrrd", ".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")


def gui_export_dir(case: str) -> Path:
    return DATA_ROOT / case / "gui_export"


def case_dicom_series(case: str) -> DicomSeries | None:
    """The DICOM series a case's CT was loaded from (its inference config's
    source_ct), or None when it came from NIfTI/NRRD or the case has no config."""
    config_file = Path("configs") / case / "inference_config.json"
    if not config_file.exists():
        return None
    source_ct = json.loads(config_file.read_text()).get("source_ct")
    if not source_ct:
        return None
    try:
        return resolve_dicom_series(Path(source_ct))
    except (FileNotFoundError, ValueError):
        return None


def stage_dicom_series(series: DicomSeries, directory: Path) -> None:
    """Copies only `series`' slices into `directory`, for rt-utils: it takes
    every pixel-data file in a directory as part of the series, which breaks
    on RTDOSE/RTIMAGE files next to the CT (see source/data/phantom.py's
    _stage_ct_only_series)."""
    for i, file_name in enumerate(series.file_names):
        shutil.copyfile(file_name, directory / f"{i:05d}.dcm")


def strip_volume_suffix(file_name: str) -> str:
    lower = file_name.lower()
    for suffix in _VOLUME_SUFFIXES:
        if lower.endswith(suffix):
            return file_name[: -len(suffix)]
    return file_name


def structure_name_from_filename(path: Path) -> str:
    """"case_03_OAR_OpticNrv_L.seg.nrrd" -> "OpticNrv_L" (HaN-Seg naming),
    "brainstem.seg.nrrd" -> "brainstem"."""
    return re.sub(r"^case_\d+_OAR_", "", strip_volume_suffix(path.name))


# ============================================================
# Label names for multi-label files
# ============================================================
def parse_label_names(obj: object) -> dict[int, str]:
    """{label_id: name} from any of the label-map JSON shapes in the wild:
    nnU-Net v2 dataset.json ({"labels": {name: id}}), nnU-Net v1
    ({"labels": {"1": name}}), this GUI's own sidecar ({"labels": [names]}),
    or a bare dict of either orientation. Region-based (list-valued) nnU-Net
    labels and background (id 0) are skipped."""
    if isinstance(obj, dict) and "labels" in obj:
        obj = obj["labels"]
    if isinstance(obj, list):
        return {i + 1: str(name) for i, name in enumerate(obj) if name}

    names: dict[int, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, int) and not isinstance(value, bool):
                names[value] = str(key)
            elif isinstance(value, str) and str(key).isdigit():
                names[int(key)] = value
    return {label_id: name for label_id, name in names.items() if 0 < label_id <= 255}


# How many folders above a label file to look for dataset.json/labels.json.
# nnU-Net keeps dataset.json in the dataset root, one level above labelsTr/
# (e.g. nnUNet_raw/Dataset501_HanSeg/{dataset.json,labelsTr/case_09.nii.gz});
# searching further up would risk picking up an unrelated dataset's names.
SIDECAR_PARENT_LEVELS = 2


def sidecar_label_names(label_path: Path) -> dict[int, str] | None:
    """{label_id: name} for a multi-label file, from the nearest label-map
    JSON: <stem>.json beside it, then dataset.json/labels.json in its own
    folder and up to SIDECAR_PARENT_LEVELS folders above."""
    candidates = [label_path.with_name(strip_volume_suffix(label_path.name) + ".json")]
    directory = label_path.resolve().parent
    for _ in range(SIDECAR_PARENT_LEVELS + 1):
        candidates += [directory / "dataset.json", directory / "labels.json"]
        if directory.parent == directory:
            break
        directory = directory.parent

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            names = parse_label_names(json.loads(candidate.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if names:
            print(f"Label names from {candidate}")
            return names
    return None


# ============================================================
# Mask sources
# ============================================================
def multilabel_masks(path: Path, grid: GuiGrid, fallback_names: dict[int, str] | None = None) -> tuple[np.ndarray, list[str]]:
    emit("reading", message=f"Reading {path.name}")
    image = sitk.ReadImage(str(path))
    emit("converting", message="Resampling onto the CT grid")
    array = label_array_on_grid(image, grid, path.name)
    names = sidecar_label_names(path) or fallback_names or {}

    present_ids = np.flatnonzero(np.bincount(array.reshape(-1), minlength=256))[1:].tolist()  # skip background
    unnamed = [label_id for label_id in present_ids if label_id not in names]
    if unnamed:
        where = "the label-map JSON found" if names else f"no dataset.json/labels.json within {SIDECAR_PARENT_LEVELS} folders above {path.name}"
        print(f"WARNING: {path.name}: label ids {unnamed} have no name ({where}); shown as label_<id>")
    return array, label_names_list(names, int(array.max()))


def _slicer_segments(image: sitk.Image) -> list[tuple[str, int, int]]:
    """(name, label value, layer) per segment from 3D Slicer's .seg.nrrd
    metadata; empty for a plain binary mask (e.g. HaN-Seg's)."""
    keys = set(image.GetMetaDataKeys())
    segments = []
    i = 0
    while f"Segment{i}_Name" in keys:
        value_key, layer_key = f"Segment{i}_LabelValue", f"Segment{i}_Layer"
        value = int(image.GetMetaData(value_key)) if value_key in keys else i + 1
        layer = int(image.GetMetaData(layer_key)) if layer_key in keys else 0
        segments.append((image.GetMetaData(f"Segment{i}_Name"), value, layer))
        i += 1
    return segments


def seg_nrrd_masks(paths: list[Path], grid: GuiGrid) -> tuple[np.ndarray, list[str]]:
    painter = LabelPainter(grid.array_shape)
    for n, path in enumerate(paths):
        emit("converting", n, len(paths), f"Reading {path.name}")
        image = sitk.ReadImage(str(path))
        segments = _slicer_segments(image)

        if not segments:
            if image.GetNumberOfComponentsPerPixel() != 1:
                raise ValueError(f"{path.name}: multi-layer segmentation without Slicer segment metadata")
            mask = grid.resample(image > 0, sitk.sitkNearestNeighbor)
            painter.add(structure_name_from_filename(path), sitk.GetArrayViewFromImage(mask) > 0)
            continue

        for name, value, layer in segments:
            layer_image = sitk.VectorIndexSelectionCast(image, layer) if image.GetNumberOfComponentsPerPixel() > 1 else image
            mask = grid.resample(layer_image == value, sitk.sitkNearestNeighbor)
            painter.add(name, sitk.GetArrayViewFromImage(mask) > 0)

    painter.report_overlaps("segmentations")
    if not painter.names:
        raise ValueError("All dropped segmentations were empty")
    return painter.array, painter.names


def rtstruct_referenced_series_uids(ds: pydicom.Dataset) -> set[str]:
    uids = set()
    for frame in ds.get("ReferencedFrameOfReferenceSequence", []):
        for study in frame.get("RTReferencedStudySequence", []):
            for series in study.get("RTReferencedSeriesSequence", []):
                uids.add(str(series.SeriesInstanceUID))
    return uids


def find_rtstructs(series: DicomSeries) -> list[Path]:
    """RTSTRUCT files next to `series` that reference it (or reference
    nothing at all, which some exporters omit)."""
    found = []
    for path in sorted(series.directory.iterdir()):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=["Modality"])
        except (InvalidDicomError, OSError):
            continue
        if ds.get("Modality") != "RTSTRUCT":
            continue
        refs = rtstruct_referenced_series_uids(pydicom.dcmread(str(path), stop_before_pixels=True))
        if not refs or series.series_uid in refs:
            found.append(path)
    return found


def rasterize_rtstruct(
    rtstruct_path: Path, series: DicomSeries, grid: GuiGrid, series_image: sitk.Image | None = None
) -> tuple[np.ndarray, list[str]]:
    """Every contoured ROI in the RTSTRUCT, as a label volume on `grid`."""
    from rt_utils import RTStructBuilder

    with tempfile.TemporaryDirectory() as tmp:
        emit("converting", message=f"Staging {len(series.file_names)} CT slices for RTSTRUCT rasterization")
        stage_dicom_series(series, Path(tmp))

        emit("converting", message=f"Parsing {rtstruct_path.name}")
        rtstruct = RTStructBuilder.create_from(dicom_series_path=tmp, rt_struct_path=str(rtstruct_path))
        if series_image is None:
            series_image = read_dicom_series(series)

        nx, ny, nz = series_image.GetSize()
        painter = LabelPainter((nz, ny, nx))
        roi_names = rtstruct.get_roi_names()
        for n, roi_name in enumerate(roi_names):
            emit("converting", n, len(roi_names), f"Rasterizing {roi_name}")
            try:
                mask = rtstruct.get_roi_mask_by_name(roi_name)  # (rows, cols, slices)
            except Exception as e:  # noqa: BLE001 -- vendor RTSTRUCTs vary widely; one bad ROI shouldn't sink the rest
                print(f"  {roi_name}: skipped ({type(e).__name__}: {e})")
                continue
            painter.add(roi_name, np.transpose(mask, (2, 0, 1)))
        painter.report_overlaps(rtstruct_path.name)

    if not painter.names:
        raise ValueError(f"{rtstruct_path.name}: no ROI with contours")

    label_image = sitk.GetImageFromArray(painter.array)
    label_image.CopyInformation(series_image)
    emit("converting", message="Resampling onto the CT grid")
    array = sitk.GetArrayFromImage(grid.resample(label_image, sitk.sitkNearestNeighbor))
    return array, painter.names


def rtstruct_masks(rtstruct_path: Path, grid: GuiGrid, case: str) -> tuple[np.ndarray, list[str]]:
    ds = pydicom.dcmread(str(rtstruct_path), stop_before_pixels=True)
    modality = ds.get("Modality", "")
    if modality != "RTSTRUCT":
        raise ValueError(f"{rtstruct_path.name} is a DICOM {modality or '(no modality)'} file, not an RTSTRUCT -- load image slices as CT instead")
    refs = rtstruct_referenced_series_uids(ds)

    # The referenced CT series: next to the RTSTRUCT (typical of TPS exports),
    # else the series this case's CT was loaded from.
    candidates: list[DicomSeries] = []
    directory = rtstruct_path.parent
    for series_uid in sitk.ImageSeriesReader_GetGDCMSeriesIDs(str(directory)):
        if series_uid in refs:
            file_names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(str(directory), series_uid)
            candidates.append(DicomSeries(directory, series_uid, list(file_names)))

    series = case_dicom_series(case)
    if series is not None and (not refs or series.series_uid in refs):
        candidates.append(series)

    if not candidates:
        wanted = ", ".join(sorted(refs)) or "(not recorded in the RTSTRUCT)"
        raise FileNotFoundError(
            f"Couldn't find the CT series {rtstruct_path.name} references (SeriesInstanceUID {wanted}). "
            "Keep the RTSTRUCT in the same folder as its CT slices, or load the CT from that DICOM series first."
        )
    return rasterize_rtstruct(rtstruct_path, candidates[0], grid)


def classify_mask_sources(sources: list[Path]) -> tuple[str, list[Path]]:
    files: list[Path] = []
    for source in sources:
        if source.is_dir():
            files += sorted(p for p in source.iterdir() if p.name.lower().endswith(".seg.nrrd"))
        elif source.exists():
            files.append(source)
        else:
            raise FileNotFoundError(f"{source} does not exist")
    if not files:
        raise FileNotFoundError("No .seg.nrrd files found in the dropped folder(s)")

    lower = [f.name.lower() for f in files]
    if all(name.endswith(".seg.nrrd") for name in lower):
        return "seg_nrrd", files
    if len(files) == 1 and lower[0].endswith(".dcm"):
        return "rtstruct", files
    if len(files) == 1 and lower[0].endswith(_VOLUME_SUFFIXES):
        return "multilabel", files
    raise ValueError(
        "Unsupported mask selection: drop one multi-label NIfTI/NRRD file, one RTSTRUCT .dcm, "
        f"or one or more .seg.nrrd files (got: {', '.join(f.name for f in files)})"
    )


# ============================================================
# Entry points
# ============================================================
def load_ct(src: Path, case: str) -> None:
    out_dir = gui_export_dir(case)

    emit("reading", message=f"Reading {src.name}")
    series = resolve_dicom_series(src)
    raw_image = read_dicom_series(series) if series is not None else load_volume_image(src)

    emit("converting", message="Orienting for the viewer")
    image = to_gui_orientation(raw_image)

    for stem in ("ct", "masks", "prediction", "dose"):
        for ext in (".bin", ".json"):
            (out_dir / f"{stem}{ext}").unlink(missing_ok=True)

    emit("writing", message=f"Writing {out_dir / 'ct.bin'}")
    grid = export_ct_image(image, out_dir, "ct", src.name)
    del image

    if series is not None:
        rtstructs = find_rtstructs(series)
        if len(rtstructs) > 1:
            print(f"Found {len(rtstructs)} RTSTRUCTs for this series; using {rtstructs[0].name}")
        if rtstructs:
            try:
                array, names = rasterize_rtstruct(rtstructs[0], series, grid, series_image=raw_image)
                write_gui_volume(array, grid, out_dir, SLOT_FILE_NAMES["gt"], "uint8", names)
                print(f"Ground truth from {rtstructs[0].name}: {', '.join(names)}")
            except Exception as e:  # noqa: BLE001 -- the CT itself loaded fine; don't fail the whole job over its contours
                traceback.print_exc()
                print(f"WARNING: couldn't load ground truth from {rtstructs[0].name}: {e}")

    emit("done", fraction=1.0, message=f"Loaded CT {grid.size[0]}x{grid.size[1]}x{grid.size[2]}")


def load_masks(sources: list[Path], case: str, slot: str) -> None:
    out_dir = gui_export_dir(case)
    ct_json = out_dir / "ct.json"
    if not ct_json.exists():
        raise FileNotFoundError(f"{ct_json} not found -- load a CT for case {case!r} first")
    grid = GuiGrid.from_gui_json(ct_json)

    kind, paths = classify_mask_sources(sources)
    if kind == "seg_nrrd":
        array, names = seg_nrrd_masks(paths, grid)
    elif kind == "rtstruct":
        array, names = rtstruct_masks(paths[0], grid, case)
    else:
        fallback = None
        if slot == "prediction":  # a prediction from this case's configured model: use its label names
            try:
                from source.inference_config import load_config, resolve_label_ids

                fallback = {i: name for name, i in resolve_label_ids(load_config(case).model).items()}
            except Exception as e:  # noqa: BLE001 -- names are optional; any config/model lookup failure falls back to label_<id>
                print(f"(No label names from the case's model config: {e})")
        array, names = multilabel_masks(paths[0], grid, fallback)

    name = SLOT_FILE_NAMES[slot]
    emit("writing", message=f"Writing {out_dir / (name + '.bin')}")
    write_gui_volume(array, grid, out_dir, name, "uint8", names)
    emit("done", fraction=1.0, message=f"Loaded {len(names)} structures")


def load_dose(src: Path, case: str) -> None:
    out_dir = gui_export_dir(case)
    ct_json = out_dir / "ct.json"
    if not ct_json.exists():
        raise FileNotFoundError(f"{ct_json} not found -- load a CT for case {case!r} first")

    emit("reading", message=f"Reading {src.name}")
    ds = pydicom.dcmread(str(src), stop_before_pixels=True)
    modality = ds.get("Modality", "")
    if modality != "RTDOSE":
        raise ValueError(f"{src.name} is a DICOM {modality or '(no modality)'} file, not an RTDOSE")

    series = case_dicom_series(case)
    if series is not None:
        ct = pydicom.dcmread(series.file_names[0], stop_before_pixels=True, specific_tags=["FrameOfReferenceUID"])
        ct_frame, dose_frame = str(ct.get("FrameOfReferenceUID", "")), str(ds.get("FrameOfReferenceUID", ""))
        if ct_frame and dose_frame and ct_frame != dose_frame:
            emit_notice(f"{src.name} uses a different frame of reference than this case's CT, so the dose may be misaligned.")

    emit("converting", message="Resampling the dose onto the CT grid")
    export_dose_on_grid(src, GuiGrid.from_gui_json(ct_json), out_dir)
    emit("done", fraction=1.0, message=f"Loaded dose from {src.name}")


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    ct = sub.add_parser("ct")
    ct.add_argument("src", type=Path)
    ct.add_argument("--case", required=True)

    masks = sub.add_parser("masks")
    masks.add_argument("src", type=Path, nargs="+")
    masks.add_argument("--case", required=True)
    masks.add_argument("--slot", choices=sorted(SLOT_FILE_NAMES), default="gt")

    dose = sub.add_parser("dose")
    dose.add_argument("src", type=Path)
    dose.add_argument("--case", required=True)

    args = parser.parse_args()
    try:
        if args.command == "ct":
            load_ct(args.src, args.case)
        elif args.command == "dose":
            load_dose(args.src, args.case)
        else:
            load_masks(args.src, args.case, args.slot)
    except Exception as e:
        traceback.print_exc()
        emit_error(f"{type(e).__name__}: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
