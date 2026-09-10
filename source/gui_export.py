"""Exports a case's current prediction from the GUI (Metrics > Export):
- prediction.nii.gz + dataset.json ({"labels": {name: id}}), always.
  Dragging that file back onto the GUI keeps the structure names.
- prediction_rtstruct.dcm, a DICOM RTSTRUCT referencing the case's CT series,
  when that CT was loaded from DICOM, so a treatment planning system can
  import the model's contours.

gui/src/main.cpp writes metrics.csv/metrics.json into the same folder first;
structure colors come from metrics.json so the RTSTRUCT matches the viewer.

Usage: uv run python -m source.gui_export --case <case> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from source.export_gui_volume import DicomSeries, GuiGrid, read_dicom_series
from source.gui_load import case_dicom_series, gui_export_dir, stage_dicom_series
from source.gui_progress import configure_stdio, emit, emit_error, emit_notice


def read_gui_label_image(json_path: Path) -> tuple[sitk.Image, list[str]]:
    """A GUI label volume (.json + .bin) as a SimpleITK image with its real geometry."""
    meta = json.loads(json_path.read_text())
    array = np.fromfile(json_path.with_suffix(".bin"), dtype=np.uint8).reshape(meta["shape"])
    image = sitk.GetImageFromArray(array)
    image.SetSpacing([float(v) for v in meta["spacing"]])
    image.SetOrigin([float(v) for v in meta["origin"]])
    image.SetDirection([float(v) for v in meta["direction"]])
    return image, list(meta.get("labels", []))


def structure_colors(out_dir: Path) -> dict[str, list[int]]:
    """{name: [r, g, b]} from the GUI's metrics.json, or {} (rt-utils then picks colors)."""
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    rows = json.loads(metrics_path.read_text()).get("structures", [])
    return {row["name"]: [int(c) for c in row["color"]] for row in rows if "name" in row and "color" in row}


def export_nifti(image: sitk.Image, labels: list[str], out_dir: Path) -> Path:
    path = out_dir / "prediction.nii.gz"
    sitk.WriteImage(image, str(path))
    names = {"background": 0} | {name: label_id for label_id, name in enumerate(labels, start=1) if name}
    (out_dir / "dataset.json").write_text(json.dumps({"labels": names}, indent=2))
    return path


def export_rtstruct(
    image: sitk.Image, labels: list[str], series: DicomSeries, out_dir: Path, colors: dict[str, list[int]]
) -> Path:
    """One ROI per non-empty structure, contoured on `series`' own slices."""
    from rt_utils import RTStructBuilder

    emit("converting", message="Resampling the prediction onto the DICOM series grid")
    native = sitk.GetArrayFromImage(GuiGrid.of(read_dicom_series(series)).resample(image, sitk.sitkNearestNeighbor))

    path = out_dir / "prediction_rtstruct.dcm"
    with tempfile.TemporaryDirectory() as tmp:
        stage_dicom_series(series, Path(tmp))
        rtstruct = RTStructBuilder.create_new(dicom_series_path=tmp)
        rtstruct.set_series_description("dicom_rt GUI prediction")
        written = 0
        for label_id, name in enumerate(labels, start=1):
            mask = native == label_id
            if not name or not mask.any():
                continue
            emit("converting", label_id - 1, len(labels), f"Contouring {name}")
            # rt-utils masks are (rows, cols, slices) in its slice order: ascending position, as SimpleITK reads the series.
            color = {"color": colors[name]} if name in colors else {}  # without one, rt-utils picks a color
            rtstruct.add_roi(mask=np.transpose(mask, (1, 2, 0)), name=name, **color)
            written += 1
        if written == 0:
            raise ValueError("The prediction has no non-empty structures to export")
        rtstruct.save(str(path))
    return path


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        prediction_json = gui_export_dir(args.case) / "prediction.json"
        if not prediction_json.exists():
            raise FileNotFoundError(f"No prediction to export for case {args.case!r} ({prediction_json} not found)")
        args.out.mkdir(parents=True, exist_ok=True)

        emit("writing", message="Writing prediction.nii.gz")
        image, labels = read_gui_label_image(prediction_json)
        export_nifti(image, labels, args.out)

        series = case_dicom_series(args.case)
        if series is None:
            emit_notice("RTSTRUCT export needs a CT loaded from a DICOM series; this case's CT wasn't, so only NIfTI was written.")
        else:
            rtstruct_path = export_rtstruct(image, labels, series, args.out, structure_colors(args.out))
            print(f"Wrote {rtstruct_path}")
        emit("done", fraction=1.0, message=f"Exported to {args.out}")
    except Exception as e:
        traceback.print_exc()
        emit_error(f"{type(e).__name__}: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
