"""Runs inference (locally or on a cloud backend, per that case's
inference_config.json) on the given case's CT, and exports the result as a
GUI overlay -- this is what the "3D" GUI's "Run Inference" button
(gui/src/inference.{h,cpp}) subprocess-launches, passing the loaded case's
directory name as argv[1] (e.g. "phantom" or "hanseg_cases/case_03"). Exit
code 0 = success; the GUI reloads
data/<case>/gui_export/prediction(.bin/.json) after this exits. Progress and
failures are reported to the GUI via source/gui_progress.py.

Usage: uv run python -m source.run_inference_for_gui <case_dir>
"""

import sys
import tempfile
import traceback
from pathlib import Path

from source.export_gui_volume import export_prediction_volume
from source.gui_progress import configure_stdio, emit, emit_error
from source.inference_backends import run
from source.inference_config import load_config


def main() -> None:
    configure_stdio()
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <case_dir>  (e.g. 'phantom' or 'hanseg_cases/case_03')", file=sys.stderr)
        raise SystemExit(2)

    case_name = sys.argv[1]
    case_dir = Path("data") / case_name
    try:
        config = load_config(case_name)
        print(f"Running inference for {case_name} via the {config.backend!r} backend...")

        with tempfile.TemporaryDirectory() as tmp:
            prediction_path, label_ids = run(config, Path(tmp))
            emit("exporting", message="Writing the prediction overlay")
            export_prediction_volume(prediction_path, case_dir / "gui_export", label_ids=label_ids)
        emit("done", fraction=1.0, message="Inference complete")
    except Exception as e:
        traceback.print_exc()
        emit_error(f"{type(e).__name__}: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
