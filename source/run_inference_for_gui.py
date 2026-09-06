"""Runs nnU-Net inference on the phantom CT and exports the result as a GUI
overlay, in one step -- this is what the "3D" GUI's "Run Inference" button
(gui/src/inference.{h,cpp}) subprocess-launches. Exit code 0 = success; the
GUI reloads data/phantom/gui_export/prediction(.bin/.json) after this exits.
"""

from source.data.phantom import CASE_ID
from source.export_gui_volume import DEFAULT_OUT_DIR, export_prediction_volume
from source.predict import PREDICTION_DIR, predict


def main() -> None:
    predict()
    export_prediction_volume(PREDICTION_DIR / f"{CASE_ID}.nii.gz", DEFAULT_OUT_DIR)


if __name__ == "__main__":
    main()
