"""Run nnU-Net inference on the phantom CT and produce a multi-label prediction.

Requires a trained model under nnUNet_results (locally via scripts/train_nnunet.sh,
or downloaded from the Modal results volume -- see source/modal_nnunet.py).
"""

import os
import subprocess
from pathlib import Path

from source.data.nnunet_dataset import DATASET_ID, nnunet_env, resolve_nnunet_results
from source.data.phantom import NNUNET_INPUT_DIR, build_ct_nifti

CONFIGURATION = "3d_fullres"
# Matches the completed production run (source/modal_nnunet.py's
# PRODUCTION_TRAINER/PRODUCTION_FOLDS) -- fold 4 was skipped for budget.
TRAINER = "nnUNetTrainer_500epochs"
PLANS = "nnUNetPlans"
FOLDS = ("0", "1", "2", "3")

PREDICTION_DIR = Path("data/phantom/nnunet_prediction")


def predict(
    input_dir: Path = NNUNET_INPUT_DIR,
    output_dir: Path = PREDICTION_DIR,
    dataset_id: int | str = DATASET_ID,
    configuration: str = CONFIGURATION,
    folds: tuple[str, ...] = FOLDS,
    trainer: str = TRAINER,
    plans: str = PLANS,
) -> Path:
    if not any(input_dir.glob("*_0000.nii.gz")):
        build_ct_nifti(out_dir=input_dir)

    results_dir = resolve_nnunet_results()
    if not results_dir.exists():
        raise FileNotFoundError(
            f"{results_dir} does not exist — no trained model found. "
            "Set the nnUNet_results env var, or train first (see scripts/train_nnunet.sh "
            "or `modal run source.modal_nnunet::main --step train`)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", configuration,
        "-f", *folds,
        "-tr", trainer,
        "-p", plans,
    ]
    print("Running:", " ".join(cmd))
    env = {**os.environ, **nnunet_env()}
    subprocess.run(cmd, check=True, env=env)
    print(f"Prediction written -> {output_dir}")
    return output_dir


if __name__ == "__main__":
    predict()
