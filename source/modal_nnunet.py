from pathlib import Path

from source.data.nnunet_dataset import (
    DATASET_FOLDER_NAME,
    DATASET_ID,
    resolve_nnunet_raw,
)
from source.modal_utils import (
    RAW_VOLUME_NAME,
    VOL_PREPROCESSED,
    VOL_RAW,
    VOL_RESULTS,
    app,
    image,
    preprocessed_vol,
    raw_volume,
    results_vol,
    submit_commands,
)

# ============ Config ==============
CONFIGURATION = "3d_fullres"
FOLD = 0
FOLDS = (0, 1, 2, 3, 4)
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetPlans"
USE_NPZ = True


GPU = "A10G"


PRODUCTION_TRAINER = "nnUNetTrainer_500epochs"
PRODUCTION_FOLDS = (0, 1, 2, 3)
WARMSTART_CHECKPOINT = (
    f"{VOL_RESULTS}/{DATASET_FOLDER_NAME}/nnUNetTrainer_100epochs__{PLANS}__{CONFIGURATION}/fold_0/checkpoint_final.pth"
)


def upload_dataset(local_dir: Path | None = None) -> None:
    """Push the locally-built nnU-Net raw dataset onto the Modal raw volume.
    """
    import subprocess

    local_dir = local_dir or (resolve_nnunet_raw() / DATASET_FOLDER_NAME)
    if not local_dir.exists():
        raise FileNotFoundError(f"{local_dir} does not exist — run `python -m source.data.nnunet_dataset` first")

    print(f"Uploading {local_dir} -> volume:/{DATASET_FOLDER_NAME} ...")
    subprocess.run(
        ["modal", "volume", "put", "-f", RAW_VOLUME_NAME, str(local_dir), f"/{DATASET_FOLDER_NAME}"],
        check=True,
    )
    print("Upload finished.")


@app.function(
    image=image,
    gpu="A10G",                           
    cpu=8,
    memory=32768,                         # 32 GB RAM recommended
    volumes={
        VOL_RAW: raw_volume,
        VOL_PREPROCESSED: preprocessed_vol,
    },
    timeout=4 * 60 * 60,                  # 4 hours
)
def plan_and_preprocess(dataset_id: int | str = DATASET_ID):
    import subprocess

    print("Starting plan_and_preprocess...")
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "--verify_dataset_integrity",
        "-np", "4",                       # number of processes
    ]
    subprocess.run(cmd, check=True)

    preprocessed_vol.commit()
    print("Preprocessing finished and committed to volume.")



@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=32768,
    volumes={VOL_RAW: raw_volume, VOL_RESULTS: results_vol},
    timeout=60 * 60,
)
def predict_case(
    case_id: str,
    dataset_id: str = str(DATASET_ID),
    configuration: str = CONFIGURATION,
    folds: str = ",".join(str(f) for f in PRODUCTION_FOLDS),  # comma-separated -- Modal's CLI can't parse tuple[int, ...]
    trainer: str = PRODUCTION_TRAINER,
    plans: str = PLANS,
) -> str:
    """Predicts one case already present on the raw volume's imagesTr/, and
    writes the result to <results volume>/predictions/<case_id>.nii.gz.
    Returns that path (fetch it with `modal volume get`)."""
    import shutil
    import subprocess
    from pathlib import Path

    input_dir = Path("/tmp/predict_input")
    output_dir = Path("/tmp/predict_output")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = VOL_RAW / DATASET_FOLDER_NAME / "imagesTr" / f"{case_id}_0000.nii.gz"
    shutil.copyfile(src, input_dir / f"{case_id}_0000.nii.gz")

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir), "-o", str(output_dir),
        "-d", str(dataset_id), "-c", configuration,
        "-f", *folds.split(","),
        "-tr", trainer, "-p", plans,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    dest = Path(str(VOL_RESULTS)) / "predictions" / f"{case_id}.nii.gz"  # VOL_RESULTS is a PurePosixPath (no I/O methods)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_dir / f"{case_id}.nii.gz", dest)
    results_vol.commit()
    print(f"Wrote {dest}")
    return str(dest)


# ============================================================
# 2c. Predict (arbitrary CT bytes) -- the GUI's "modal" inference backend
# ============================================================
# Unlike predict_case, this doesn't assume the CT is already staged on the
# raw volume: it takes the NIfTI file's bytes directly as an argument, so it
# works for a user's own dataset (source/inference_backends.py's run_modal
# re-encodes whatever format -- DICOM dir, NRRD, NIfTI -- through
# SimpleITK first). Only the trained model needs to already exist on the
# results volume, at the usual nnU-Net layout for the given
# dataset_id/trainer/plans/configuration. Must be `modal deploy`ed (not just
# `modal run`) for Function.from_name() to find it from an ad hoc script.
@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=32768,
    volumes={VOL_RESULTS: results_vol},
    timeout=60 * 60,
)
def predict_nifti_bytes(
    nifti_bytes: bytes,
    dataset_id: str = str(DATASET_ID),
    configuration: str = CONFIGURATION,
    folds: str = ",".join(str(f) for f in PRODUCTION_FOLDS),
    trainer: str = PRODUCTION_TRAINER,
    plans: str = PLANS,
) -> tuple[bytes, dict[str, int]]:
    """Predicts one CT given as raw NIfTI bytes; returns (prediction NIfTI
    bytes, {structure_name: label_id}). The label map is read here, from the
    trained model's own dataset.json on the results volume, rather than
    hardcoded -- generic to whatever dataset_id/trainer/plans/configuration
    is requested, not just the HanSeg model this project ships with. The
    caller (source/inference_backends.py's run_modal) has no filesystem
    access to this volume, so it can't resolve this itself."""
    import json
    import subprocess
    from pathlib import Path

    input_dir = Path("/tmp/predict_input")
    output_dir = Path("/tmp/predict_output")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_id = "case"
    (input_dir / f"{case_id}_0000.nii.gz").write_bytes(nifti_bytes)

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir), "-o", str(output_dir),
        "-d", str(dataset_id), "-c", configuration,
        "-f", *folds.split(","),
        "-tr", trainer, "-p", plans,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    results_dir = Path(str(VOL_RESULTS))  # VOL_RESULTS is a PurePosixPath (no I/O methods)
    matches = sorted(results_dir.glob(f"Dataset{int(dataset_id):03d}_*"))
    if not matches:
        raise FileNotFoundError(f"No Dataset{int(dataset_id):03d}_* folder under {results_dir}")
    dataset_json = matches[0] / f"{trainer}__{plans}__{configuration}" / "dataset.json"
    labels = json.loads(dataset_json.read_text())["labels"]
    label_ids = {name: label_id for name, label_id in labels.items() if name != "background"}

    return (output_dir / f"{case_id}.nii.gz").read_bytes(), label_ids


# ============================================================
# 3. Train
# ============================================================
@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=32768,                         # 32 GB system RAM
    volumes={
        VOL_RAW: raw_volume,
        VOL_PREPROCESSED: preprocessed_vol,
        VOL_RESULTS: results_vol,
    },
    timeout=24 * 60 * 60,                 # 24 hours (nnU-Net can be long)
    retries=1,
)
def train(
    dataset_id: int | str = DATASET_ID,
    configuration: str = CONFIGURATION,
    fold: int | str = FOLD,
    trainer: str = TRAINER,
    plans: str = PLANS,
    use_npz: bool = USE_NPZ,
    continue_training: bool = False,
    pretrained_weights: str | None = None,
):
    import os
    import subprocess

    print(f"CUDA available: {os.system('nvidia-smi') == 0}")
    print(f"Training {dataset_id} | {configuration} | fold {fold}")

    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        configuration,
        str(fold),
        "-tr", trainer,
        "-p", plans,
        "-device", "cuda",
    ]

    if use_npz:
        cmd.append("--npz")
    if continue_training:
        cmd.append("--c")
    if pretrained_weights:
        cmd += ["-pretrained_weights", pretrained_weights]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Persist results
    results_vol.commit()
    print("Training finished. Results committed to volume 'nnunet-results'.")


def _fold_command(
    fold: int,
    trainer: str = TRAINER,
    pretrained_weights: str | None = None,
    continue_training: bool = False,
) -> list[str]:
    cmd = [
        "nnUNetv2_train", str(DATASET_ID), CONFIGURATION, str(fold),
        "-tr", trainer, "-p", PLANS, "-device", "cuda",
    ]
    if USE_NPZ:
        cmd.append("--npz")
    if continue_training:
        # `--c` resumes from checkpoint_latest.pth (full weights + optimizer +
        # epoch + LR schedule). Mutually exclusive with a fresh warm start --
        # the checkpoint already contains everything -pretrained_weights would.
        cmd.append("--c")
    elif pretrained_weights:
        cmd += ["-pretrained_weights", pretrained_weights]
    return cmd


@app.local_entrypoint()
def main(
    step: str = "train",                  # "upload" | "preprocess" | "train" | "train_all_folds" | "train_production" | "all"
    fold: str = str(FOLD),
    trainer: str = TRAINER,
    continue_training: bool = False,
):
    if step in ("upload", "all"):
        print("→ Uploading dataset to Modal volume...")
        upload_dataset()

    if step in ("preprocess", "all"):
        print("→ Running plan_and_preprocess...")
        plan_and_preprocess.remote(DATASET_ID)

    if step == "train":
        print(f"→ Starting training (fold={fold}, trainer={trainer})...")
        train.remote(
            dataset_id=DATASET_ID,
            configuration=CONFIGURATION,
            fold=fold,
            trainer=trainer,
            continue_training=continue_training,
        )
    elif step in ("train_all_folds", "all"):
        print(f"→ Submitting all {len(FOLDS)} folds as parallel Modal jobs (trainer={trainer})...")
        submit_commands([_fold_command(f, trainer, continue_training=continue_training) for f in FOLDS])
    elif step == "train_production":
        action = "Resuming" if continue_training else "Submitting"
        origin = "from checkpoint_latest.pth" if continue_training else f"warm-started from {WARMSTART_CHECKPOINT}"
        print(f"→ {action} {len(PRODUCTION_FOLDS)} production folds ({PRODUCTION_TRAINER}, {origin})...")
        submit_commands([
            _fold_command(
                f,
                PRODUCTION_TRAINER,
                pretrained_weights=WARMSTART_CHECKPOINT,
                continue_training=continue_training,
            )
            for f in PRODUCTION_FOLDS
        ])
