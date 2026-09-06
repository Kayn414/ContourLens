"""Run nnU-Net preprocessing/training for the HanSeg dataset on Modal.

Data flow: source/data/hanseg.py + source/data/nnunet_dataset.py build the
nnU-Net raw dataset locally at data/nnUNet_raw/Dataset501_HanSeg. `upload`
pushes that onto the Modal raw volume before `preprocess`/`train` can see it.
"""

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

# ============ Production run config ==============
# 4 folds (0-3, fold 4 skipped for budget), 500 epochs each, warm-started
# from the fold-0 100-epoch smoke-test checkpoint (see conversation: smoke
# test confirmed ~69.8s/epoch, checkpoint verified present on the results
# volume). Kick off DETACHED so a local disconnect doesn't stop it:
# `modal run --detach source/modal_nnunet.py::main --step train_production`
# (add `--continue-training` to resume folds from checkpoint_latest.pth).
PRODUCTION_TRAINER = "nnUNetTrainer_500epochs"
PRODUCTION_FOLDS = (0, 1, 2, 3)
WARMSTART_CHECKPOINT = (
    f"{VOL_RESULTS}/{DATASET_FOLDER_NAME}/nnUNetTrainer_100epochs__{PLANS}__{CONFIGURATION}/fold_0/checkpoint_final.pth"
)


def upload_dataset(local_dir: Path | None = None) -> None:
    """Push the locally-built nnU-Net raw dataset onto the Modal raw volume.

    Shells out to `modal volume put` rather than the Volume.batch_upload()
    Python API -- batch_upload() was observed to hang indefinitely (no
    progress, no error, no network activity) on a ~3.7GB/84-file upload,
    while the CLI completed the same upload reliably with visible progress.
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


# ============================================================
# 2b. Predict (single case) -- for checking model accuracy on a HaN-Seg case
# ============================================================
# nnU-Net's segmentation-export step holds full-resolution per-class
# probability maps in RAM; a high-res HaN-Seg CT can exceed what's free on a
# dev machine already running an IDE/browser/etc (observed: repeatedly
# killed locally by an OOM guard even with -npp 1 -nps 1). Ample memory here
# sidesteps that rather than fighting it locally.
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
        # 4 folds x nnUNetTrainer_500epochs, warm-started from the fold-0
        # smoke-test checkpoint. Decided cost: ~$71 (training + CV validation).
        #
        # Launch DETACHED so a local-client disconnect (e.g. PC reboot) does not
        # kill the run:
        #   modal run --detach source/modal_nnunet.py::main --step train_production
        # To resume after an interruption (picks up each fold from its
        # checkpoint_latest.pth, saved every 50 epochs):
        #   modal run --detach source/modal_nnunet.py::main --step train_production --continue-training
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
