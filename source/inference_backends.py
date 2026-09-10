"""Inference backends: given a source CT (any format source/export_gui_volume.py's
load_volume_image() accepts -- NIfTI, NRRD, or a DICOM series directory) and
a ModelConfig, produce a multi-label prediction .nii.gz.

- "local": runs nnUNetv2_predict on this machine.
- "modal": calls the deployed Modal function named in ModalBackendConfig
  (source/modal_nnunet.py's predict_nifti_bytes by default), which needs no
  pre-staged volume data -- only the trained model already present on the
  results volume. Ample remote RAM sidesteps the local-memory ceiling that
  motivated adding this (see source/modal_nnunet.py's predict_case docstring).
  Deploys source/modal_nnunet.py automatically on first use if the app isn't
  live yet (ensure_modal_deployed): Function.from_name() only finds deployed
  apps, not ephemeral `modal run` ones.

Both return (the local path to the downloaded/written prediction .nii.gz,
{structure_name: label_id}) -- the label map comes from whatever model the
config actually points at (source.inference_config.resolve_label_ids for
"local"; read server-side and returned alongside the prediction bytes for
"modal", since the results volume isn't visible from the calling machine).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import SimpleITK as sitk

from source.export_gui_volume import load_volume_image
from source.gui_progress import (
    TqdmProgressParser,
    emit,
    emit_notice,
    run_forwarding_output,
    run_with_progress,
)
from source.inference_config import (
    InferenceConfig,
    ModalBackendConfig,
    ModelConfig,
    resolve_label_ids,
)


def _stage_as_nifti(source_ct: Path, dest_dir: Path, case_id: str) -> Path:
    """Re-encodes `source_ct` (whatever format) as a canonical NIfTI file
    named the way nnU-Net expects its single-channel input, so both backends
    hand it a clean, uniform file regardless of the original source format.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged = dest_dir / f"{case_id}_0000.nii.gz"
    sitk.WriteImage(load_volume_image(source_ct), str(staged))
    return staged


def run_local(
    source_ct: Path, output_dir: Path, model: ModelConfig, case_id: str = "case", device: int | None = None
) -> tuple[Path, dict[str, int]]:
    """`device`: the CUDA GPU index to pin this prediction to (source/batch_evaluate.py
    runs one case per GPU); None leaves nnU-Net's default device."""
    from source.data.nnunet_dataset import nnunet_env

    input_dir = output_dir / "_nnunet_input"
    emit("staging", message=f"Converting {source_ct.name} to NIfTI for nnU-Net")
    staged = _stage_as_nifti(source_ct, input_dir, case_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **nnunet_env()}
    if model.results_dir:
        env["nnUNet_results"] = model.results_dir
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)

    cmd = [
        "nnUNetv2_predict",
        "-i", str(staged.parent), "-o", str(output_dir),
        "-d", str(model.dataset_id), "-c", model.configuration,
        "-f", *[str(f) for f in model.folds],
        "-tr", model.trainer, "-p", model.plans,
    ]
    print("Running:", " ".join(cmd))
    emit("predicting", message="Loading model and preprocessing (nnU-Net)")
    run_with_progress(cmd, TqdmProgressParser(n_passes=len(model.folds)), env=env)
    return output_dir / f"{case_id}.nii.gz", resolve_label_ids(model)


# What `modal deploy` is pointed at to create the app ModalBackendConfig names by default.
MODAL_APP_MODULE = "source/modal_nnunet.py"


def ensure_modal_deployed(modal_cfg: ModalBackendConfig):
    """The deployed Modal function `modal_cfg` names, running `modal deploy
    MODAL_APP_MODULE` first if that app isn't live yet -- Function.from_name()
    only finds deployed apps, and the app a `modal run` training launch
    creates stops when that run ends. The GUI shows @@notice lines about it,
    so the user knows why this run took longer and that the app now exists.
    """
    import modal
    from modal.exception import NotFoundError

    def lookup():
        fn = modal.Function.from_name(modal_cfg.app_name, modal_cfg.function_name)
        fn.hydrate()
        return fn

    try:
        return lookup()
    except NotFoundError:
        pass

    emit("deploying", message=f"Deploying {MODAL_APP_MODULE} to Modal (one-time)")
    emit_notice(f"Modal app '{modal_cfg.app_name}' wasn't deployed; deploying {MODAL_APP_MODULE} now (one-time).")
    run_forwarding_output([sys.executable, "-m", "modal", "deploy", MODAL_APP_MODULE])
    try:
        fn = lookup()
    except NotFoundError as e:
        raise RuntimeError(
            f"Deployed {MODAL_APP_MODULE}, but {modal_cfg.app_name}::{modal_cfg.function_name} still isn't found. "
            "The inference config's Modal app/function names must match modal.App(...) in source/modal_utils.py "
            f"and a function defined in {MODAL_APP_MODULE}."
        ) from e
    emit_notice(f"Deployed Modal app '{modal_cfg.app_name}'; later runs will use it directly.")
    return fn


def predict_modal_bytes(fn, nifti_bytes: bytes, model: ModelConfig) -> tuple[bytes, dict[str, int]]:
    """One remote prediction: (prediction NIfTI bytes, {structure_name: label_id})."""
    return fn.remote(
        nifti_bytes,
        dataset_id=str(model.dataset_id),
        configuration=model.configuration,
        folds=",".join(str(f) for f in model.folds),
        trainer=model.trainer,
        plans=model.plans,
    )


def run_modal(source_ct: Path, output_dir: Path, model: ModelConfig, modal_cfg: ModalBackendConfig, case_id: str = "case") -> tuple[Path, dict[str, int]]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        emit("staging", message=f"Converting {source_ct.name} to NIfTI")
        staged = _stage_as_nifti(source_ct, Path(tmp), case_id)
        nifti_bytes = staged.read_bytes()

    fn = ensure_modal_deployed(modal_cfg)
    print(f"Calling Modal function {modal_cfg.app_name}::{modal_cfg.function_name} ({len(nifti_bytes):,} bytes)...")
    # A deployed function's remote tqdm output isn't streamed back to this
    # caller, so cloud runs only get this indeterminate stage (no per-patch progress).
    emit("remote", message=f"Uploading and running on Modal ({modal_cfg.app_name}::{modal_cfg.function_name})")
    result_bytes, label_ids = predict_modal_bytes(fn, nifti_bytes, model)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{case_id}.nii.gz"
    out_path.write_bytes(result_bytes)
    return out_path, label_ids


def run(config: InferenceConfig, output_dir: Path, case_id: str = "case") -> tuple[Path, dict[str, int]]:
    source_ct = Path(config.source_ct)
    if not source_ct.exists():
        raise FileNotFoundError(f"inference_config.json's source_ct does not exist: {source_ct}")

    if config.backend == "local":
        return run_local(source_ct, output_dir, config.model, case_id)
    if config.backend == "modal":
        return run_modal(source_ct, output_dir, config.model, config.modal, case_id)
    raise ValueError(f"Unknown backend {config.backend!r} (expected 'local' or 'modal')")
