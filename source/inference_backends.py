"""Inference backends: given a source CT (any format source/export_gui_volume.py's
load_volume_image() accepts -- NIfTI, NRRD, or a DICOM series directory) and
a ModelConfig, produce a multi-label prediction .nii.gz.

- "local": runs nnUNetv2_predict on this machine.
- "modal": calls the deployed Modal function named in ModalBackendConfig
  (source/modal_nnunet.py's predict_nifti_bytes by default), which needs no
  pre-staged volume data -- only the trained model already present on the
  results volume. Ample remote RAM sidesteps the local-memory ceiling that
  motivated adding this (see source/modal_nnunet.py's predict_case docstring).
  Requires `modal deploy source/modal_nnunet.py` once (Function.from_name()
  only finds deployed apps, not ephemeral `modal run` ones).

Both return the local path to the downloaded/written prediction .nii.gz.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import SimpleITK as sitk

from source.export_gui_volume import load_volume_image
from source.inference_config import InferenceConfig, ModalBackendConfig, ModelConfig


def _stage_as_nifti(source_ct: Path, dest_dir: Path, case_id: str) -> Path:
    """Re-encodes `source_ct` (whatever format) as a canonical NIfTI file
    named the way nnU-Net expects its single-channel input, so both backends
    hand it a clean, uniform file regardless of the original source format.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged = dest_dir / f"{case_id}_0000.nii.gz"
    sitk.WriteImage(load_volume_image(source_ct), str(staged))
    return staged


def run_local(source_ct: Path, output_dir: Path, model: ModelConfig, case_id: str = "case") -> Path:
    from source.data.nnunet_dataset import nnunet_env

    input_dir = output_dir / "_nnunet_input"
    staged = _stage_as_nifti(source_ct, input_dir, case_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **nnunet_env()}
    if model.results_dir:
        env["nnUNet_results"] = model.results_dir

    cmd = [
        "nnUNetv2_predict",
        "-i", str(staged.parent), "-o", str(output_dir),
        "-d", str(model.dataset_id), "-c", model.configuration,
        "-f", *[str(f) for f in model.folds],
        "-tr", model.trainer, "-p", model.plans,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    return output_dir / f"{case_id}.nii.gz"


def run_modal(source_ct: Path, output_dir: Path, model: ModelConfig, modal_cfg: ModalBackendConfig, case_id: str = "case") -> Path:
    import tempfile

    import modal

    with tempfile.TemporaryDirectory() as tmp:
        staged = _stage_as_nifti(source_ct, Path(tmp), case_id)
        nifti_bytes = staged.read_bytes()

    fn = modal.Function.from_name(modal_cfg.app_name, modal_cfg.function_name)
    print(f"Calling Modal function {modal_cfg.app_name}::{modal_cfg.function_name} ({len(nifti_bytes):,} bytes)...")
    result_bytes = fn.remote(
        nifti_bytes,
        dataset_id=str(model.dataset_id),
        configuration=model.configuration,
        folds=",".join(str(f) for f in model.folds),
        trainer=model.trainer,
        plans=model.plans,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{case_id}.nii.gz"
    out_path.write_bytes(result_bytes)
    return out_path


def run(config: InferenceConfig, output_dir: Path, case_id: str = "case") -> Path:
    source_ct = Path(config.source_ct)
    if not source_ct.exists():
        raise FileNotFoundError(f"inference_config.json's source_ct does not exist: {source_ct}")

    if config.backend == "local":
        return run_local(source_ct, output_dir, config.model, case_id)
    if config.backend == "modal":
        return run_modal(source_ct, output_dir, config.model, config.modal, case_id)
    raise ValueError(f"Unknown backend {config.backend!r} (expected 'local' or 'modal')")
