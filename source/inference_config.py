"""Per-case inference configuration: which trained model to run, and where
(locally, or on a cloud GPU backend). Loaded by source/run_inference_for_gui.py
(the GUI's "Run Inference" button invokes it with the loaded case's name).

Lives under configs/<case_name>/inference_config.json -- NOT under data/
(which is entirely gitignored, being large regenerable/external fixtures;
these small configs are meant to be tracked and shared). <case_name> is the
same relative name used as the GUI's argv[1] and under data/, e.g. "phantom"
or "hanseg_cases/case_03" -- see configs/phantom/inference_config.json and
configs/hanseg_cases/case_03/inference_config.json for real examples. Schema:

    {
      "backend": "local" | "modal",
      "source_ct": "path/to/a/nifti/or/nrrd/file/or/a/dicom/series/dir",
      "model": {
        "dataset_id": "501",
        "configuration": "3d_fullres",
        "trainer": "nnUNetTrainer_500epochs",
        "plans": "nnUNetPlans",
        "folds": [0, 1, 2, 3],
        "results_dir": null   // override nnUNet_results; null = use the env var/default
      },
      "modal": {              // only read when backend == "modal"
        "app_name": "nnunet-training",
        "function_name": "predict_nifti_bytes"
      }
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    dataset_id: str = "501"
    configuration: str = "3d_fullres"
    trainer: str = "nnUNetTrainer_500epochs"
    plans: str = "nnUNetPlans"
    folds: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    results_dir: str | None = None


@dataclass
class ModalBackendConfig:
    app_name: str = "nnunet-training"
    function_name: str = "predict_nifti_bytes"


@dataclass
class InferenceConfig:
    backend: str = "local"  # "local" | "modal"
    source_ct: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    modal: ModalBackendConfig = field(default_factory=ModalBackendConfig)


CONFIGS_ROOT = Path("configs")


def config_path(case_name: str) -> Path:
    return CONFIGS_ROOT / case_name / "inference_config.json"


def load_config(case_name: str) -> InferenceConfig:
    path = config_path(case_name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Write one (see configs/phantom/inference_config.json "
            "for a real example) before using Run Inference on this case."
        )
    raw = json.loads(path.read_text())
    return InferenceConfig(
        backend=raw.get("backend", "local"),
        source_ct=raw["source_ct"],
        model=ModelConfig(**raw.get("model", {})),
        modal=ModalBackendConfig(**raw.get("modal", {})),
    )


def write_config(case_name: str, config: InferenceConfig) -> Path:
    path = config_path(case_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2))
    return path


def resolve_label_ids(model: ModelConfig) -> dict[str, int]:
    """Reads {structure_name: label_id} straight from the trained model's own
    dataset.json (nnU-Net writes one alongside every trained model, at
    <results_dir>/Dataset<id>_<Name>/<trainer>__<plans>__<configuration>/dataset.json)
    """
    from source.data.nnunet_dataset import resolve_nnunet_results

    results_dir = Path(model.results_dir) if model.results_dir else resolve_nnunet_results()
    dataset_id = int(model.dataset_id)
    matches = sorted(results_dir.glob(f"Dataset{dataset_id:03d}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No Dataset{dataset_id:03d}_* folder under {results_dir} -- check "
            "inference_config.json's model.dataset_id/results_dir."
        )

    dataset_json = matches[0] / f"{model.trainer}__{model.plans}__{model.configuration}" / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(
            f"{dataset_json} not found -- check model.trainer/plans/configuration "
            "in inference_config.json."
        )

    labels = json.loads(dataset_json.read_text())["labels"]
    return {name: label_id for name, label_id in labels.items() if name != "background"}
