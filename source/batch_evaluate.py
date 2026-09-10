"""Batch evaluation over one nnU-Net fold's validation cases (the GUI's
View > Batch Evaluation).

The model and backend come from a case's inference config, but only fold K's
weights run, on exactly the cases fold K held out (nnU-Net's
splits_final.json), so every score is on data that model never trained on.
CTs are read from nnUNet_raw's imagesTr and ground truth from labelsTr, with
structure names from the dataset's dataset.json. Modal predictions run up to
--parallel cases at once; local predictions run one case per available GPU
(at most --parallel), each pinned to its own GPU.

Writes to --out: predictions/<case>.nii.gz, results.json (per case and
structure, plus per-structure mean/std and per-case mean Dice) and results.csv.

Usage: uv run python -m source.batch_evaluate --case <case> --fold K --out <dir> [--parallel N] [--limit M]
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import shutil
import tempfile
import traceback
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from source.evaluate import dice, hausdorff95
from source.export_gui_volume import GuiGrid
from source.gui_metrics import crop_to_union
from source.gui_progress import configure_stdio, emit, emit_error, emit_notice
from source.inference_config import InferenceConfig, load_config
from source.structure_names import load_aliases, match_key


def find_dataset_dirs(dataset_id: str) -> tuple[Path, Path]:
    """(nnUNet_raw/Dataset<id>_*, nnUNet_preprocessed/Dataset<id>_*)."""
    from source.data.nnunet_dataset import (
        resolve_nnunet_preprocessed,
        resolve_nnunet_raw,
    )

    pattern = f"Dataset{int(dataset_id):03d}_*"
    dirs = []
    for root in (resolve_nnunet_raw(), resolve_nnunet_preprocessed()):
        matches = sorted(root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No {pattern} folder under {root}")
        dirs.append(matches[0])
    return dirs[0], dirs[1]


def fold_validation_cases(preprocessed_dir: Path, fold: int) -> list[str]:
    splits = json.loads((preprocessed_dir / "splits_final.json").read_text())
    if not 0 <= fold < len(splits):
        raise ValueError(f"Fold {fold} is out of range: splits_final.json has {len(splits)} folds")
    return list(splits[fold]["val"])


def score_case(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: tuple[float, float, float],
    gt_names: dict[int, str],
    pred_names: dict[int, str],
    aliases: dict[str, str] | None = None,
) -> list[dict]:
    """Dice/HD95 per ground-truth structure, pairing ids by structure match key
    (through `aliases`, defaulting to configs/structure_aliases.json)."""
    aliases = load_aliases() if aliases is None else aliases
    pred_id_by_key = {match_key(name, aliases): label_id for label_id, name in pred_names.items()}
    rows = []
    for gt_id, name in sorted(gt_names.items()):
        gt_mask = gt == gt_id
        pred_id = pred_id_by_key.get(match_key(name, aliases))
        if pred_id is None:
            rows.append({"structure": name, "dice": None, "hd95_mm": None, "gt_voxels": int(gt_mask.sum()), "pred_voxels": None})
            continue
        pred_mask = pred == pred_id
        d = dice(pred_mask, gt_mask)
        gt_crop, pred_crop = crop_to_union(gt_mask, pred_mask)
        h = hausdorff95(pred_crop, gt_crop, spacing)
        rows.append({
            "structure": name,
            "dice": None if np.isnan(d) else round(float(d), 4),
            "hd95_mm": None if np.isnan(h) else round(float(h), 2),
            "gt_voxels": int(gt_mask.sum()),
            "pred_voxels": int(pred_mask.sum()),
        })
    return rows


def summarize(cases: dict[str, dict]) -> dict:
    def stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4), "n": len(values)}

    per_structure: dict[str, dict[str, list[float]]] = {}
    per_case: dict[str, dict] = {}
    for case_name, result in cases.items():
        rows = result.get("structures")
        if not rows:
            continue
        dices = [row["dice"] for row in rows if row["dice"] is not None]
        per_case[case_name] = {"mean_dice": round(float(np.mean(dices)), 4) if dices else None}
        for row in rows:
            values = per_structure.setdefault(row["structure"], {"dice": [], "hd95_mm": []})
            for key in ("dice", "hd95_mm"):
                if row[key] is not None:
                    values[key].append(row[key])
    return {
        "structures": {name: {key: stats(vals) for key, vals in values.items()} for name, values in per_structure.items()},
        "cases": per_case,
    }


def local_gpu_count() -> int:
    """CUDA GPUs local nnU-Net can use (0 without a usable torch/CUDA)."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001 -- no usable torch/CUDA just means no local GPUs
        return 0


def local_worker_devices(parallel: int, gpu_count: int) -> list[int | None]:
    """One entry per concurrent local prediction: the GPU index to pin it to,
    capped at the GPUs actually present; [None] (a single CPU run) without any."""
    if gpu_count <= 0:
        return [None]
    return list(range(max(1, min(parallel, gpu_count))))


def predict_cases(
    config: InferenceConfig, image_paths: dict[str, Path], predictions_dir: Path, parallel: int
) -> Iterator[tuple[str, Path | None, dict[str, int] | None, str | None]]:
    """(case, prediction path, {name: id}, error) per case, as each finishes.
    Modal runs up to `parallel` cases at once; local runs one case per GPU
    (at most `parallel`), each pinned to its own device."""
    from source.inference_backends import (
        _stage_as_nifti,
        ensure_modal_deployed,
        predict_modal_bytes,
        run_local,
    )

    if config.backend == "modal":
        fn = ensure_modal_deployed(config.modal)
        workers = max(1, parallel)

        def predict_one(case: str) -> tuple[Path, dict[str, int]]:
            with tempfile.TemporaryDirectory() as tmp:
                nifti_bytes = _stage_as_nifti(image_paths[case], Path(tmp), case).read_bytes()
            result_bytes, label_ids = predict_modal_bytes(fn, nifti_bytes, config.model)
            path = predictions_dir / f"{case}.nii.gz"
            path.write_bytes(result_bytes)
            return path, label_ids
    else:
        gpu_count = local_gpu_count()
        devices = local_worker_devices(parallel, gpu_count)
        if len(devices) < parallel:
            emit_notice(f"Local backend: {gpu_count} GPU(s) available, so {len(devices)} case(s) at a time instead of {parallel}.")
        free_devices: queue.Queue[int | None] = queue.Queue()
        for device in devices:
            free_devices.put(device)
        workers = len(devices)

        def predict_one(case: str) -> tuple[Path, dict[str, int]]:
            device = free_devices.get()  # a GPU no other running case is using
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    prediction, label_ids = run_local(image_paths[case], Path(tmp), config.model, case_id=case, device=device)
                    path = predictions_dir / f"{case}.nii.gz"
                    shutil.copyfile(prediction, path)
                return path, label_ids
            finally:
                free_devices.put(device)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(predict_one, case): case for case in image_paths}
        for future in as_completed(futures):
            case = futures[future]
            try:
                path, label_ids = future.result()
                yield case, path, label_ids, None
            except Exception as e:  # noqa: BLE001 -- one failed case shouldn't discard the rest of the batch
                yield case, None, None, f"{type(e).__name__}: {e}"


def run_batch(case: str, fold: int, out_dir: Path, parallel: int = 2, limit: int = 0) -> dict:
    config = load_config(case)
    config = replace(config, model=replace(config.model, folds=[fold]))

    raw_dir, preprocessed_dir = find_dataset_dirs(config.model.dataset_id)
    dataset = json.loads((raw_dir / "dataset.json").read_text())
    file_ending = dataset.get("file_ending", ".nii.gz")
    gt_names = {label_id: name for name, label_id in dataset["labels"].items() if isinstance(label_id, int) and label_id > 0}

    cases = fold_validation_cases(preprocessed_dir, fold)
    if limit > 0:
        cases = cases[:limit]
    image_paths = {c: raw_dir / "imagesTr" / f"{c}_0000{file_ending}" for c in cases}
    emit_notice(f"Fold {fold} validation set: {len(cases)} cases, predicted with fold {fold}'s weights only.")

    predictions_dir = out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"case": case, "fold": fold, "backend": config.backend, "model": asdict(config.model), "cases": {}}

    emit("predicting", 0, len(cases), f"0/{len(cases)} cases done")
    for done, (case_name, prediction_path, label_ids, error) in enumerate(predict_cases(config, image_paths, predictions_dir, parallel), 1):
        if error is not None or prediction_path is None or label_ids is None:
            print(f"{case_name}: FAILED {error}")
            results["cases"][case_name] = {"error": error}
        else:
            gt_image = sitk.ReadImage(str(raw_dir / "labelsTr" / f"{case_name}{file_ending}"))
            pred_image = GuiGrid.of(gt_image).resample(sitk.ReadImage(str(prediction_path)), sitk.sitkNearestNeighbor)
            rows = score_case(
                sitk.GetArrayFromImage(gt_image), sitk.GetArrayFromImage(pred_image), gt_image.GetSpacing(),
                gt_names, {label_id: name for name, label_id in label_ids.items()},
            )
            results["cases"][case_name] = {"structures": rows}
            print(f"{case_name}: " + ", ".join(f"{r['structure']}={r['dice']}" for r in rows))
        emit("predicting", done, len(cases), f"{case_name} done ({done}/{len(cases)})")

    results["summary"] = summarize(results["cases"])
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "structure", "dice", "hd95_mm", "gt_voxels", "pred_voxels", "error"])
        for case_name, result in results["cases"].items():
            if "error" in result:
                writer.writerow([case_name, "", "", "", "", "", result["error"]])
            for row in result.get("structures", []):
                writer.writerow([case_name, row["structure"], row["dice"], row["hd95_mm"], row["gt_voxels"], row["pred_voxels"], ""])
    return results


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, help="Case whose inference config (model/backend) to use")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=2, help="Concurrent predictions: Modal cloud GPUs, or local GPUs (capped at the GPUs present)")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N validation cases (smoke testing)")
    args = parser.parse_args()

    try:
        results = run_batch(args.case, args.fold, args.out, args.parallel, args.limit)
        failed = sum(1 for r in results["cases"].values() if "error" in r)
        emit("done", fraction=1.0, message=f"{len(results['cases']) - failed} cases scored, {failed} failed -> {args.out}")
    except Exception as e:
        traceback.print_exc()
        emit_error(f"{type(e).__name__}: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
