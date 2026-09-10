import json
from pathlib import Path

import numpy as np
import pytest

from source.batch_evaluate import (
    fold_validation_cases,
    local_worker_devices,
    score_case,
    summarize,
)


def test_local_workers_are_capped_at_available_gpus():
    assert local_worker_devices(parallel=2, gpu_count=1) == [0]
    assert local_worker_devices(parallel=3, gpu_count=4) == [0, 1, 2]
    assert local_worker_devices(parallel=2, gpu_count=0) == [None]  # no GPU: one CPU run


def test_fold_validation_cases(tmp_path: Path):
    splits = [{"train": ["a", "b"], "val": ["c"]}, {"train": ["a", "c"], "val": ["b"]}]
    (tmp_path / "splits_final.json").write_text(json.dumps(splits))

    assert fold_validation_cases(tmp_path, 1) == ["b"]
    with pytest.raises(ValueError, match="out of range"):
        fold_validation_cases(tmp_path, 2)


def test_score_case_pairs_through_aliases_and_marks_unpredicted_structures():
    gt = np.zeros((10, 10, 10), np.uint8)
    gt[1:4, 1:4, 1:4] = 1
    gt[6:9, 6:9, 6:9] = 2
    pred = np.zeros_like(gt)
    pred[1:4, 1:4, 1:4] = 5

    aliases = {"opticnrvl": "opticnervel"}  # its own table: configs/structure_aliases.json is user-editable
    rows = {
        r["structure"]: r
        for r in score_case(gt, pred, (1.0, 1.0, 1.0), {1: "OpticNrv_L", 2: "Parotid_L"}, {5: "optic_nerve_l"}, aliases)
    }

    assert rows["OpticNrv_L"]["dice"] == 1.0 and rows["OpticNrv_L"]["hd95_mm"] == 0.0
    assert rows["Parotid_L"]["dice"] is None and rows["Parotid_L"]["pred_voxels"] is None


def test_summarize_skips_failed_cases_and_missing_values():
    cases = {
        "a": {"structures": [{"structure": "x", "dice": 0.8, "hd95_mm": 2.0}]},
        "b": {"structures": [{"structure": "x", "dice": 0.6, "hd95_mm": None}]},
        "c": {"error": "boom"},
    }
    summary = summarize(cases)

    assert summary["structures"]["x"]["dice"] == {"mean": 0.7, "std": 0.1, "n": 2}
    assert summary["structures"]["x"]["hd95_mm"]["n"] == 1
    assert summary["cases"] == {"a": {"mean_dice": 0.8}, "b": {"mean_dice": 0.6}}
