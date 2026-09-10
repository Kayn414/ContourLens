"""Per-structure HD95 for the GUI's Metrics table. gui/src/main.cpp queues
this after an inference run, or whenever ground truth and a prediction are
both loaded.

The GUI passes the (ground-truth id, prediction id) pairs it has already
matched (gui/src/structure_metrics.cpp, aliases included), so the pairing
logic lives in one place. Writes data/<case>/gui_export/metrics_hd95.json:
    {"pairs": {"<gt_id>:<pred_id>": <hd95 mm> | null}}
which the GUI only trusts while it is newer than both label volumes.

Usage: uv run python -m source.gui_metrics hd95 --case <case> --pairs 1:1,5:5
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from source.evaluate import hausdorff95
from source.gui_progress import configure_stdio, emit, emit_error

HD95_FILE_NAME = "metrics_hd95.json"


def load_gui_labels(base_path: Path) -> tuple[np.ndarray, tuple[float, float, float], list[str]]:
    """A GUI label volume (`<base>.bin` + `<base>.json`): (array (z, y, x), spacing (x, y, z), labels)."""
    meta = json.loads(base_path.with_name(base_path.name + ".json").read_text())
    array = np.fromfile(base_path.with_name(base_path.name + ".bin"), dtype=np.uint8).reshape(meta["shape"])
    sx, sy, sz = meta["spacing"]
    return array, (sx, sy, sz), list(meta.get("labels", []))


def crop_to_union(a: np.ndarray, b: np.ndarray, margin: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Both masks cropped to their union's bounding box plus `margin` voxels.
    HD95 only depends on the two surfaces, and full-volume distance maps (per
    structure, over e.g. a 1024x1024x194 CT) are what make it slow."""
    union = a | b
    if not union.any():
        return a, b
    region = []
    for axis in range(3):
        present = np.flatnonzero(union.any(axis=tuple(i for i in range(3) if i != axis)))
        region.append(slice(max(int(present[0]) - margin, 0), min(int(present[-1]) + margin + 1, union.shape[axis])))
    bounds = tuple(region)
    return a[bounds], b[bounds]


def parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for item in text.split(","):
        if item.strip():
            gt_id, pred_id = item.strip().split(":")
            pairs.append((int(gt_id), int(pred_id)))
    return pairs


def hd95_for_pairs(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: tuple[float, float, float],
    pairs: list[tuple[int, int]],
    names: list[str] | None = None,
) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for n, (gt_id, pred_id) in enumerate(pairs):
        label = names[gt_id - 1] if names and 0 < gt_id <= len(names) else f"{gt_id}:{pred_id}"
        emit("computing", n, len(pairs), f"HD95 {label} ({n + 1}/{len(pairs)})")
        gt_mask, pred_mask = crop_to_union(gt == gt_id, pred == pred_id)
        value = hausdorff95(pred_mask, gt_mask, spacing)
        results[f"{gt_id}:{pred_id}"] = None if np.isnan(value) else round(float(value), 2)
    return results


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    hd95 = sub.add_parser("hd95")
    hd95.add_argument("--case", required=True)
    hd95.add_argument("--pairs", required=True, help="Comma-separated <gt_id>:<pred_id> pairs")
    args = parser.parse_args()

    try:
        export_dir = Path("data") / args.case / "gui_export"
        gt, spacing, names = load_gui_labels(export_dir / "masks")
        pred, _, _ = load_gui_labels(export_dir / "prediction")
        if gt.shape != pred.shape:
            raise ValueError(f"Ground truth {gt.shape} and prediction {pred.shape} aren't on the same grid")

        results = hd95_for_pairs(gt, pred, spacing, parse_pairs(args.pairs), names)
        (export_dir / HD95_FILE_NAME).write_text(json.dumps({"pairs": results}, indent=2))
        scored = [v for v in results.values() if v is not None]
        emit("done", fraction=1.0, message=f"HD95 for {len(scored)} of {len(results)} structures")
    except Exception as e:
        traceback.print_exc()
        emit_error(f"{type(e).__name__}: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
