"""Score a model's phantom prediction against the ground-truth OAR masks
pulled from the real Eclipse RTSTRUCT (source/data/phantom.py).

Reports per-structure Dice and 95th-percentile Hausdorff distance (HD95).
Structures with no ground truth in the phantom (spinal_cord — see
phantom.py's docstring) or no prediction are reported as unscored rather
than silently given a misleading 0.
"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from source.data.hanseg import LABEL_IDS
from source.data.phantom import CASE_ID, OUT_DIR as PHANTOM_MASKS_DIR
from source.predict import PREDICTION_DIR


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")  # both empty -- undefined, not a perfect score
    return 2.0 * intersection / denom


def hausdorff95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...]) -> float:
    """HD95 via SimpleITK distance maps: 95th percentile of surface-to-surface
    distances, in both directions (pred->gt and gt->pred), symmetric max.
    """
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")

    def _surface_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_img = sitk.GetImageFromArray(a.astype(np.uint8))
        a_img.SetSpacing(spacing)
        b_img = sitk.GetImageFromArray(b.astype(np.uint8))
        b_img.SetSpacing(spacing)

        a_surface = sitk.GetArrayFromImage(sitk.LabelContour(a_img)) > 0
        b_dist_map = sitk.GetArrayFromImage(
            sitk.Abs(sitk.SignedMaurerDistanceMap(b_img, squaredDistance=False, useImageSpacing=True))
        )
        return b_dist_map[a_surface]

    d_pred_to_gt = _surface_distances(pred, gt)
    d_gt_to_pred = _surface_distances(gt, pred)
    return max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))


def evaluate(
    prediction_dir: Path = PREDICTION_DIR,
    gt_dir: Path = PHANTOM_MASKS_DIR,
    case_id: str = CASE_ID,
) -> dict[str, dict[str, float]]:
    pred_path = prediction_dir / f"{case_id}.nii.gz"
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} not found — run source/predict.py first")

    pred_image = sitk.ReadImage(str(pred_path))
    pred_array = sitk.GetArrayFromImage(pred_image)
    spacing = pred_image.GetSpacing()  # (x, y, z) -- what SimpleITK Image.SetSpacing expects

    results = {}
    for canonical_name, label_id in LABEL_IDS.items():
        gt_path = gt_dir / f"{canonical_name}.nii.gz"
        pred_mask = pred_array == label_id

        if not gt_path.exists():
            results[canonical_name] = {"dice": None, "hd95_mm": None, "note": "no ground truth in phantom"}
            continue

        gt_array = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))) > 0
        if gt_array.sum() == 0:
            results[canonical_name] = {"dice": None, "hd95_mm": None, "note": "ground truth mask is empty"}
            continue

        d = dice(pred_mask, gt_array)
        h = hausdorff95(pred_mask, gt_array, spacing)
        results[canonical_name] = {
            "dice": None if np.isnan(d) else round(float(d), 4),
            "hd95_mm": None if np.isnan(h) else round(float(h), 2),
            "pred_voxels": int(pred_mask.sum()),
            "gt_voxels": int(gt_array.sum()),
        }

    return results


def main() -> None:
    results = evaluate()
    print(json.dumps(results, indent=2))

    scored = [r["dice"] for r in results.values() if r.get("dice") is not None]
    if scored:
        print(f"\nMean Dice over {len(scored)} scored structures: {np.mean(scored):.4f}")
    else:
        print("\nNo structures were scorable (no valid ground truth / predictions).")


if __name__ == "__main__":
    main()
