import numpy as np

from source.evaluate import hausdorff95
from source.gui_metrics import crop_to_union, hd95_for_pairs, parse_pairs


def test_cropping_gives_the_same_hd95_as_the_full_volume():
    gt = np.zeros((20, 40, 40), bool)
    gt[5:12, 10:20, 10:22] = True
    pred = np.zeros_like(gt)
    pred[6:13, 12:21, 9:20] = True
    spacing = (0.8, 0.8, 2.5)

    gt_crop, pred_crop = crop_to_union(gt, pred)

    assert gt_crop.size < gt.size
    assert np.isclose(hausdorff95(pred_crop, gt_crop, spacing), hausdorff95(pred, gt, spacing))


def test_hd95_is_null_when_a_side_is_empty():
    gt = np.zeros((10, 10, 10), np.uint8)
    gt[2:5, 2:5, 2:5] = 1
    pred = np.zeros_like(gt)
    assert hd95_for_pairs(gt, pred, (1.0, 1.0, 1.0), [(1, 1)]) == {"1:1": None}


def test_parse_pairs():
    assert parse_pairs("1:1, 5:7") == [(1, 1), (5, 7)]
