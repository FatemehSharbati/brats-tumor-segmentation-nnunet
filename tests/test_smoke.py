"""Smoke tests: BraTS regions, converter, evaluation (numpy/nibabel only)."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from src.evaluate import brats_scores, region_mask, dice_score


def _seg():
    seg = np.zeros((20, 20, 20), dtype=np.int16)
    seg[4:8, 4:8, 4:8] = 1
    seg[8:12, 8:12, 8:12] = 2
    seg[12:16, 12:16, 12:16] = 4
    return seg


def test_region_masks():
    seg = _seg()
    assert region_mask(seg, "WT").sum() == 3 * 4 ** 3
    assert region_mask(seg, "TC").sum() == 2 * 4 ** 3
    assert region_mask(seg, "ET").sum() == 4 ** 3


def test_perfect_scores():
    seg = _seg()
    scores = brats_scores(seg, seg)
    assert scores == {"WT": 1.0, "TC": 1.0, "ET": 1.0}, scores


def test_partial_scores():
    seg = _seg()
    pred = np.zeros_like(seg)
    pred[seg == 4] = 4  # only enhancing right
    scores = brats_scores(pred, seg)
    assert scores["ET"] == 1.0
    assert scores["WT"] < 1.0
    assert scores["TC"] < 1.0


def test_dice_score_edge_cases():
    empty = np.zeros((6, 6, 6), dtype=np.int16)
    full = np.ones((6, 6, 6), dtype=np.int16)
    # empty vs empty is a perfect match, not a division-by-zero
    assert dice_score(empty, empty) == 1.0
    # no overlap at all
    assert dice_score(empty, full) == 0.0
    assert dice_score(full, empty) == 0.0
    # empty segmentations score perfectly for every region
    scores = brats_scores(empty, empty)
    assert scores == {"WT": 1.0, "TC": 1.0, "ET": 1.0}, scores


def test_region_mask_unknown_region():
    try:
        region_mask(np.zeros((4, 4, 4), dtype=np.int16), "XX")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown region")


def test_converter(tmp_path):
    import nibabel as nib
    root = tmp_path / "BraTS"
    case = root / "BraTS2021_00000"
    case.mkdir(parents=True)
    for m in ["t1", "t1ce", "t2", "flair", "seg"]:
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4)),
                 str(case / f"BraTS2021_00000_{m}.nii.gz"))
    from src.brats_to_nnunet import convert
    n = convert(str(root), str(tmp_path / "nnraw"))
    assert n == 1
    dj = json.loads((tmp_path / "nnraw" / "dataset.json").read_text())
    assert dj["numTraining"] == 1
    assert len(list((tmp_path / "nnraw" / "imagesTr").glob("*.nii.gz"))) == 4
    assert len(list((tmp_path / "nnraw" / "labelsTr").glob("*.nii.gz"))) == 1
