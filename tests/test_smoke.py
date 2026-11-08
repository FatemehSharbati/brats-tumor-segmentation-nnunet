"""Smoke tests: BraTS regions, converter, evaluation (numpy/nibabel only)."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.evaluate import (
    brats_scores, region_mask, dice_score, evaluate_files, print_report,
)


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


def test_evaluate_files_mismatched_lengths(tmp_path):
    """A count mismatch must fail loudly, not silently drop unpaired cases."""
    pred = tmp_path / "p.nii.gz"
    tgt = tmp_path / "t.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.int16), np.eye(4)), str(pred))
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.int16), np.eye(4)), str(tgt))
    with pytest.raises(ValueError, match="count mismatch"):
        evaluate_files([pred, pred], [tgt])


def test_evaluate_files_shape_mismatch(tmp_path):
    """A shape mismatch must raise, not trip an assert that -O can strip."""
    pred = tmp_path / "p.nii.gz"
    tgt = tmp_path / "t.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.int16), np.eye(4)), str(pred))
    nib.save(nib.Nifti1Image(np.zeros((6, 6, 6), np.int16), np.eye(4)), str(tgt))
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_files([pred], [tgt])


def test_print_report_empty(capsys):
    """An empty result list reports nothing instead of printing NaN means."""
    print_report([])
    assert "no results to report" in capsys.readouterr().out


def test_converter_skips_incomplete_cases(tmp_path, capsys):
    """A case missing a modality is skipped; only complete cases are counted."""
    root = tmp_path / "BraTS"
    complete = root / "BraTS2021_00000"
    incomplete = root / "BraTS2021_00001"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    for m in ["t1", "t1ce", "t2", "flair", "seg"]:
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4)),
                 str(complete / f"BraTS2021_00000_{m}.nii.gz"))
    # incomplete case: flair file missing
    for m in ["t1", "t1ce", "t2", "seg"]:
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4)),
                 str(incomplete / f"BraTS2021_00001_{m}.nii.gz"))
    from src.brats_to_nnunet import convert
    n = convert(str(root), str(tmp_path / "nnraw"))
    assert n == 1
    assert "skip BraTS2021_00001" in capsys.readouterr().out
    dj = json.loads((tmp_path / "nnraw" / "dataset.json").read_text())
    assert dj["numTraining"] == 1
    assert dj["labels"] == {"background": 0, "NCR": 1, "ED": 2, "ET": 4}
    assert dj["channel_names"] == {"0": "t1", "1": "t1ce", "2": "t2", "3": "flair"}
    assert len(list((tmp_path / "nnraw" / "imagesTr").glob("*.nii.gz"))) == 4
    assert len(list((tmp_path / "nnraw" / "labelsTr").glob("*.nii.gz"))) == 1


def test_converter_empty_root(tmp_path):
    """An empty BraTS root fails with a clear error instead of writing junk."""
    from src.brats_to_nnunet import convert
    root = tmp_path / "BraTS"
    root.mkdir()
    with pytest.raises(SystemExit, match="no case directories"):
        convert(str(root), str(tmp_path / "nnraw"))


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
