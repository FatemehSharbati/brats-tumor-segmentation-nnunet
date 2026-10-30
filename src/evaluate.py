"""BraTS label convention and region-wise Dice.

BraTS seg labels: 0 background, 1 necrotic/non-enhancing core,
2 edema, 4 enhancing tumor.
Regions: WT={1,2,4}, TC={1,4}, ET={4}.
"""
import numpy as np

REGIONS = {
    "WT": (1, 2, 4),
    "TC": (1, 4),
    "ET": (4,),
}


def region_mask(seg, region):
    seg = np.asarray(seg)
    mask = np.zeros_like(seg, dtype=bool)
    for label in REGIONS[region]:
        mask |= seg == label
    return mask


def dice_score(pred, target):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def brats_scores(pred_seg, target_seg):
    """Dice for WT, TC, ET. Returns dict."""
    return {
        region: dice_score(region_mask(pred_seg, region), region_mask(target_seg, region))
        for region in REGIONS
    }


def evaluate_files(pred_paths, target_paths):
    """Evaluate paired NIfTI files. Returns list of (name, scores)."""
    import nibabel as nib
    results = []
    for pp, tp in zip(pred_paths, target_paths):
        pred = np.asanyarray(nib.load(str(pp)).dataobj)
        target = np.asanyarray(nib.load(str(tp)).dataobj)
        assert pred.shape == target.shape, f"shape mismatch: {pp}"
        results.append((str(pp), brats_scores(pred, target)))
    return results


def print_report(results):
    regions = list(REGIONS)
    print(f"{'case':40s}" + "".join(f"{r:>10s}" for r in regions))
    agg = {r: [] for r in regions}
    for name, scores in results:
        print(f"{name[-40:]:40s}" + "".join(f"{scores[r]:10.3f}" for r in regions))
        for r in regions:
            agg[r].append(scores[r])
    print(f"{'mean':40s}" + "".join(f"{np.mean(agg[r]):10.3f}" for r in regions))
