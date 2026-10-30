"""Synthetic BraTS-like demo: fake 4-channel data, tiny train, evaluate.

Generates toy volumes where a bright "tumor" ball has BraTS-like labels
(1 core, 2 edema ring, 4 enhancing spot), trains SegResNet for a couple of
epochs on CPU, runs inference and prints WT/TC/ET Dice.
"""
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def make_case(shape=(48, 48, 48), seed=0):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    c = np.array(shape) / 2 + rng.uniform(-3, 3, size=3)
    d2 = (xx - c[2]) ** 2 + (yy - c[1]) ** 2 + (zz - c[0]) ** 2
    seg = np.zeros(shape, dtype=np.int16)
    seg[d2 < 144] = 2          # edema
    seg[d2 < 64] = 1           # core
    seg[d2 < 16] = 4           # enhancing
    mods = []
    for i in range(4):
        base = rng.normal(50, 10, shape).astype(np.float32)
        base[seg == 2] += 25
        base[seg == 1] += 10
        base[seg == 4] += 40 + i * 5
        mods.append(base)
    return mods, seg


def write_case(case_dir, case_id, seed):
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    mods, seg = make_case(seed=seed)
    names = ["t1", "t1ce", "t2", "flair"]
    paths = {}
    for name, m in zip(names, mods):
        p = case_dir / f"{case_id}_{name}.nii.gz"
        nib.save(nib.Nifti1Image(m, np.eye(4)), str(p))
        paths[name] = str(p)
    sp = case_dir / f"{case_id}_seg.nii.gz"
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(sp))
    paths["seg"] = str(sp)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Synthetic BraTS demo")
    ap.add_argument("--out", default="out/demo")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    data_dir = out / "data"
    rows = []
    for i, split in enumerate(["train", "train", "val"]):
        p = write_case(data_dir, f"synth-{i:02d}", seed=i)
        rows.append((split, p))
    for split in ("train", "val"):
        with open(out / f"{split}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t1", "t1ce", "t2", "flair", "seg"])
            w.writeheader()
            w.writerows([r for s, r in rows if s == split])

    from src.train_monai import train
    train(str(out / "train.csv"), str(out / "val.csv"), str(out / "runs"),
          epochs=args.epochs, patch=32, batch=1)

    # inference on the val case + evaluate
    import torch
    from src.infer import load_model
    from src.evaluate import brats_scores
    val = [r for s, r in rows if s == "val"][0]
    device = "cpu"
    model = load_model(str(out / "runs" / "best.pt"), device)
    from monai.transforms import Compose, LoadImage, EnsureChannelFirst
    import torch as _torch
    tf = Compose([LoadImage(image_only=True), EnsureChannelFirst()])
    mods = _torch.stack([_torch.as_tensor(np.asanyarray(tf(val[m]))) for m in
                         ("t1", "t1ce", "t2", "flair")]).unsqueeze(0)
    with _torch.no_grad():
        from monai.inferers import sliding_window_inference
        pred = sliding_window_inference(mods, (32, 32, 32), 1, model).argmax(1)
    target = np.asanyarray(nib.load(val["seg"]).dataobj)
    print("synthetic val scores:", brats_scores(pred[0].numpy(), target))


if __name__ == "__main__":
    main()
