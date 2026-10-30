"""Sliding-window inference with a trained SegResNet -> NIfTI segmentation."""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Spacing, Orientation,
    ScaleIntensityRange,
)


def load_model(ckpt, device):
    model = SegResNet(spatial_dims=3, in_channels=4, out_channels=5, init_filters=16)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description="BraTS inference")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--t1", required=True)
    ap.add_argument("--t1ce", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--flair", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", type=int, default=96)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Orientation(axcodes="RAS"),
        ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
    ])
    mods = [tf(getattr(args, m)) for m in ("t1", "t1ce", "t2", "flair")]
    image = torch.stack([torch.as_tensor(np.asanyarray(m)) for m in mods]).unsqueeze(0)
    ref = nib.load(args.t1)

    model = load_model(args.ckpt, device)
    with torch.no_grad():
        pred = sliding_window_inference(
            image.to(device), (args.patch,) * 3, 1, model).argmax(1)
    seg = pred[0].cpu().numpy().astype(np.int16)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(seg, ref.affine, ref.header), str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
