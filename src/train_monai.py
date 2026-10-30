"""MONAI SegResNet training for BraTS-style 4-channel -> label data.

CSV lists with header: t1,t1ce,t2,flair,seg (one case per row).
Loss: DiceCELoss. Patch-based training with RandCropByPosNegLabeld.
"""
import argparse
import csv
from pathlib import Path

import torch
from monai.data import Dataset, DataLoader, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, RandCropByPosNegLabeld, ToTensord,
)


def read_list(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "image": [r["t1"], r["t1ce"], r["t2"], r["flair"]],
                "label": r["seg"],
            })
    return rows


def get_transforms(patch=(96, 96, 96), pixdim=(1.0, 1.0, 1.0)):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250,
                             b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=patch, pos=1, neg=1, num_samples=2),
        ToTensord(keys=["image", "label"]),
    ])


def train(train_csv, val_csv, out_dir, epochs=200, lr=1e-4, batch=1, patch=96,
          device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tf = get_transforms(patch=(patch, patch, patch))

    train_ds = Dataset(read_list(train_csv), transform=tf)
    val_ds = Dataset(read_list(val_csv), transform=get_transforms(patch=(patch, patch, patch)))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=2, collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    model = SegResNet(spatial_dims=3, in_channels=4, out_channels=5,
                      init_filters=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    best = -1.0
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for batch_data in train_loader:
            x = batch_data["image"].to(device)
            y = batch_data["label"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        print(f"epoch {epoch + 1}/{epochs} loss={tot / max(n, 1):.4f}", flush=True)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            dices = []
            with torch.no_grad():
                for v in val_loader:
                    x = v["image"].to(device)
                    pred = sliding_window_inference(x, (patch, patch, patch), 1,
                                                    model).argmax(1)
                    y = v["label"].to(device)
                    for c in (1, 2, 4):
                        p = (pred == c)
                        t = (y[:, 0] == c)
                        inter = (p & t).float().sum()
                        d = (2 * inter / (p.float().sum() + t.float().sum() + 1e-8))
                        dices.append(d.item())
            score = sum(dices) / max(len(dices), 1)
            print(f"  val mean dice={score:.4f}", flush=True)
            if score > best:
                best = score
                torch.save(model.state_dict(), out_dir / "best.pt")
                print("  saved best.pt", flush=True)
    torch.save(model.state_dict(), out_dir / "last.pt")
    print("done.")


def main():
    ap = argparse.ArgumentParser(description="Train SegResNet on BraTS-style data")
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--val-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--patch", type=int, default=96)
    args = ap.parse_args()
    train(args.train_list, args.val_list, args.out, args.epochs, args.lr,
          args.batch, args.patch)


if __name__ == "__main__":
    main()
