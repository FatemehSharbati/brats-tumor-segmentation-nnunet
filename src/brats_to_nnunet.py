"""Convert a BraTS raw download to nnU-Net v2 raw dataset format.

Expected BraTS layout:
    <brats-root>/BraTS2021_00000/BraTS2021_00000_t1.nii.gz
    <brats-root>/BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz
    <brats-root>/BraTS2021_00000/BraTS2021_00000_t2.nii.gz
    <brats-root>/BraTS2021_00000/BraTS2021_00000_flair.nii.gz
    <brats-root>/BraTS2021_00000/BraTS2021_00000_seg.nii.gz

Output (nnU-Net v2 raw):
    <out>/imagesTr/<case>_0000.nii.gz ... <case>_0003.nii.gz   (t1,t1ce,t2,flair)
    <out>/labelsTr/<case>.nii.gz
    <out>/dataset.json
"""
import argparse
import json
import shutil
from pathlib import Path

MODALITIES = ["t1", "t1ce", "t2", "flair"]


def convert(brats_root, out_dir, dataset_id=500, dataset_name="BraTS2021"):
    brats_root = Path(brats_root)
    out_dir = Path(out_dir)
    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(d for d in brats_root.iterdir() if d.is_dir())
    if not case_dirs:
        raise SystemExit(f"no case directories found in {brats_root}")

    n_ok = 0
    for case_dir in case_dirs:
        case_id = case_dir.name  # e.g. BraTS2021_00000
        mods = [case_dir / f"{case_id}_{m}.nii.gz" for m in MODALITIES]
        seg = case_dir / f"{case_id}_seg.nii.gz"
        if not all(p.exists() for p in mods) or not seg.exists():
            print(f"  skip {case_id}: missing files")
            continue
        for i, p in enumerate(mods):
            shutil.copy(p, images_tr / f"{case_id}_{i:04d}.nii.gz")
        shutil.copy(seg, labels_tr / f"{case_id}.nii.gz")
        n_ok += 1

    dataset_json = {
        "channel_names": {str(i): m for i, m in enumerate(MODALITIES)},
        "labels": {"background": 0, "NCR": 1, "ED": 2, "ET": 4},
        "numTraining": n_ok,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    (out_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"converted {n_ok} cases -> {out_dir}")
    return n_ok


def main():
    ap = argparse.ArgumentParser(description="BraTS -> nnU-Net v2 raw format")
    ap.add_argument("--brats-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-id", type=int, default=500)
    ap.add_argument("--dataset-name", default="BraTS2021")
    args = ap.parse_args()
    convert(args.brats_root, args.out, args.dataset_id, args.dataset_name)


if __name__ == "__main__":
    main()
