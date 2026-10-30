# BraTS Brain Tumor Segmentation

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Brain tumor segmentation on the **BraTS** challenge data, with two training routes:

1. **nnU-Net v2** (recommended) — data converter `BraTS → nnU-Net raw format`
   plus the exact training/inference commands.
2. **MONAI U-Net** — a from-scratch training script in this repo for learning
   and for small experiments.

Evaluation follows the BraTS convention: Dice for whole tumor (WT),
tumor core (TC) and enhancing tumor (ET).

## BraTS label convention

BraTS segmentations use labels `{0: background, 1: necrotic/non-enhancing core,
2: edema, 4: enhancing tumor}`. Regions:

| Region | Labels |
|---|---|
| Whole tumor (WT) | 1, 2, 4 |
| Tumor core (TC) | 1, 4 |
| Enhancing tumor (ET) | 4 |

Four MRI modalities per case: `t1`, `t1ce`, `t2`, `flair`.

## Quickstart — nnU-Net v2 route

```bash
pip install nnunetv2

# 1. convert BraTS raw layout to nnU-Net format (needs BraTS download w/ registration)
python -m src.brats_to_nnunet --brats-root /data/BraTS2021 \
    --out /data/nnUNet_raw/Dataset500_BraTS2021

# 2. plan + preprocess + train (GPU)
nnUNetv2_plan_and_preprocess -d 500 --verify_dataset_integrity
nnUNetv2_train 500 3d_fullres 0

# 3. inference
nnUNetv2_predict -i /data/test_images -o /data/test_pred \
    -d 500 -c 3d_fullres -f 0
```

## Quickstart — MONAI route (this repo)

```bash
pip install -r requirements.txt

# synthetic demo: fake BraTS-like data, 2 epochs on CPU, then evaluate
python examples/synthetic_demo.py --out out/demo

# real training (needs BraTS-style NIfTI pairs)
python -m src.train_monai --train-list train.csv --val-list val.csv \
    --out runs/exp1 --epochs 200
```

`train.csv` / `val.csv` format: `t1,t1ce,t2,flair,seg` paths per row.

## Modules

| Module | What it does |
|---|---|
| `src/brats_to_nnunet.py` | BraTS → nnU-Net v2 raw dataset (imagesTr/labelsTr/dataset.json) |
| `src/train_monai.py` | MONAI SegResNet training with Dice+CE loss |
| `src/infer.py` | Sliding-window inference → NIfTI |
| `src/evaluate.py` | WT/TC/ET Dice from BraTS label convention |

## Data access

BraTS requires registration: [Medical Segmentation Decathlon / BraTS](https://www.med.upenn.edu/cbica/brats/).
This repo ships **no medical data** — `examples/synthetic_demo.py` generates
toy data so every code path is testable.

## References

- Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE TMI 2015.
- Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods 2021.
- Cardoso et al., "MONAI: An open-source framework for deep learning in healthcare", arXiv 2022.

## License

MIT — see `LICENSE`.
