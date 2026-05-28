# Lane Detection — 7 U-Net Variants Comparative Study

A systematic comparison of 7 U-Net architectures for semantic lane segmentation on the TuSimple dataset, evaluated using 10-Fold Cross-Validation.

## Models
- Attention U-Net
- U-Net++
- ResUNet
- TransUNet
- nnU-Net
- U²-Net
- W-Net

## Dataset
[TuSimple Lane Detection](https://www.kaggle.com/datasets/manideep1108/tusimple) — 3,646 highway driving images (512×256px), annotated with 2–4 lane polylines per frame.

## Setup

```bash
pip install torch torchvision albumentations opencv-python scikit-learn tqdm scipy einops timm pandas matplotlib kaggle
```

Place your `kaggle.json` at `~/.kaggle/kaggle.json` (get it from Kaggle → Account → Create New Token), then run all cells in `fyp-2026-kaggle.ipynb`.

> Recommended: Kaggle or Google Colab with a GPU. CPU training is supported but slow.

## Training Config

| Setting | Value |
|---|---|
| Input size | 512 × 256 |
| Optimizer | Adam (lr=1e-4) |
| Loss | BCE + Dice |
| Cross-validation | 10-Fold |
| Epochs per fold | 5 |
| Batch size | 8 |

## Authors
Sanhita Kulshrestha · Pragya Raj · Samiksha Dubey
