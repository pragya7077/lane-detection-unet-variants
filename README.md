# 🚗 Lane Detection — Comparative Study of 7 U-Net Variants

A systematic benchmark of **7 U-Net architectures** for semantic lane segmentation on the [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple) dataset, evaluated using **10-Fold Cross-Validation** across 7 metrics.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Models Compared](#models-compared)
- [Dataset](#dataset)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Authors](#authors)

---

## Overview

Lane detection is a foundational perception task in autonomous driving. This project treats it as a **binary semantic segmentation problem** — classifying each pixel as lane or non-lane — and systematically compares 7 progressive U-Net variants to identify the best-performing architecture.

All models are trained and evaluated under **identical conditions** to isolate the contribution of each architectural innovation.

---

## Models Compared

| Model | Key Innovation |
|---|---|
| **Attention U-Net** | Attention gates on skip connections to suppress background noise |
| **U-Net++** | Dense nested sub-networks for richer multi-scale feature aggregation |
| **ResUNet** | Residual (identity shortcut) connections inside encoder/decoder blocks |
| **TransUNet** | Vision Transformer bottleneck for long-range global context |
| **nnU-Net** | Self-configuring framework that auto-tunes architecture hyperparameters |
| **U²-Net** | Nested U-blocks (RSU) enabling fine + coarse feature extraction at every stage |
| **W-Net** | Two chained U-Nets: first segments, second refines using original + first-stage output |

---

## Dataset

**TuSimple Lane Detection Dataset**

| Property | Value |
|---|---|
| Source | TuSimple Inc. (2017 public lane detection challenge) |
| Training images | 3,646 |
| Image resolution | 1280×720 → resized to 512×256 |
| Lanes per frame | 2–4 |
| Annotation format | 2D polylines at fixed y-intervals |

Download via Kaggle: `manideep1108/tusimple`

---

## Results

10-Fold Cross-Validation averages on TuSimple. **Ranked by DSC ↑** (higher is better). HD95 ↓ (lower is better).

| Rank | Model | DSC ↑ | Jaccard ↑ | Precision ↑ | Recall ↑ | HD95 ↓ | U-Score ↑ | Accuracy ↑ |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **U²-Net** | **0.6905** | **0.5342** | 0.6465 | 0.7447 | **8.155** | **0.6017** | **0.9817** |
| 🥈 2 | W-Net | 0.6877 | 0.5314 | **0.6470** | 0.7394 | 10.289 | 0.5989 | 0.9817 |
| 🥉 3 | U-Net++ | 0.6840 | 0.5267 | 0.6370 | 0.7431 | 9.571 | 0.5945 | 0.9812 |
| 4 | Attention U-Net | 0.6803 | 0.5221 | 0.6228 | **0.7545** | 9.859 | 0.5902 | 0.9806 |
| 5 | ResUNet | 0.6799 | 0.5219 | 0.6408 | 0.7287 | 10.508 | 0.5899 | 0.9812 |
| 6 | TransUNet | 0.6677 | 0.5085 | 0.6412 | 0.7039 | 14.354 | 0.5768 | 0.9810 |
| 7 | nnU-Net | 0.6234 | 0.4604 | 0.5142 | **0.8093** | 19.926 | 0.5291 | 0.9720 |

---

## Repository Structure

```
lane-detection-unet-variants/
│
├── fyp-2026-kaggle.ipynb        # Main notebook (all parts)
│
├── README.md
│
└── results/                     # Output directory (generated on run)
    ├── kfold_csvs/
    │   ├── fold_01_performance.csv
    │   ├── ...
    │   ├── fold_10_performance.csv
    │   └── fold_11_AVERAGE_summary.csv   ← main results table
    ├── tusimple_samples.png              ← dataset visualizations
    ├── training_curves_kfold.png
    ├── loss_<ModelName>.png              ← per-model loss curves
    ├── foldwise_dsc.png
    ├── foldwise_hausdorff.png
    ├── foldwise_uscore.png
    ├── visual_predictions.png            ← all 7 models on same image
    ├── performance_bar_chart_cv.png
    ├── dendrogram.png                    ← hierarchical clustering
    ├── complexity_timing.csv
    └── complexity_timing_chart.png
```

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/lane-detection-unet-variants.git
cd lane-detection-unet-variants
```

### 2. Install dependencies
```bash
pip install torch torchvision albumentations opencv-python scikit-learn tqdm scipy einops timm pandas matplotlib kaggle
```

### 3. Set up Kaggle API
- Go to [kaggle.com](https://www.kaggle.com) → Account → **Create New Token**
- Place `kaggle.json` at `~/.kaggle/kaggle.json`

### 4. Run the notebook
Open `fyp-2026-kaggle.ipynb` in Jupyter or Kaggle and run all cells top to bottom.

> **Recommended:** Run on Kaggle (free GPU) or Google Colab with a T4/A100 GPU. CPU-only training is supported but very slow.

---

## Methodology

| Setting | Value |
|---|---|
| Input resolution | 512 × 256 |
| Optimizer | Adam (lr=1e-4, weight decay=1e-5) |
| Loss function | BCE + Dice (combined for class imbalance) |
| Cross-validation | 10-Fold |
| Epochs per fold | 5 (50 total per model) |
| Batch size | 8 |
| Augmentations | Horizontal flip, brightness/contrast jitter, Gaussian noise, motion blur, random shadow, affine transforms |

**Evaluation Metrics:** DSC (Dice), Jaccard (IoU), Precision, Recall, Hausdorff-95, U-Score, Pixel Accuracy

---

## Key Findings

- **U²-Net is the best overall performer** — highest DSC (0.6905), Jaccard (0.5342), U-Score (0.6017), and lowest HD95 (8.155). Its nested RSU blocks capture both fine and coarse lane features simultaneously.
- **W-Net is a close second** — dual encoder-decoder design produces the highest Precision (0.6470), confirming sharper, less noisy predictions.
- **Attention gates and dense skip connections outperform transformers** for this task — Attention U-Net and U-Net++ both beat TransUNet on DSC.
- **TransUNet underperforms** (rank 6, DSC 0.6677) — TuSimple's structured highway lanes rely more on local texture than long-range global context, limiting the benefit of the ViT encoder.
- **nnU-Net has the highest Recall (0.8093) but worst DSC** — it over-segments, catching more lane pixels but with high spatial inaccuracy (HD95: 19.93).

---

## Authors

**Sanhita Kulshrestha** (PRN: 2214110419) · **Pragya Raj** (PRN: 2214110424) · **Samiksha Dubey** (PRN: 2214110456)
