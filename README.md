# Boundary Aware Residual Attention UNet for Satellite Image Segmentation

A **Residual Attention U-Net** for semantic segmentation of Sentinel-2 satellite imagery over Sylhet, Bangladesh. The model classifies 7 LULC classes using residual connections, attention gates, NDVI/NDWI spectral index fusion, and a morphological boundary-weighted Dice + Focal loss. Snow/Ice and Clouds are excluded as they are absent in the Sylhet study area.

## Features
- **Residual Attention U-Net** — residual connections for gradient flow + attention gates on skip connections
- **6-Channel Feature Fusion** — on-the-fly NDVI and NDWI appended to B2/B3/B4/B8 bands
- **Boundary-Weighted Focal Loss** — morphological edge extraction scales loss at class boundaries
- **Automated Patching** — handles large GeoTIFFs by patchifying into 256×256 segments
- **Full Evaluation Pipeline** — per-class metrics, confusion matrices, color-coded boundary predictions
- **Ablation Study Runner** — automated 3-configuration ablation with comparison table
- **Flexible LR Schedules** — Cosine Annealing or validation-plateau decay

## Project Structure
```
.
├── main.py                 # Training + evaluation pipeline
├── run_ablation.py         # Automated 5-run ablation study
├── src/
│   ├── model.py            # Residual Attention U-Net architecture
│   ├── dataset.py          # Data loading, patching, preprocessing
│   ├── utils.py            # Loss functions, metrics, visualization
│   └── boundary_metrics.py # Boundary-aware evaluation + color-coded viz
├── dataset/
│   ├── sylhet_sentinel2_30m_2023.tif
│   └── sylhet_esri_lulc_30m_mask_2023.tif
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

### Hardware Requirements
- **GPU recommended**: `--batch_size 16` needs ~8GB VRAM
- **CPU fallback**: reduce `--batch_size` to 4

## Usage

### Training + Evaluation

```bash
python main.py --data_path dataset --output_dir results
```

With explicit GeoTIFF paths:
```bash
python main.py \
  --image_tif dataset/sylhet_sentinel2_30m_2023.tif \
  --mask_tif dataset/sylhet_esri_lulc_30m_mask_2023.tif \
  --output_dir results
```

### Ablation Study

Run all 3 configurations automatically:

```bash
python run_ablation.py --data_path dataset --epochs 200 --batch_size 16
```

This runs the following configurations sequentially and produces a comparison table:

| Run | Configuration | Attention/Residual | Boundary Loss | NDVI/NDWI | LR Schedule |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | **Version 1 (Baseline)** | ✅ | ❌ | ❌ | Cosine |
| 2 | **Version 2 (Baseline + NDVI/NDWI)** | ✅ | ❌ | ✅ | Cosine |
| 3 | **Version 3 (Proposed)** | ✅ | ✅ | ✅ | Cosine |

Output structure:
```
ablation_results/
├── 1_Baseline/
├── 2_Baseline_NDVI/
├── 3_Proposed/
└── ablation_comparison.csv     # Side-by-side metric comparison
```

### Empirical Results

The final results of the ablation study (trained on Kaggle GPU T4 for 200 epochs) are shown below:

#### Global Performance Comparison

| Configuration | Overall Accuracy | Mean IoU (7 classes) | Weighted IoU | Mean F1 | BF Score (Edge F1) | Boundary IoU | Duration (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Version 1 (Baseline)** | 90.74% | 70.14% | 83.81% | 81.10% | 0.6184 | 44.76% | 1897.7 |
| **Version 2 (Baseline + NDVI/NDWI)** | 91.19% | 72.43% | 84.56% | 82.91% | 0.6326 | 46.26% | 1907.0 |
| **Version 3 (Proposed - Mult 0.5)** | 91.55% | 73.53% | 85.14% | 83.73% | 0.6404 | 47.10% | 1907.2 |
| **Version 4 (Proposed - Mult 1.0)** | 91.52% | 74.15% | 85.17% | 84.18% | 0.6493 | 48.07% | 1916.4 |
| **Version 5 (Proposed - Mult 2.0)** | **91.69%** | **74.98%** | **85.40%** | **84.86%** | **0.6571** | **48.93%** | 1911.3 |

> **Note:** Mean IoU is computed over the 7 active classes only (Snow/Ice and Clouds excluded — both absent in Sylhet).

*Key Insight:* Moving from V1 to V5 yields a net improvement of **+4.84% Mean IoU**, **+3.87% BF Score (Edge F1)**, and **+4.17% Boundary IoU**, demonstrating the compounding benefits of multi-spectral index fusion and morphological edge-weighted loss.

#### Per-Class IoU Comparison

| Class | Version 1 (Baseline) | Version 2 (Baseline + NDVI/NDWI) | Version 3 (Mult 0.5) | Version 4 (Mult 1.0) | Version 5 (Mult 2.0) | Net Progress (V1 → V5) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Water** | 94.01% | 94.22% | 94.30% | 94.46% | **94.49%** | +0.48% |
| **Trees** | 69.85% | 70.84% | 72.52% | 71.64% | **72.61%** | +2.76% |
| **Flooded Vegetation** | 46.00% | 50.96% | 52.54% | 54.08% | **55.53%** | **+9.53%** |
| **Crops** | 88.35% | 88.62% | 88.98% | 89.08% | **89.13%** | +0.78% |
| **Built Area** | 55.73% | 59.41% | 61.05% | 61.09% | **61.60%** | **+5.87%** |
| **Bare Ground** | 85.60% | 88.53% | 89.53% | **91.37%** | 90.32% | **+4.72%** |
| **Rangeland** | 51.42% | 54.44% | 55.76% | 57.29% | **61.20%** | **+9.78%** |

### Manual Ablation (individual runs)

```bash
# Version 1 (Baseline)
python main.py --boundary_multiplier 0.0 --no_ndvi --output_dir results/baseline

# Version 2 (Baseline + NDVI/NDWI)
python main.py --boundary_multiplier 0.0 --output_dir results/baseline_ndvi

# Version 3 (Proposed)
python main.py --output_dir results/proposed
```

### Evaluation Outputs

Each run produces the following in `--output_dir`:

| File | Description |
|---|---|
| `evaluation_results.csv` | Per-class IoU, Precision, Recall, F1, Accuracy + summary |
| `confusion_matrix.png` | Heatmap (raw counts + normalized %) |
| `per_class_iou.png` | Per-class IoU bar chart with mean IoU lines |
| `all_metrics_chart.png` | Grouped bar chart (IoU, Precision, Recall, F1) |
| `predictions.png` | Samples: Input → NDVI → Ground Truth → Prediction |
| `training_history.png` | Loss and accuracy curves |
| `class_legend.png` | Color legend for segmentation classes |
| `boundary_results_global.csv` | Global BF-score, Boundary IoU |
| `boundary_results_per_class.csv` | Per-class boundary metrics |
| `boundary_predictions.png` | Color-coded boundary predictions (see below) |
| `boundary_metrics_chart.png` | Per-class boundary metrics bar chart |

### Boundary Prediction Visualization

The boundary prediction figure uses a 4-column layout with color-coded correctness:

| Column | Content |
|---|---|
| Input | RGB composite |
| GT Boundary | Ground truth boundaries in **cyan** |
| Pred Boundary | 🟢 **Green** = Correct (TP), 🔴 **Red** = Missed (FN), 🟡 **Yellow** = False (FP) |
| Error Map | Errors only on dark background |

## Parameters

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `dataset` | Path to dataset directory |
| `--image_tif` | `None` | Sentinel-2 GeoTIFF path (overrides auto-detection) |
| `--mask_tif` | `None` | ESRI LULC mask GeoTIFF path (overrides auto-detection) |
| `--patch_size` | `256` | Patch dimensions (square) |
| `--patch_step` | `256` | Step size for patching (< patch_size for overlap) |
| `--epochs` | `200` | Max training epochs |
| `--batch_size` | `16` | Training batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--patience` | `30` | Early stopping patience |
| `--model_save_path` | `model.keras` | Path to save trained model |
| `--output_dir` | `results` | Directory for evaluation outputs |
| `--num_samples` | `5` | Prediction samples to visualize |
| `--disable_attention` | `false` | Disable attention gates |
| `--disable_residual` | `false` | Disable residual connections |
| `--boundary_kernel_size` | `3` | Morphological gradient kernel size |
| `--boundary_multiplier` | `2.0` | Boundary pixel loss weight (0.0 to disable) |
| `--lr_schedule` | `cosine` | LR schedule: `cosine` or `plateau` |
| `--no_ndvi` | `false` | Disable NDVI/NDWI index computation |

## Boundary-Aware Evaluation

Boundary regions are extracted using the **morphological gradient**:

```
B(L) = dilate(L, k) − erode(L, k)
```

| Metric | Formula | Description |
|---|---|---|
| **Boundary Precision** | TP / (TP + FP) | Predicted boundary pixels that are true boundaries |
| **Boundary Recall** | TP / (TP + FN) | True boundary pixels correctly predicted |
| **BF-score** | 2·P·R / (P+R) | Harmonic mean of boundary precision and recall |
| **Boundary IoU** | TP / (TP+FP+FN) | IoU restricted to boundary pixels |

## Classes (ESRI LULC)

Snow/Ice and Clouds are **excluded** — neither class has any pixels in the Sylhet study area. They are remapped to label 255 (ignored by the loss and all metrics).

| Model ID | ESRI Code | Class | Color |
|:---:|:---:|---|---|
| 0 | 1 | Water | ![#1A5BAB](https://via.placeholder.com/15/1A5BAB/1A5BAB.png) Blue |
| 1 | 2 | Trees | ![#358221](https://via.placeholder.com/15/358221/358221.png) Green |
| 2 | 4 | Flooded Vegetation | ![#87D19E](https://via.placeholder.com/15/87D19E/87D19E.png) Light Green |
| 3 | 5 | Crops | ![#FFDB5C](https://via.placeholder.com/15/FFDB5C/FFDB5C.png) Yellow |
| 4 | 7 | Built Area | ![#ED022A](https://via.placeholder.com/15/ED022A/ED022A.png) Red |
| 5 | 8 | Bare Ground | ![#EDE9E4](https://via.placeholder.com/15/EDE9E4/EDE9E4.png) Beige |
| 6 | 11 | Rangeland | ![#C6AD8D](https://via.placeholder.com/15/C6AD8D/C6AD8D.png) Tan |
| *(ignored)* | 9 | ~~Snow/Ice~~ | absent in Sylhet |
| *(ignored)* | 10 | ~~Clouds~~ | absent in Sylhet |

## Kaggle Deployment

1. Push repo to GitHub
2. Create a Kaggle Notebook → set **GPU T4 x2** + **Internet ON**
3. Import notebook and set your repo URL:
   ```python
   GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
   ```
4. **Run All** — results save to `/kaggle/working/results/`

| Parameter | Value | Notes |
|---|---|---|
| `--patch_step` | `128` | 50% overlap for more samples |
| `--batch_size` | `16` | Safe for 16GB T4/P100 VRAM |
| `--epochs` | `200` | With early stopping (patience=30) |

## Acknowledgments
- [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) — multispectral satellite imagery
- [ESRI LULC](https://www.arcgis.com/home/item.html?id=cfcb7609de5f478eb7666240902d4d3d) — land use / land cover labels
