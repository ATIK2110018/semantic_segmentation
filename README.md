# Semantic Segmentation with Residual Attention U-Net

A **Residual Attention U-Net** for semantic segmentation of satellite imagery using Sentinel-2 multispectral data and ESRI LULC labels. The model incrementally combines residual connections, attention gates, boundary-weighted loss, and spectral indices (NDVI/NDWI), trained with a combined Dice + Focal loss.

## Features
- **Residual Attention U-Net** — residual connections for gradient flow + attention gates on skip connections
- **6-Channel Feature Fusion** — on-the-fly NDVI and NDWI appended to B2/B3/B4/B8 bands
- **Boundary-Weighted Focal Loss** — morphological edge extraction scales loss at class boundaries
- **Automated Patching** — handles large GeoTIFFs by patchifying into 256×256 segments
- **Full Evaluation Pipeline** — per-class metrics, confusion matrices, color-coded boundary predictions
- **Ablation Study Runner** — automated 5-configuration ablation with comparison table
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

Run all 5 configurations automatically:

```bash
python run_ablation.py --data_path dataset --epochs 200 --batch_size 16
```

This runs the following configurations sequentially and produces a comparison table:

| Run | Model | Attention | Residual | Boundary Loss | NDVI/NDWI |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | Plain U-Net | ❌ | ❌ | ❌ | ❌ |
| 2 | Attention U-Net | ✅ | ❌ | ❌ | ❌ |
| 3 | Attention + Residual U-Net | ✅ | ✅ | ❌ | ❌ |
| 4 | Full Model (w/o NDVI/NDWI) | ✅ | ✅ | ✅ | ❌ |
| 5 | **Full Model (Proposed)** | ✅ | ✅ | ✅ | ✅ |

Output structure:
```
ablation_results/
├── 1_Plain_UNet/
├── 2_Attention_UNet/
├── 3_Attention_Residual_UNet/
├── 4_Full_Without_Indices/
├── 5_Full_Model/
└── ablation_comparison.csv     # Side-by-side metric comparison
```

### Manual Ablation (individual runs)

```bash
# Plain U-Net
python main.py --disable_attention --disable_residual --boundary_multiplier 0.0 --no_ndvi \
  --output_dir results/plain_unet

# Full model
python main.py --output_dir results/full_model
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

| ID | Class | Color |
|---|---|---|
| 1 | Water | ![#1A5BAB](https://via.placeholder.com/15/1A5BAB/1A5BAB.png) Blue |
| 2 | Trees | ![#358221](https://via.placeholder.com/15/358221/358221.png) Green |
| 3 | Flooded Vegetation | ![#87D19E](https://via.placeholder.com/15/87D19E/87D19E.png) Light Green |
| 4 | Crops | ![#FFDB5C](https://via.placeholder.com/15/FFDB5C/FFDB5C.png) Yellow |
| 5 | Built Area | ![#ED022A](https://via.placeholder.com/15/ED022A/ED022A.png) Red |
| 6 | Bare Ground | ![#EDE9E4](https://via.placeholder.com/15/EDE9E4/EDE9E4.png) Beige |
| 7 | Snow/Ice | ![#F2FAFF](https://via.placeholder.com/15/F2FAFF/F2FAFF.png) White |
| 8 | Clouds | ![#C8C8C8](https://via.placeholder.com/15/C8C8C8/C8C8C8.png) Gray |
| 9 | Rangeland | ![#C6AD8D](https://via.placeholder.com/15/C6AD8D/C6AD8D.png) Tan |

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
