# Semantic Segmentation with Residual Attention U-Net

This project implements a **Residual Attention U-Net** for semantic segmentation of aerial imagery using the [Humans in the Loop](https://humansintheloop.org/) dataset. The model combines residual connections and attention gates within a U-Net architecture, trained with a combined Dice + Focal loss.

## Features
- **Residual Attention U-Net**: Residual connections for gradient flow + attention gates for skip connection refinement
- **Custom Loss**: Combined Dice and Focal loss with class weighting and unlabeled pixel masking
- **Automated Patching**: Handles large aerial images by patchifying them into 256×256 segments
- **Built-in Evaluation**: Per-class metrics (IoU, Precision, Recall, F1), confusion matrix, and prediction visualizations — all generated automatically after training
- **Ablation Study Support**: Toggle residual/attention components via CLI flags

## Project Structure
```
.
├── src/
│   ├── model.py            # Residual Attention U-Net architecture
│   ├── dataset.py          # Data loading, patching, and preprocessing
│   └── utils.py            # Loss functions, metrics, and visualization helpers
├── main.py                 # Training + full evaluation pipeline
├── dataset/                # GeoTIFF dataset directory
│   ├── sylhet_sentinel2_30m_2023.tif
│   └── sylhet_esri_lulc_30m_mask_2023.tif
├── models/                 # Saved model weights (.keras)
├── results/                # Evaluation outputs (plots, CSV, confusion matrix)
└── requirements.txt        # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

### Hardware Requirements
- **GPU recommended**: Default config (`--batch_size 16`) is optimized for GPUs with at least 8GB VRAM.
- **CPU fallback**: Reduce `--batch_size` to 4 or increase `--patch_step` if running locally without a GPU.

## Usage

### Training + Evaluation (single run)

Training automatically runs full evaluation after completion.

**Sentinel-2 + ESRI LULC GeoTIFFs (using the `dataset/` folder):**
```bash
python main.py \
  --data_path dataset \
  --patch_step 160 \
  --output_dir results/sentinel_esri
```

**Local (if using original aerial imagery dataset):**
```bash
python main.py --data_path "Semantic segmentation dataset" --output_dir results/aerial_imagery
```

**Kaggle:** See the [Kaggle Deployment](#kaggle-deployment) section below.

### Ablation Study

Run each variant to compare the contribution of Residual and Attention components:

**1. Full Model (Residual + Attention):**
```bash
python main.py \
  --model_save_path models/residual_attention_unet.keras \
  --output_dir results/full_model
```

**2. Residual Only (no attention):**
```bash
python main.py --disable_attention \
  --model_save_path models/residual_only_unet.keras \
  --output_dir results/residual_only
```

**3. Attention Only (no residual):**
```bash
python main.py --disable_residual \
  --model_save_path models/attention_only_unet.keras \
  --output_dir results/attention_only
```

**4. Base U-Net (no residual, no attention):**
```bash
python main.py --disable_attention --disable_residual \
  --model_save_path models/base_unet.keras \
  --output_dir results/base_unet
```

### Evaluation Outputs

After each run, the `--output_dir` will contain:

| File | Description |
|---|---|
| `evaluation_results.csv` | Per-class IoU, Precision, Recall, F1, Accuracy + summary metrics |
| `confusion_matrix.png` | Heatmap (raw counts + normalized %) |
| `per_class_iou.png` | Per-class IoU bar chart with mean IoU lines |
| `all_metrics_chart.png` | Grouped bar chart (IoU, Precision, Recall, F1) |
| `predictions.png` | 8 samples: Input → Ground Truth → Prediction |
| `training_history.png` | Loss and IoU curves over epochs |
| `class_legend.png` | Color legend for the 6 segmentation classes |

### Parameters

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `dataset` | Path to dataset directory |
| `--image_tif` | `None` | Optional Sentinel-2 GeoTIFF path (overrides auto-detection) |
| `--mask_tif` | `None` | Optional ESRI LULC mask GeoTIFF path (overrides auto-detection) |
| `--patch_size` | `256` | Patch dimensions (square) |
| `--patch_step` | `256` | Step size for patching (use < patch_size for overlap) |
| `--epochs` | `200` | Max training epochs |
| `--batch_size` | `16` | Training batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--patience` | `30` | Early stopping patience |
| `--model_save_path` | `model.keras` | Path to save trained model |
| `--output_dir` | `results` | Directory for evaluation outputs |
| `--num_samples` | `8` | Number of prediction samples to visualize |
| `--disable_attention` | `false` | Disable attention gates (ablation) |
| `--disable_residual` | `false` | Disable residual connections (ablation) |

## Classes (ESRI LULC)

The GeoTIFF dataset uses ESRI Land Use / Land Cover classes:

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

### Step 1 — Push to GitHub

Make sure your dataset TIF files and notebook are committed:
```bash
git add .
git commit -m "Add 30m GeoTIFF dataset and Kaggle notebook"
git push origin main
```

### Step 2 — Create a Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Set **Accelerator → GPU T4 x2** (Settings panel on the right)
3. Enable **Internet** (Settings → Internet → On)
4. Import `kaggle_notebook.ipynb` from this repo (File → Import Notebook → GitHub URL)
   - Or copy-paste the notebook cells manually

### Step 3 — Set Your Repo URL & Run

In **Cell 1** of the notebook, set:
```python
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
```
Then **Run All** — training will stream live output directly in the notebook.

### Kaggle Training Parameters

| Parameter | Value | Notes |
|---|---|---|
| `--patch_size` | `256` | Full resolution patches |
| `--patch_step` | `128` | 50% overlap for more samples |
| `--batch_size` | `16` | Safe for 16 GB T4/P100 VRAM |
| `--epochs` | `200` | With early stopping (patience=30) |
| `--lr` | `1e-4` | Adam optimizer |

Outputs are saved to `/kaggle/working/results/30m_full_model/` and displayed inline after training.

## Acknowledgments
- [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) — multispectral satellite imagery
- [ESRI LULC](https://www.arcgis.com/home/item.html?id=cfcb7609de5f478eb7666240902d4d3d) — land use / land cover labels
