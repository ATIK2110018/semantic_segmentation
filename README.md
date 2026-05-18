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
├── evaluate.py             # Standalone evaluation (load saved model)
├── kaggle_evaluate.py      # Self-contained Kaggle evaluation (no src/ imports)
├── models/                 # Saved model weights (.keras)
├── results/                # Evaluation outputs (plots, CSV, confusion matrix)
├── requirements.txt        # Dependencies
└── Semantic segmentation dataset/   # Dataset directory
```

## Installation

```bash
pip install -r requirements.txt
```

### Hardware Requirements
- **GPU recommended**: Default config (`--batch_size 16`) is optimized for Kaggle/Colab GPUs (T4 or P100)
- **CPU fallback**: Reduce `--batch_size` to 4 or increase `--patch_step` if running locally

## Usage

### Training + Evaluation (single run)

Training automatically runs full evaluation after completion — no separate step needed.

**Local:**
```bash
python main.py --output_dir results/full_model
```

**Kaggle:**
```bash
python main.py \
  --data_path "/kaggle/input/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery/Semantic segmentation dataset" \
  --model_save_path /kaggle/working/residual_attention_unet.keras \
  --output_dir /kaggle/working/results/full_model
```

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
| `--data_path` | `Semantic segmentation dataset` | Path to dataset |
| `--patch_size` | `256` | Patch dimensions |
| `--patch_step` | `160` | Step size for patching |
| `--epochs` | `170` | Max training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `8e-6` | Initial learning rate |
| `--patience` | `20` | Early stopping patience |
| `--model_save_path` | `model.keras` | Path to save trained model |
| `--output_dir` | `results` | Directory for evaluation outputs |
| `--num_samples` | `8` | Number of prediction samples to visualize |
| `--disable_attention` | `false` | Disable attention gates (ablation) |
| `--disable_residual` | `false` | Disable residual connections (ablation) |

## Classes

| ID | Class | Color |
|---|---|---|
| 0 | Unlabeled | ![#9B9B9B](https://via.placeholder.com/15/9B9B9B/9B9B9B.png) Gray |
| 1 | Building | ![#3C1098](https://via.placeholder.com/15/3C1098/3C1098.png) Dark Purple |
| 2 | Land | ![#8429F6](https://via.placeholder.com/15/8429F6/8429F6.png) Purple |
| 3 | Road | ![#6EC1E4](https://via.placeholder.com/15/6EC1E4/6EC1E4.png) Light Blue |
| 4 | Vegetation | ![#FEDD3A](https://via.placeholder.com/15/FEDD3A/FEDD3A.png) Yellow |
| 5 | Water | ![#E2A929](https://via.placeholder.com/15/E2A929/E2A929.png) Orange |

## Dataset

The [Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) dataset should be organized as:
```
Semantic segmentation dataset/
├── Tile 1/
│   ├── images/
│   │   └── image_part_001.jpg
│   └── masks/
│       └── image_part_001.png
├── Tile 2/
│   └── ...
└── Tile 8/
```

## Acknowledgments
- [segmentation-models](https://github.com/qubvel/segmentation_models) — metrics and loss functions
- [patchify](https://github.com/dovahcrow/patchify.py) — image patching
- [Humans in the Loop](https://humansintheloop.org/) — aerial imagery dataset
