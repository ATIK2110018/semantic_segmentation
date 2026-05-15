# Semantic Segmentation with Residual Attention U-Net

This project implements a Residual Attention U-Net for semantic segmentation of aerial imagery. The model is built using TensorFlow and Keras.

## Features
- **Residual Attention U-Net**: Incorporates residual connections and attention gates for better feature learning.
- **Custom Loss**: Combined Dice and Focal loss with masking for unlabeled classes.
- **Automated Patching**: Handles large aerial images by patchifying them into smaller segments.
- **Clean Structure**: Modularized code for dataset handling, model architecture, and training.

## Project Structure
```
.
├── src/
│   ├── model.py         # Model architecture
│   ├── dataset.py       # Data loading and preprocessing
│   └── utils.py         # Helper functions (losses, metrics, visualization)
├── main.py              # Entry point for training
├── requirements.txt     # Dependencies
└── Semantic segmentation dataset/ # Dataset directory
```

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Hardware Requirements
The default configuration (`--batch_size 16`) is optimized for environments with a GPU like Google Colab or Kaggle (e.g., T4 or P100). 
If you still encounter OOM errors or want to run it locally with limited RAM, reduce the `--batch_size` to 4 or increase the `--patch_step`.

## Usage

### Training
To train the model with default parameters:
```bash
python main.py
```

To customize training:
```bash
python main.py --epochs 100 --batch_size 16 --lr 1e-5 --visualize
```

### Parameters
- `--data_path`: Path to the dataset (default: "Semantic segmentation dataset").
- `--patch_size`: Size of patches (default: 256).
- `--patch_step`: Step size for patching (default: 160).
- `--epochs`: Number of training epochs (default: 200).
- `--batch_size`: Batch size (default: 16).
- `--lr`: Learning rate (default: 8e-6).
- `--model_save_path`: Path to save the model (default: "model.keras").
- `--visualize`: Show predictions after training.

## Dataset
The dataset should be organized in tiles, with `images/` and `masks/` subdirectories in each tile.
Example:
`Semantic segmentation dataset/Tile 1/images/image_part_001.jpg`
`Semantic segmentation dataset/Tile 1/masks/image_part_001.png`

## Acknowledgments
- Uses `segmentation-models` and `patchify` libraries.
