"""
Kaggle Evaluation Notebook — Residual-Attention-U-Net
=====================================================
Self-contained evaluation script. No external src/ imports needed.

Setup on Kaggle:
  1. Add the dataset: "humansintheloop/semantic-segmentation-of-aerial-imagery"
  2. Upload your saved model as a Kaggle Dataset (e.g., "your-username/segmentation-model")
  3. Enable GPU accelerator
  4. Run this notebook

Outputs saved to /kaggle/working/results/
"""

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from patchify import patchify
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import csv

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Update these paths for your Kaggle environment
# ═══════════════════════════════════════════════════════════════════════════════

# Try common Kaggle dataset paths
DATASET_PATHS = [
    "/kaggle/input/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery/Semantic segmentation dataset",
    "/kaggle/input/semantic-segmentation-of-aerial-imagery/Semantic segmentation dataset",
    "Semantic segmentation dataset",  # local fallback
]

# Try common model paths
MODEL_PATHS = [
    "/kaggle/input/segmentation-model/residual_attention_unet.keras",
    "/kaggle/input/residual-attention-unet/residual_attention_unet.keras",
    "/kaggle/working/residual_attention_unet.keras",
    "models/residual_attention_unet.keras",  # local fallback
]

OUTPUT_DIR = "/kaggle/working/results"
PATCH_SIZE = 256
PATCH_STEP = 160
BATCH_SIZE = 16
NUM_VIS_SAMPLES = 8
SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["Unlabeled", "Building", "Land", "Road", "Vegetation", "Water"]
NUM_CLASSES = 6

CLASSES_RGB = {
    "Building":   [60, 16, 152],
    "Land":       [132, 41, 246],
    "Road":       [110, 193, 228],
    "Vegetation": [254, 221, 58],
    "Water":      [226, 169, 41],
    "Unlabeled":  [155, 155, 155],
}

CUSTOM_CMAP = ListedColormap([
    (155/255, 155/255, 155/255),  # Unlabeled
    (60/255,  16/255, 152/255),   # Building
    (132/255, 41/255, 246/255),   # Land
    (110/255, 193/255, 228/255),  # Road
    (254/255, 221/255, 58/255),   # Vegetation
    (226/255, 169/255, 41/255),   # Water
])

CLASS_COLORS_HEX = ['#9B9B9B', '#3C1098', '#8429F6', '#6EC1E4', '#FEDD3A', '#E2A929']


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Find paths
# ═══════════════════════════════════════════════════════════════════════════════

def find_path(candidates, label):
    for p in candidates:
        if os.path.exists(p):
            print(f"✅ Found {label} at: {p}")
            return p
    raise FileNotFoundError(
        f"❌ Could not find {label}. Tried:\n" + "\n".join(f"  - {p}" for p in candidates)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING (self-contained — mirrors src/dataset.py)
# ═══════════════════════════════════════════════════════════════════════════════

def rgb_to_label(mask):
    if mask.ndim == 2 or mask.shape[-1] == 1:
        return mask.astype(np.uint8)
    label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    label_mask[np.all(mask == CLASSES_RGB["Unlabeled"],  axis=-1)] = 0
    label_mask[np.all(mask == CLASSES_RGB["Building"],   axis=-1)] = 1
    label_mask[np.all(mask == CLASSES_RGB["Land"],       axis=-1)] = 2
    label_mask[np.all(mask == CLASSES_RGB["Road"],       axis=-1)] = 3
    label_mask[np.all(mask == CLASSES_RGB["Vegetation"], axis=-1)] = 4
    label_mask[np.all(mask == CLASSES_RGB["Water"],      axis=-1)] = 5
    return label_mask


def load_data(data_path, patch_size=256, step=160):
    images, masks = [], []
    tiles = [f"Tile {i}" for i in range(1, 9)]
    for tile in tiles:
        tile_path = os.path.join(data_path, tile)
        if not os.path.exists(tile_path):
            continue
        img_dir = os.path.join(tile_path, 'images')
        msk_dir = os.path.join(tile_path, 'masks')
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        for img_file in img_files:
            mask_file = img_file.replace('.jpg', '.png')
            img_path = os.path.join(img_dir, img_file)
            msk_path = os.path.join(msk_dir, mask_file)
            if not os.path.exists(msk_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            msk = cv2.imread(msk_path, cv2.IMREAD_COLOR)
            if img is not None and msk is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                new_h = (h // patch_size) * patch_size
                new_w = (w // patch_size) * patch_size
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                msk = cv2.resize(msk, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                label_msk = rgb_to_label(msk)
                img_patches = patchify(img, (patch_size, patch_size, 3), step=step)
                msk_patches = patchify(label_msk, (patch_size, patch_size), step=step)
                for r in range(img_patches.shape[0]):
                    for c in range(img_patches.shape[1]):
                        images.append(img_patches[r, c, 0])
                        masks.append(msk_patches[r, c])
    return np.array(images), np.array(masks)


def prepare_test_set(data_path, patch_size=256, step=160):
    """Load dataset and return only the test split (same split as training)."""
    x, y = load_data(data_path, patch_size, step)
    n_classes = len(np.unique(y))
    y_cat = to_categorical(y, num_classes=n_classes)
    _, x_test, _, y_test = train_test_split(x, y_cat, test_size=0.25, random_state=42)
    x_test = x_test.astype('float32') / 255.0
    return x_test, y_test, n_classes


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM LOSS (needed to load the model)
# ═══════════════════════════════════════════════════════════════════════════════

def get_masked_loss(class_weights):
    import segmentation_models as sm
    dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
    focal_loss = sm.losses.CategoricalFocalLoss()
    def masked_total_loss(y_true, y_pred):
        mask = tf.not_equal(tf.argmax(y_true, axis=-1), 0)
        mask = tf.cast(mask, tf.float32)
        loss = 0.5 * dice_loss(y_true, y_pred) + 0.5 * focal_loss(y_true, y_pred)
        return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-6)
    return masked_total_loss


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_confusion_matrix(model, x_test, y_test, batch_size=16):
    print("Computing predictions...")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_batches = (len(x_test) + batch_size - 1) // batch_size
    for i in range(0, len(x_test), batch_size):
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches}", end='\r')
        x_batch = x_test[i:i+batch_size]
        y_batch = y_test[i:i+batch_size]
        y_pred_batch = model.predict(x_batch, verbose=0)
        y_true_flat = np.argmax(y_batch, axis=-1).flatten()
        y_pred_flat = np.argmax(y_pred_batch, axis=-1).flatten()
        cm += confusion_matrix(y_true_flat, y_pred_flat, labels=np.arange(NUM_CLASSES))
    print(f"\n  Done — {len(x_test)} samples processed.")
    return cm


def compute_metrics(cm):
    per_class = {}
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        per_class[CLASS_NAMES[i]] = {
            'class_id': i, 'iou': iou, 'precision': precision,
            'recall': recall, 'f1': f1, 'accuracy': accuracy,
            'support': int(cm[i, :].sum())
        }

    all_ious = [per_class[n]['iou'] for n in CLASS_NAMES]
    labeled_ious = [per_class[n]['iou'] for n in CLASS_NAMES if n != "Unlabeled"]
    total_px = sum(per_class[n]['support'] for n in CLASS_NAMES)

    summary = {
        'overall_accuracy': np.trace(cm) / (cm.sum() + 1e-8),
        'mean_iou_all': np.mean(all_ious),
        'mean_iou_labeled': np.mean(labeled_ious),
        'weighted_iou': sum(per_class[n]['iou'] * per_class[n]['support'] for n in CLASS_NAMES) / (total_px + 1e-8),
        'mean_f1': np.mean([per_class[n]['f1'] for n in CLASS_NAMES]),
    }
    return per_class, summary


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, output_dir):
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[0], linewidths=0.5)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label'); axes[0].set_xlabel('Predicted Label')
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label'); axes[1].set_xlabel('Predicted Label')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show(); plt.close()


def plot_per_class_iou(per_class, summary, output_dir):
    ious = [per_class[n]['iou'] for n in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(CLASS_NAMES, ious, color=CLASS_COLORS_HEX, edgecolor='black', linewidth=0.8)
    for bar, iou in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{iou:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(y=summary['mean_iou_all'], color='red', linestyle='--', linewidth=2,
               label=f"Mean IoU (all): {summary['mean_iou_all']:.4f}")
    ax.axhline(y=summary['mean_iou_labeled'], color='green', linestyle='-.', linewidth=2,
               label=f"Mean IoU (labeled): {summary['mean_iou_labeled']:.4f}")
    ax.set_ylim(0, 1.0); ax.set_ylabel('IoU Score', fontsize=13)
    ax.set_title('Per-Class IoU Scores', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150, bbox_inches='tight')
    plt.show(); plt.close()


def plot_all_metrics(per_class, output_dir):
    x = np.arange(len(CLASS_NAMES))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    for j, (key, label, color) in enumerate(zip(
        ['iou', 'precision', 'recall', 'f1'],
        ['IoU', 'Precision', 'Recall', 'F1-Score'], colors)):
        vals = [per_class[n][key] for n in CLASS_NAMES]
        ax.bar(x + j*width, vals, width, label=label, color=color, alpha=0.85)
    ax.set_xlabel('Class', fontsize=13); ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Per-Class Evaluation Metrics', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width*1.5); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_chart.png'), dpi=150, bbox_inches='tight')
    plt.show(); plt.close()


def visualize_predictions(model, x_test, y_test, output_dir, num_samples=8, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4.5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    for i, idx in enumerate(indices):
        test_img = x_test[idx]
        true_mask = np.argmax(y_test[idx], axis=-1)
        pred_mask = np.argmax(model.predict(np.expand_dims(test_img, 0), verbose=0)[0], axis=-1)
        axes[i, 0].imshow(test_img); axes[i, 0].set_title(f'Input (#{idx})', fontsize=12, fontweight='bold'); axes[i, 0].axis('off')
        axes[i, 1].imshow(true_mask, cmap=CUSTOM_CMAP, vmin=0, vmax=5, interpolation='nearest')
        axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold'); axes[i, 1].axis('off')
        axes[i, 2].imshow(pred_mask, cmap=CUSTOM_CMAP, vmin=0, vmax=5, interpolation='nearest')
        axes[i, 2].set_title('Prediction', fontsize=12, fontweight='bold'); axes[i, 2].axis('off')
    plt.suptitle('Model Predictions vs Ground Truth', fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=150, bbox_inches='tight')
    plt.show(); plt.close()

    # Legend
    fig, ax = plt.subplots(figsize=(6, 2.5)); ax.axis('off')
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS_HEX)):
        ax.add_patch(plt.Rectangle((0.05, 0.85 - i*0.15), 0.08, 0.1, facecolor=color, edgecolor='black', linewidth=0.5))
        ax.text(0.16, 0.90 - i*0.15, f'{i}: {name}', fontsize=11, va='center')
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 1); ax.set_title('Class Legend', fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'class_legend.png'), dpi=150, bbox_inches='tight')
    plt.show(); plt.close()


def print_and_save_results(per_class, summary, output_dir):
    print("\n" + "=" * 85)
    print("  EVALUATION RESULTS — Residual-Attention-U-Net")
    print("=" * 85)
    header = f"{'Class':<14} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'Support':>10}"
    print(header); print("-" * 85)
    for name in CLASS_NAMES:
        m = per_class[name]
        print(f"{name:<14} {m['iou']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['accuracy']:>10.4f} {m['support']:>10,}")
    print("-" * 85)
    print(f"\n  Overall Pixel Accuracy:     {summary['overall_accuracy']:.4f}")
    print(f"  Mean IoU (all 6 classes):   {summary['mean_iou_all']:.4f}")
    print(f"  Mean IoU (5 labeled only):  {summary['mean_iou_labeled']:.4f}")
    print(f"  Weighted IoU:               {summary['weighted_iou']:.4f}")
    print(f"  Mean F1-Score:              {summary['mean_f1']:.4f}")
    print("=" * 85)

    # Save CSV
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Class', 'Class_ID', 'IoU', 'Precision', 'Recall', 'F1', 'Accuracy', 'Support'])
        for name in CLASS_NAMES:
            m = per_class[name]
            w.writerow([name, m['class_id'], f"{m['iou']:.6f}", f"{m['precision']:.6f}",
                       f"{m['recall']:.6f}", f"{m['f1']:.6f}", f"{m['accuracy']:.6f}", m['support']])
        w.writerow([])
        w.writerow(['Summary Metric', 'Value'])
        for k, v in summary.items():
            w.writerow([k, f"{v:.6f}"])
    print(f"\n📄 CSV saved to: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🖥️  GPU: {gpus}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find paths
    data_path = find_path(DATASET_PATHS, "dataset")
    model_path = find_path(MODEL_PATHS, "model")

    # 1. Load test data
    print("\n📦 [1/5] Loading dataset...")
    x_test, y_test, n_classes = prepare_test_set(data_path, PATCH_SIZE, PATCH_STEP)
    print(f"  Test set: {len(x_test)} samples, {n_classes} classes")

    # 2. Load model
    print("\n🧠 [2/5] Loading model...")
    import segmentation_models as sm
    dummy_weights = np.ones(n_classes, dtype=np.float32) / n_classes
    custom_objects = {
        'masked_total_loss': get_masked_loss(dummy_weights),
        'iou_score': sm.metrics.IOUScore(per_image=False),
        'f1-score': sm.metrics.FScore(beta=1),
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"  ✅ Loaded: {model.name} ({model.count_params():,} parameters)")

    # 3. Compute metrics
    print("\n📊 [3/5] Computing metrics...")
    cm = compute_confusion_matrix(model, x_test, y_test, BATCH_SIZE)

    per_class, summary = compute_metrics(cm)
    print_and_save_results(per_class, summary, OUTPUT_DIR)

    # 4. Visualizations
    print("\n🎨 [4/5] Generating visualizations...")
    plot_confusion_matrix(cm, OUTPUT_DIR)
    plot_per_class_iou(per_class, summary, OUTPUT_DIR)
    plot_all_metrics(per_class, OUTPUT_DIR)

    # 5. Prediction visuals
    print("\n🖼️  [5/5] Generating prediction samples...")
    visualize_predictions(model, x_test, y_test, OUTPUT_DIR, NUM_VIS_SAMPLES, SEED)

    # Save confusion matrix
    np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), cm)

    print(f"\n✅ All done! Results saved to: {OUTPUT_DIR}/")
