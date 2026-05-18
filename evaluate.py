"""
Comprehensive Evaluation Script for Residual-Attention-U-Net
============================================================
Loads a saved model, runs it on the test set, and generates:
  1. Per-class metrics (IoU, Precision, Recall, F1, Accuracy)
  2. Mean IoU (mIoU) and overall accuracy
  3. Confusion matrix heatmap
  4. Prediction visualizations (image | ground truth | prediction)
  5. Per-class IoU bar chart
  6. Summary table printed to console and saved as CSV

Usage:
  python evaluate.py --model_path models/residual_attention_unet.keras
  python evaluate.py --model_path models/residual_attention_unet.keras --output_dir results/full_model
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix
import csv

from src.dataset import prepare_dataset
from src.utils import get_masked_loss, calculate_class_weights

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Unlabeled", "Building", "Land", "Road", "Vegetation", "Water"]
NUM_CLASSES = 6

# Custom colormap matching the dataset's RGB labels
CUSTOM_CMAP = ListedColormap([
    (155/255, 155/255, 155/255),  # 0 - Unlabeled (gray)
    (60/255,  16/255, 152/255),   # 1 - Building (dark purple)
    (132/255, 41/255, 246/255),   # 2 - Land (purple)
    (110/255, 193/255, 228/255),  # 3 - Road (light blue)
    (254/255, 221/255, 58/255),   # 4 - Vegetation (yellow)
    (226/255, 169/255, 41/255),   # 5 - Water (orange)
])

CLASS_COLORS_HEX = ['#9B9B9B', '#3C1098', '#8429F6', '#6EC1E4', '#FEDD3A', '#E2A929']


# ─── Metrics Computation ─────────────────────────────────────────────────────

def compute_confusion_matrix(model, x_test, y_test, batch_size=16):
    """Compute confusion matrix by predicting in batches."""
    print("Computing predictions on test set...")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    
    total_batches = (len(x_test) + batch_size - 1) // batch_size
    for i in range(0, len(x_test), batch_size):
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches}...", end='\r')
        
        x_batch = x_test[i:i+batch_size]
        y_batch = y_test[i:i+batch_size]
        
        y_pred_batch = model.predict(x_batch, verbose=0)
        
        y_true_flat = np.argmax(y_batch, axis=-1).flatten()
        y_pred_flat = np.argmax(y_pred_batch, axis=-1).flatten()
        
        cm += confusion_matrix(y_true_flat, y_pred_flat, labels=np.arange(NUM_CLASSES))
    
    print(f"  Done — processed {len(x_test)} samples.")
    return cm


def compute_per_class_metrics(cm):
    """Compute IoU, Precision, Recall, F1, and Accuracy per class from confusion matrix."""
    metrics = {}
    
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
        
        metrics[CLASS_NAMES[i]] = {
            'class_id': i,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': int(cm[i, :].sum())  # total pixels for this class
        }
    
    return metrics


def compute_summary_metrics(cm, per_class):
    """Compute overall summary metrics."""
    # Overall pixel accuracy
    overall_accuracy = np.trace(cm) / (cm.sum() + 1e-8)
    
    # Mean IoU (all classes)
    all_ious = [per_class[name]['iou'] for name in CLASS_NAMES]
    mean_iou_all = np.mean(all_ious)
    
    # Mean IoU (excluding Unlabeled — more meaningful)
    labeled_ious = [per_class[name]['iou'] for name in CLASS_NAMES if name != "Unlabeled"]
    mean_iou_labeled = np.mean(labeled_ious)
    
    # Weighted IoU (weighted by class support)
    total_pixels = sum(per_class[name]['support'] for name in CLASS_NAMES)
    weighted_iou = sum(
        per_class[name]['iou'] * per_class[name]['support'] 
        for name in CLASS_NAMES
    ) / (total_pixels + 1e-8)
    
    # Mean F1
    mean_f1 = np.mean([per_class[name]['f1'] for name in CLASS_NAMES])
    
    return {
        'overall_accuracy': overall_accuracy,
        'mean_iou_all': mean_iou_all,
        'mean_iou_labeled': mean_iou_labeled,
        'weighted_iou': weighted_iou,
        'mean_f1': mean_f1,
    }


# ─── Visualization Functions ─────────────────────────────────────────────────

def plot_confusion_matrix(cm, output_dir):
    """Plot and save a normalized confusion matrix heatmap."""
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[0], linewidths=0.5)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Normalized (percentage)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_iou(per_class, summary, output_dir):
    """Plot a bar chart of per-class IoU with mean IoU line."""
    classes = CLASS_NAMES
    ious = [per_class[name]['iou'] for name in classes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, ious, color=CLASS_COLORS_HEX, edgecolor='black', linewidth=0.8)
    
    # Add value labels on bars
    for bar, iou in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{iou:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Mean IoU lines
    ax.axhline(y=summary['mean_iou_all'], color='red', linestyle='--', linewidth=2,
               label=f"Mean IoU (all): {summary['mean_iou_all']:.4f}")
    ax.axhline(y=summary['mean_iou_labeled'], color='green', linestyle='-.', linewidth=2,
               label=f"Mean IoU (labeled): {summary['mean_iou_labeled']:.4f}")
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('IoU Score', fontsize=13)
    ax.set_title('Per-Class IoU Scores', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'per_class_iou.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_metrics_chart(per_class, output_dir):
    """Plot grouped bar chart showing IoU, Precision, Recall, F1 per class."""
    classes = CLASS_NAMES
    metrics_keys = ['iou', 'precision', 'recall', 'f1']
    metric_labels = ['IoU', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(classes))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    for j, (key, label, color) in enumerate(zip(metrics_keys, metric_labels, colors)):
        values = [per_class[name][key] for name in classes]
        bars = ax.bar(x + j * width, values, width, label=label, color=color, alpha=0.85)
    
    ax.set_xlabel('Class', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Per-Class Evaluation Metrics', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'all_metrics_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_predictions(model, x_test, y_test, output_dir, num_samples=8, seed=42):
    """Generate prediction comparison grids: Image | Ground Truth | Prediction."""
    np.random.seed(seed)
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4.5 * num_samples))
    
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i, idx in enumerate(indices):
        test_img = x_test[idx]
        true_mask = np.argmax(y_test[idx], axis=-1)
        pred_mask = np.argmax(model.predict(np.expand_dims(test_img, 0), verbose=0)[0], axis=-1)
        
        # Original Image
        axes[i, 0].imshow(test_img)
        axes[i, 0].set_title(f'Input Image (#{idx})', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Ground Truth
        axes[i, 1].imshow(true_mask, cmap=CUSTOM_CMAP, vmin=0, vmax=5, interpolation='nearest')
        axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(pred_mask, cmap=CUSTOM_CMAP, vmin=0, vmax=5, interpolation='nearest')
        axes[i, 2].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.suptitle('Model Predictions vs Ground Truth', fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Also generate a color legend
    _save_legend(output_dir)


def _save_legend(output_dir):
    """Save a standalone class color legend."""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')
    
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS_HEX)):
        ax.add_patch(plt.Rectangle((0.05, 0.85 - i * 0.15), 0.08, 0.1, 
                                    facecolor=color, edgecolor='black', linewidth=0.5))
        ax.text(0.16, 0.90 - i * 0.15, f'{i}: {name}', fontsize=11, va='center')
    
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.set_title('Class Legend', fontsize=13, fontweight='bold')
    
    save_path = os.path.join(output_dir, 'class_legend.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_results(per_class, summary):
    """Print a formatted results table to console."""
    print("\n" + "=" * 80)
    print("  EVALUATION RESULTS — Residual-Attention-U-Net")
    print("=" * 80)
    
    header = f"{'Class':<14} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'Support':>10}"
    print(header)
    print("-" * 80)
    
    for name in CLASS_NAMES:
        m = per_class[name]
        print(f"{name:<14} {m['iou']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['accuracy']:>10.4f} {m['support']:>10,}")
    
    print("-" * 80)
    print(f"\n  Overall Pixel Accuracy:     {summary['overall_accuracy']:.4f}")
    print(f"  Mean IoU (all 6 classes):   {summary['mean_iou_all']:.4f}")
    print(f"  Mean IoU (5 labeled only):  {summary['mean_iou_labeled']:.4f}")
    print(f"  Weighted IoU:               {summary['weighted_iou']:.4f}")
    print(f"  Mean F1-Score:              {summary['mean_f1']:.4f}")
    print("=" * 80 + "\n")


def save_results_csv(per_class, summary, output_dir):
    """Save results as a CSV file."""
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Per-class metrics
        writer.writerow(['Class', 'Class_ID', 'IoU', 'Precision', 'Recall', 'F1', 'Accuracy', 'Support'])
        for name in CLASS_NAMES:
            m = per_class[name]
            writer.writerow([name, m['class_id'], f"{m['iou']:.6f}", f"{m['precision']:.6f}",
                           f"{m['recall']:.6f}", f"{m['f1']:.6f}", f"{m['accuracy']:.6f}", m['support']])
        
        writer.writerow([])
        
        # Summary metrics
        writer.writerow(['Summary Metric', 'Value'])
        writer.writerow(['Overall Pixel Accuracy', f"{summary['overall_accuracy']:.6f}"])
        writer.writerow(['Mean IoU (all classes)', f"{summary['mean_iou_all']:.6f}"])
        writer.writerow(['Mean IoU (labeled only)', f"{summary['mean_iou_labeled']:.6f}"])
        writer.writerow(['Weighted IoU', f"{summary['weighted_iou']:.6f}"])
        writer.writerow(['Mean F1-Score', f"{summary['mean_f1']:.6f}"])
    
    print(f"  Saved: {csv_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    else:
        print("No GPU found, using CPU.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # 1. Load dataset (same split as training — random_state=42)
    print("\n[1/5] Loading dataset...")
    _, x_test, _, y_test, n_classes = prepare_dataset(
        args.data_path,
        patch_size=args.patch_size,
        step=args.patch_step
    )
    print(f"  Test set: {len(x_test)} samples, {n_classes} classes")
    
    # 2. Load model
    print("\n[2/5] Loading model...")
    
    # Register custom loss for loading
    dummy_weights = np.ones(n_classes, dtype=np.float32) / n_classes
    custom_objects = {
        'masked_total_loss': get_masked_loss(dummy_weights),
        'iou_score': tf.keras.metrics.MeanIoU(num_classes=n_classes),
        'f1-score': tf.keras.metrics.F1Score,
    }
    
    try:
        import segmentation_models as sm
        custom_objects['iou_score'] = sm.metrics.IOUScore(per_image=False)
        custom_objects['f1-score'] = sm.metrics.FScore(beta=1)
    except ImportError:
        pass
    
    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
    print(f"  Model loaded: {model.name}")
    print(f"  Parameters: {model.count_params():,}")
    
    # 3. Compute metrics
    print("\n[3/5] Computing metrics...")
    cm = compute_confusion_matrix(model, x_test, y_test, batch_size=args.batch_size)
    per_class = compute_per_class_metrics(cm)
    summary = compute_summary_metrics(cm, per_class)
    
    # Print results
    print_results(per_class, summary)
    
    # 4. Generate visualizations
    print("[4/5] Generating visualizations...")
    plot_confusion_matrix(cm, args.output_dir)
    plot_per_class_iou(per_class, summary, args.output_dir)
    plot_all_metrics_chart(per_class, args.output_dir)
    visualize_predictions(model, x_test, y_test, args.output_dir, 
                         num_samples=args.num_samples, seed=args.seed)
    
    # 5. Save CSV
    print("\n[5/5] Saving results...")
    save_results_csv(per_class, summary, args.output_dir)
    
    # Save confusion matrix as numpy
    np.save(os.path.join(args.output_dir, 'confusion_matrix.npy'), cm)
    print(f"  Saved: {os.path.join(args.output_dir, 'confusion_matrix.npy')}")
    
    print(f"\n✅ Evaluation complete! All outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model")
    parser.add_argument("--model_path", type=str, default="models/residual_attention_unet.keras",
                        help="Path to the saved .keras model")
    parser.add_argument("--data_path", type=str, default="Semantic segmentation dataset",
                        help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="results/full_model",
                        help="Directory to save evaluation outputs")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size")
    parser.add_argument("--patch_step", type=int, default=160, help="Patch step")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of prediction samples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    
    args = parser.parse_args()
    main(args)
