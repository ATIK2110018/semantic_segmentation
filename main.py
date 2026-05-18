import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

# Keras 3 compatibility patch for segmentation_models
import keras
if not hasattr(keras.utils, 'generic_utils'):
    keras.utils.generic_utils = keras.utils if hasattr(keras.utils, 'get_custom_objects') else keras.saving

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import segmentation_models as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix
import csv

from src.model import build_residual_attention_unet
from src.dataset import prepare_dataset
from src.utils import get_masked_loss, calculate_class_weights, plot_history, visualize_prediction

# ─── Evaluation Constants ─────────────────────────────────────────────────────
CLASS_NAMES = ["Unlabeled", "Building", "Land", "Road", "Vegetation", "Water"]
NUM_CLASSES = 6
CLASS_COLORS_HEX = ['#9B9B9B', '#3C1098', '#8429F6', '#6EC1E4', '#FEDD3A', '#E2A929']
CUSTOM_CMAP = ListedColormap([
    (155/255, 155/255, 155/255), (60/255, 16/255, 152/255),
    (132/255, 41/255, 246/255),  (110/255, 193/255, 228/255),
    (254/255, 221/255, 58/255),  (226/255, 169/255, 41/255),
])


# ─── Evaluation Functions ─────────────────────────────────────────────────────

def full_evaluation(model, x_test, y_test, output_dir, batch_size=16, num_samples=8):
    """Run complete evaluation: metrics, confusion matrix, visualizations, CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Compute confusion matrix
    print("\n📊 Computing confusion matrix...")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_batches = (len(x_test) + batch_size - 1) // batch_size
    for i in range(0, len(x_test), batch_size):
        print(f"  Batch {i // batch_size + 1}/{total_batches}", end='\r')
        x_b = x_test[i:i+batch_size]
        y_b = y_test[i:i+batch_size]
        y_pred = model.predict(x_b, verbose=0)
        cm += confusion_matrix(
            np.argmax(y_b, axis=-1).flatten(),
            np.argmax(y_pred, axis=-1).flatten(),
            labels=np.arange(NUM_CLASSES)
        )
    print(f"  Done — {len(x_test)} samples.")
    
    # 2. Per-class metrics
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
    
    # 3. Summary metrics
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
    
    # 4. Print results
    print("\n" + "=" * 85)
    print(f"  EVALUATION RESULTS — {model.name}")
    print("=" * 85)
    print(f"{'Class':<14} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'Support':>10}")
    print("-" * 85)
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
    
    # 5. Save CSV
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
    print(f"📄 Saved: {csv_path}")
    
    # 6. Confusion matrix heatmap
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
    print(f"🖼️  Saved: {os.path.join(output_dir, 'confusion_matrix.png')}")
    
    # 7. Per-class IoU bar chart
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
    print(f"🖼️  Saved: {os.path.join(output_dir, 'per_class_iou.png')}")
    
    # 8. All metrics grouped bar chart
    x = np.arange(len(CLASS_NAMES)); width = 0.2
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
    print(f"🖼️  Saved: {os.path.join(output_dir, 'all_metrics_chart.png')}")
    
    # 9. Prediction visualizations
    np.random.seed(42)
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4.5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    for i, idx in enumerate(indices):
        test_img = x_test[idx]
        true_mask = np.argmax(y_test[idx], axis=-1)
        pred_mask = np.argmax(model.predict(np.expand_dims(test_img, 0), verbose=0)[0], axis=-1)
        axes[i, 0].imshow(test_img)
        axes[i, 0].set_title(f'Input Image (#{idx})', fontsize=12, fontweight='bold'); axes[i, 0].axis('off')
        axes[i, 1].imshow(true_mask, cmap=CUSTOM_CMAP, vmin=0, vmax=5, interpolation='nearest')
        axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold'); axes[i, 1].axis('off')
        axes[i, 2].imshow(pred_mask, cmap=CUSTOM_CMAP, vmin=0, vmax=5, interpolation='nearest')
        axes[i, 2].set_title('Prediction', fontsize=12, fontweight='bold'); axes[i, 2].axis('off')
    plt.suptitle(f'Predictions — {model.name}', fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"🖼️  Saved: {os.path.join(output_dir, 'predictions.png')}")
    
    # 10. Color legend
    fig, ax = plt.subplots(figsize=(6, 2.5)); ax.axis('off')
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS_HEX)):
        ax.add_patch(plt.Rectangle((0.05, 0.85 - i*0.15), 0.08, 0.1, facecolor=color, edgecolor='black', linewidth=0.5))
        ax.text(0.16, 0.90 - i*0.15, f'{i}: {name}', fontsize=11, va='center')
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 1); ax.set_title('Class Legend', fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'class_legend.png'), dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    
    # Save raw confusion matrix
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)
    
    print(f"\n✅ All evaluation outputs saved to: {output_dir}/")
    return per_class, summary


# ─── Main Training + Evaluation Pipeline ──────────────────────────────────────

def main(args):
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")

    # 1. Prepare Dataset
    print("Loading and preparing dataset...")
    x_train, x_test, y_train, y_test, n_classes = prepare_dataset(
        args.data_path, 
        patch_size=args.patch_size, 
        step=args.patch_step
    )
    print(f"Dataset prepared: {len(x_train)} training samples, {len(x_test)} testing samples.")

    # 2. Build Model
    use_attention = not args.disable_attention
    use_residual = not args.disable_residual
    img_height, img_width, img_channels = x_train.shape[1:]
    
    # 3. Calculate Class Weights for Loss
    y_train_labels = np.argmax(y_train, axis=-1)
    class_weights = calculate_class_weights(y_train_labels)
    print(f"Calculated class weights: {class_weights}")

    model = build_residual_attention_unet(
        n_classes, img_height, img_width, img_channels,
        use_attention=use_attention,
        use_residual=use_residual
    )
    print(f"Building model: {model.name}...")

    # 4. Compile Model
    loss_fn = get_masked_loss(class_weights)
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=[
            sm.metrics.IOUScore(per_image=False),
            sm.metrics.FScore(beta=1),
            'accuracy'
        ]
    )

    # 5. Train Model
    print("Starting training...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1, min_lr=1e-7)
    ]
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True
    )

    # 6. Save Model
    model.save(args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

    # 7. Training History Plots
    plot_history(history, save_path=os.path.join(args.output_dir, 'training_history.png'))

    # 8. Full Evaluation (metrics, confusion matrix, visualizations)
    print("\n" + "=" * 60)
    print("  STARTING FULL EVALUATION")
    print("=" * 60)
    full_evaluation(model, x_test, y_test, args.output_dir, 
                    batch_size=args.batch_size, num_samples=args.num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual Attention U-Net for Semantic Segmentation")
    parser.add_argument("--data_path", type=str, default="Semantic segmentation dataset", help="Path to dataset")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of patches")
    parser.add_argument("--patch_step", type=int, default=160, help="Step size for patching")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--model_save_path", type=str, default="model.keras", help="Path to save the model")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for evaluation outputs")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of prediction samples to visualize")
    
    # Ablation Study Arguments
    parser.add_argument("--disable_attention", action="store_true", help="Disable attention gates for ablation study")
    parser.add_argument("--disable_residual", action="store_true", help="Disable residual connections for ablation study")
    
    args = parser.parse_args()
    main(args)

