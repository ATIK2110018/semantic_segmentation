import os

import argparse
import csv

import keras

if not hasattr(keras.utils, "generic_utils"):
    keras.utils.generic_utils = (
        keras.utils if hasattr(keras.utils, "get_custom_objects") else keras.saving
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src.dataset import prepare_dataset
from src.model import build_residual_attention_unet
from src.utils import calculate_class_weights, display_image, get_masked_loss, plot_history


def make_cmap(class_colors):
    return ListedColormap(class_colors)


def full_evaluation(
    model,
    x_test,
    y_test,
    output_dir,
    class_names,
    class_colors,
    batch_size=16,
    num_samples=8,
):
    """Run complete evaluation: metrics, confusion matrix, visualizations, CSV."""
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(class_names)
    custom_cmap = make_cmap(class_colors)

    print("\nComputing confusion matrix...")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_batches = (len(x_test) + batch_size - 1) // batch_size
    for i in range(0, len(x_test), batch_size):
        print(f"  Batch {i // batch_size + 1}/{total_batches}", end="\r")
        x_b = x_test[i : i + batch_size]
        y_b = y_test[i : i + batch_size]
        y_pred = model.predict(x_b, verbose=0)
        cm += confusion_matrix(
            np.argmax(y_b, axis=-1).flatten(),
            np.argmax(y_pred, axis=-1).flatten(),
            labels=np.arange(num_classes),
        )
    print(f"  Done - {len(x_test)} samples.")

    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        per_class[name] = {
            "class_id": i,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "support": int(cm[i, :].sum()),
        }

    all_ious = [per_class[n]["iou"] for n in class_names]
    supported_names = [n for n in class_names if per_class[n]["support"] > 0]
    supported_ious = [per_class[n]["iou"] for n in supported_names]
    total_px = sum(per_class[n]["support"] for n in class_names)
    summary = {
        "overall_accuracy": np.trace(cm) / (cm.sum() + 1e-8),
        "mean_iou_all": np.mean(all_ious),
        "mean_iou_supported": np.mean(supported_ious) if supported_ious else 0.0,
        "weighted_iou": sum(
            per_class[n]["iou"] * per_class[n]["support"] for n in class_names
        )
        / (total_px + 1e-8),
        "mean_f1": np.mean([per_class[n]["f1"] for n in class_names]),
    }

    print("\n" + "=" * 95)
    print(f"  EVALUATION RESULTS - {model.name}")
    print("=" * 95)
    print(
        f"{'Class':<22} {'IoU':>8} {'Precision':>10} {'Recall':>8} "
        f"{'F1':>8} {'Accuracy':>10} {'Support':>10}"
    )
    print("-" * 95)
    for name in class_names:
        m = per_class[name]
        print(
            f"{name:<22} {m['iou']:>8.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} "
            f"{m['accuracy']:>10.4f} {m['support']:>10,}"
        )
    print("-" * 95)
    print(f"Overall Pixel Accuracy:  {summary['overall_accuracy']:.4f}")
    print(f"Mean IoU (all classes):  {summary['mean_iou_all']:.4f}")
    print(f"Mean IoU (supported):    {summary['mean_iou_supported']:.4f}")
    print(f"Weighted IoU:            {summary['weighted_iou']:.4f}")
    print(f"Mean F1-Score:           {summary['mean_f1']:.4f}")
    print("=" * 95)

    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Class", "Class_ID", "IoU", "Precision", "Recall", "F1", "Accuracy", "Support"]
        )
        for name in class_names:
            m = per_class[name]
            writer.writerow(
                [
                    name,
                    m["class_id"],
                    f"{m['iou']:.6f}",
                    f"{m['precision']:.6f}",
                    f"{m['recall']:.6f}",
                    f"{m['f1']:.6f}",
                    f"{m['accuracy']:.6f}",
                    m["support"],
                ]
            )
        writer.writerow([])
        writer.writerow(["Summary Metric", "Value"])
        for key, value in summary.items():
            writer.writerow([key, f"{value:.6f}"])
    print(f"Saved: {csv_path}")

    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        linewidths=0.5,
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        linewidths=0.5,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    confusion_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {confusion_path}")

    ious = [per_class[n]["iou"] for n in class_names]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(class_names, ious, color=class_colors, edgecolor="black", linewidth=0.8)
    for bar, iou in zip(bars, ious):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{iou:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.axhline(
        y=summary["mean_iou_all"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean IoU (all): {summary['mean_iou_all']:.4f}",
    )
    ax.axhline(
        y=summary["mean_iou_supported"],
        color="green",
        linestyle="-.",
        linewidth=2,
        label=f"Mean IoU (supported): {summary['mean_iou_supported']:.4f}",
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("IoU Score", fontsize=13)
    ax.set_title("Per-Class IoU Scores", fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    iou_path = os.path.join(output_dir, "per_class_iou.png")
    plt.savefig(iou_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {iou_path}")

    x = np.arange(len(class_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for j, (key, label, color) in enumerate(
        zip(["iou", "precision", "recall", "f1"], ["IoU", "Precision", "Recall", "F1"], colors)
    ):
        vals = [per_class[n][key] for n in class_names]
        ax.bar(x + j * width, vals, width, label=label, color=color, alpha=0.85)
    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Per-Class Evaluation Metrics", fontsize=15, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(class_names, fontsize=10, rotation=35)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, "all_metrics_chart.png")
    plt.savefig(metrics_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {metrics_path}")

    sample_count = min(num_samples, len(x_test))
    np.random.seed(42)
    indices = np.random.choice(len(x_test), sample_count, replace=False)
    fig, axes = plt.subplots(sample_count, 3, figsize=(15, 4.5 * sample_count))
    if sample_count == 1:
        axes = axes[np.newaxis, :]
    for i, idx in enumerate(indices):
        test_img = x_test[idx]
        true_mask = np.argmax(y_test[idx], axis=-1)
        pred_mask = np.argmax(
            model.predict(np.expand_dims(test_img, 0), verbose=0)[0], axis=-1
        )
        axes[i, 0].imshow(display_image(test_img))
        axes[i, 0].set_title(f"Input Image (#{idx})", fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(
            true_mask, cmap=custom_cmap, vmin=0, vmax=num_classes - 1, interpolation="nearest"
        )
        axes[i, 1].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(
            pred_mask, cmap=custom_cmap, vmin=0, vmax=num_classes - 1, interpolation="nearest"
        )
        axes[i, 2].set_title("Prediction", fontsize=12, fontweight="bold")
        axes[i, 2].axis("off")
    plt.suptitle(f"Predictions - {model.name}", fontsize=16, fontweight="bold", y=1.001)
    plt.tight_layout()
    predictions_path = os.path.join(output_dir, "predictions.png")
    plt.savefig(predictions_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {predictions_path}")

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * num_classes)))
    ax.axis("off")
    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        y = 0.95 - i * (0.9 / max(1, num_classes))
        ax.add_patch(
            plt.Rectangle((0.05, y - 0.03), 0.08, 0.05, facecolor=color, edgecolor="black")
        )
        ax.text(0.16, y, f"{i}: {name}", fontsize=10, va="center")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1)
    ax.set_title("Class Legend", fontsize=13, fontweight="bold")
    legend_path = os.path.join(output_dir, "class_legend.png")
    plt.savefig(legend_path, dpi=150, bbox_inches="tight")
    plt.close()

    np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)
    print(f"\nAll evaluation outputs saved to: {output_dir}/")
    return per_class, summary


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")

    print("Loading and preparing dataset...")
    x_train, x_test, y_train, y_test, n_classes, metadata = prepare_dataset(
        args.data_path,
        patch_size=args.patch_size,
        step=args.patch_step,
        test_size=args.test_size,
        valid_pixel_threshold=args.valid_pixel_threshold,
        image_tif=args.image_tif,
        mask_tif=args.mask_tif,
        return_metadata=True,
    )
    class_names = metadata["class_names"][:n_classes]
    class_colors = metadata["class_colors"][:n_classes]
    print(f"Dataset source: {metadata['source']}")
    if metadata["source"] == "geotiff":
        print(f"Image: {metadata['image_path']}")
        print(f"Mask:  {metadata['mask_path']}")
    print(f"Dataset prepared: {len(x_train)} training samples, {len(x_test)} testing samples.")
    print(f"Input shape: {x_train.shape[1:]}; classes: {n_classes}")

    use_attention = not args.disable_attention
    use_residual = not args.disable_residual
    img_height, img_width, img_channels = x_train.shape[1:]

    y_train_labels = np.argmax(y_train, axis=-1)
    class_weights = calculate_class_weights(
        y_train_labels,
        num_classes=n_classes,
        ignore_label=metadata["ignore_label"],
    )
    print(f"Calculated class weights: {class_weights}")

    model = build_residual_attention_unet(
        n_classes,
        img_height,
        img_width,
        img_channels,
        use_attention=use_attention,
        use_residual=use_residual,
    )
    print(f"Building model: {model.name}...")

    loss_fn = get_masked_loss(class_weights, ignore_label=metadata["ignore_label"])
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    print("Starting training...")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=15,
            verbose=1,
            min_lr=1e-7,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
    )

    model.save(args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

    plot_history(history, save_path=os.path.join(args.output_dir, "training_history.png"))

    print("\n" + "=" * 60)
    print("  STARTING FULL EVALUATION")
    print("=" * 60)
    full_evaluation(
        model,
        x_test,
        y_test,
        args.output_dir,
        class_names,
        class_colors,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual Attention U-Net for Semantic Segmentation")
    parser.add_argument("--data_path", type=str, default="dataset", help="Path to dataset")
    parser.add_argument(
        "--image_tif",
        type=str,
        default=None,
        help="Optional path to Sentinel-2 image GeoTIFF (.tif/.tiff)",
    )
    parser.add_argument(
        "--mask_tif",
        type=str,
        default=None,
        help="Optional path to ESRI LULC mask GeoTIFF (.tif/.tiff)",
    )
    parser.add_argument("--patch_size", type=int, default=256, help="Size of patches")
    parser.add_argument("--patch_step", type=int, default=256, help="Step size for patching")
    parser.add_argument("--test_size", type=float, default=0.25, help="Validation/test split fraction")
    parser.add_argument(
        "--valid_pixel_threshold",
        type=float,
        default=0.5,
        help="Minimum finite Sentinel-2 pixel fraction required to keep a patch",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--model_save_path", type=str, default="model.keras", help="Path to save the model")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for evaluation outputs")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of prediction samples to visualize")
    parser.add_argument("--disable_attention", action="store_true", help="Disable attention gates")
    parser.add_argument("--disable_residual", action="store_true", help="Disable residual connections")

    main(parser.parse_args())
