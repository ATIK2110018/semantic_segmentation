"""Boundary-aware evaluation metrics for semantic segmentation."""

import numpy as np
import cv2


def _extract_boundary_map(label_map, kernel_size=3):
    """Extract boundary pixels via morphological gradient."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    label_u8 = label_map.astype(np.uint8)
    dilated = cv2.dilate(label_u8, kernel)
    eroded = cv2.erode(label_u8, kernel)
    gradient = (dilated.astype(np.int32) - eroded.astype(np.int32)) != 0
    return gradient.astype(np.bool_)


def _boundary_confusion(true_boundary, pred_boundary):
    """Return TP, FP, FN counts for binary boundary maps."""
    tp = int(np.logical_and(true_boundary, pred_boundary).sum())
    fp = int(np.logical_and(~true_boundary, pred_boundary).sum())
    fn = int(np.logical_and(true_boundary, ~pred_boundary).sum())
    return tp, fp, fn


def compute_boundary_metrics(y_true_labels, y_pred_labels, kernel_size=3, batch_size=64):
    """Compute global BF-score and Boundary IoU over the dataset."""
    assert y_true_labels.shape == y_pred_labels.shape

    total_tp = total_fp = total_fn = total_boundary_px = total_px = 0
    n = len(y_true_labels)

    for i in range(n):
        if i % batch_size == 0:
            print(f"  Boundary extraction: sample {i + 1}/{n}", end="\r")

        true_bnd = _extract_boundary_map(y_true_labels[i], kernel_size)
        pred_bnd = _extract_boundary_map(y_pred_labels[i], kernel_size)

        tp, fp, fn = _boundary_confusion(true_bnd, pred_bnd)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_boundary_px += int(true_bnd.sum())
        total_px += true_bnd.size

    print(f"  Boundary extraction: {n}/{n} — done.          ")

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    bf_score = 2 * precision * recall / (precision + recall + 1e-8)
    b_iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)

    return {
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "bf_score": float(bf_score),
        "boundary_iou": float(b_iou),
        "boundary_pixel_frac": float(total_boundary_px / (total_px + 1e-8)),
    }


def compute_boundary_metrics_per_class(
    y_true_labels, y_pred_labels, num_classes, class_names=None, kernel_size=3
):
    """Compute per-class boundary precision, recall, BF-score, and IoU."""
    class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    results = {}

    for c, name in enumerate(class_names):
        tp = fp = fn = support = 0
        for i in range(len(y_true_labels)):
            true_c = (y_true_labels[i] == c).astype(np.uint8)
            pred_c = (y_pred_labels[i] == c).astype(np.uint8)

            true_bnd = (
                cv2.dilate(true_c, kernel).astype(np.int32)
                - cv2.erode(true_c, kernel).astype(np.int32)
            ) != 0
            pred_bnd = (
                cv2.dilate(pred_c, kernel).astype(np.int32)
                - cv2.erode(pred_c, kernel).astype(np.int32)
            ) != 0

            _tp, _fp, _fn = _boundary_confusion(true_bnd, pred_bnd)
            tp += _tp
            fp += _fp
            fn += _fn
            support += int(true_bnd.sum())

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        results[name] = {
            "precision": float(prec),
            "recall": float(rec),
            "bf_score": float(f1),
            "boundary_iou": float(iou),
            "support": support,
        }

    return results


def visualize_boundary_predictions(
    images,
    y_true_labels,
    y_pred_labels,
    indices=None,
    num_samples=6,
    kernel_size=3,
    save_path="boundary_predictions.png",
    display_fn=None,
):
    """Save boundary visualization with color-coded correct/wrong predictions.

    Colors: Green=correct(TP), Red=missed(FN), Yellow=false(FP).
    Layout: [Input | GT Boundary | Pred Boundary (color-coded) | Error Map]
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if indices is None:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(images), min(num_samples, len(images)), replace=False).tolist()

    COLOR_TP = np.array([0, 255, 0], dtype=np.uint8)
    COLOR_FN = np.array([255, 50, 50], dtype=np.uint8)
    COLOR_FP = np.array([255, 255, 0], dtype=np.uint8)
    COLOR_GT = np.array([0, 255, 255], dtype=np.uint8)

    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(20, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img = images[idx]
        if display_fn is not None:
            rgb = (np.clip(display_fn(img), 0, 1) * 255).astype(np.uint8)
        else:
            ch = min(img.shape[-1], 3)
            rgb = (np.clip(img[..., :ch], 0, 1) * 255).astype(np.uint8)
            if ch < 3:
                rgb = np.repeat(rgb, 3, axis=-1)

        true_bnd = _extract_boundary_map(y_true_labels[idx], kernel_size)
        pred_bnd = _extract_boundary_map(y_pred_labels[idx], kernel_size)

        tp_mask = np.logical_and(true_bnd, pred_bnd)
        fn_mask = np.logical_and(true_bnd, ~pred_bnd)
        fp_mask = np.logical_and(~true_bnd, pred_bnd)

        # GT boundary overlay (cyan)
        gt_overlay = rgb.copy()
        gt_overlay[true_bnd] = COLOR_GT

        # Pred boundary overlay (color-coded)
        pred_overlay = rgb.copy()
        pred_overlay[tp_mask] = COLOR_TP
        pred_overlay[fn_mask] = COLOR_FN
        pred_overlay[fp_mask] = COLOR_FP

        # Error map (dark background, errors only)
        error_map = np.zeros_like(rgb)
        error_map[tp_mask] = COLOR_TP
        error_map[fn_mask] = COLOR_FN
        error_map[fp_mask] = COLOR_FP

        col_data = [
            (f"Input (#{idx})", rgb),
            ("GT Boundary", gt_overlay),
            ("Pred Boundary", pred_overlay),
            ("Error Map", error_map),
        ]
        for col, (title, data) in enumerate(col_data):
            axes[row, col].imshow(data)
            axes[row, col].set_title(title, fontsize=11, fontweight="bold")
            axes[row, col].axis("off")

    legend_patches = [
        mpatches.Patch(color=(0, 1, 0), label="Correct (TP)"),
        mpatches.Patch(color=(1, 50 / 255, 50 / 255), label="Missed (FN)"),
        mpatches.Patch(color=(1, 1, 0), label="False (FP)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=11, frameon=True)
    plt.suptitle("Boundary Region Predictions", fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_boundary_metrics_chart(per_class_boundary, class_colors, save_path="boundary_metrics_chart.png"):
    """Save grouped bar chart of per-class boundary metrics."""
    import matplotlib.pyplot as plt

    class_names = list(per_class_boundary.keys())
    n = len(class_names)
    x = np.arange(n)
    width = 0.22

    metrics_keys = ["precision", "recall", "bf_score", "boundary_iou"]
    labels = ["Precision", "Recall", "BF-score", "Boundary IoU"]
    bar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(max(12, n * 1.4), 7))
    for j, (key, label, color) in enumerate(zip(metrics_keys, labels, bar_colors)):
        vals = [per_class_boundary[name][key] for name in class_names]
        bars = ax.bar(x + j * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=45,
            )

    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Per-Class Boundary Evaluation Metrics", fontsize=15, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(class_names, fontsize=10, rotation=35, ha="right")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
