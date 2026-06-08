"""
Boundary-Aware Evaluation Metrics for Semantic Segmentation
============================================================
Implements boundary region extraction via the morphological gradient operator
and computes Boundary F1-score (BF-score) and Boundary IoU to assess
segmentation quality near class interfaces.

Reference methodology:
    Boundary regions are delineated using the morphological gradient:
        B(x) = dilate(x, k) - erode(x, k)
    where k is a 3x3 structuring element. The resulting binary boundary
    maps are used to compute pixel-level precision, recall, F1, and IoU
    restricted to the boundary zone.
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Boundary extraction
# ---------------------------------------------------------------------------

def _extract_boundary_map(label_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Return a binary map of boundary pixels using the morphological gradient.

    The morphological gradient is defined as:
        grad(L) = dilate(L, k) - erode(L, k)

    Applied per-class: a pixel is a boundary pixel if it lies on the
    edge of *any* class region.

    Args:
        label_map: 2-D integer array of class labels (H x W).
        kernel_size: Side length of the square structuring element.

    Returns:
        boundary: Boolean array (H x W). True at boundary pixels.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    label_u8 = label_map.astype(np.uint8)

    # Morphological gradient on the full label map captures all class edges
    dilated = cv2.dilate(label_u8, kernel)
    eroded = cv2.erode(label_u8, kernel)
    gradient = (dilated.astype(np.int32) - eroded.astype(np.int32)) != 0
    return gradient.astype(np.bool_)


# ---------------------------------------------------------------------------
# Per-image boundary metric helpers
# ---------------------------------------------------------------------------

def _boundary_confusion(
    true_boundary: np.ndarray,
    pred_boundary: np.ndarray,
) -> tuple[int, int, int]:
    """Return TP, FP, FN counts for binary boundary maps."""
    tp = int(np.logical_and(true_boundary, pred_boundary).sum())
    fp = int(np.logical_and(~true_boundary, pred_boundary).sum())
    fn = int(np.logical_and(true_boundary, ~pred_boundary).sum())
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_boundary_metrics(
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    kernel_size: int = 3,
    batch_size: int = 64,
) -> dict:
    """Compute Boundary F1-score (BF-score) and Boundary IoU over a dataset.

    Boundary regions are extracted from *both* ground-truth and predicted
    label maps using the morphological gradient. Metrics are accumulated
    across all samples and reported globally.

    Args:
        y_true_labels: Integer label array of shape (N, H, W).
        y_pred_labels: Integer label array of shape (N, H, W).
        kernel_size: Structuring element size for morphological gradient.
        batch_size: Processing batch size (controls print frequency).

    Returns:
        dict with keys:
            boundary_precision  – TP / (TP + FP)
            boundary_recall     – TP / (TP + FN)
            bf_score            – harmonic mean of precision and recall
            boundary_iou        – TP / (TP + FP + FN)
            boundary_pixel_frac – fraction of pixels labelled as boundary
                                  in ground truth (dataset-level)
    """
    assert y_true_labels.shape == y_pred_labels.shape, (
        "y_true_labels and y_pred_labels must have the same shape."
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_boundary_px = 0
    total_px = 0
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
    recall    = total_tp / (total_tp + total_fn + 1e-8)
    bf_score  = 2 * precision * recall / (precision + recall + 1e-8)
    b_iou     = total_tp / (total_tp + total_fp + total_fn + 1e-8)

    return {
        "boundary_precision":  float(precision),
        "boundary_recall":     float(recall),
        "bf_score":            float(bf_score),
        "boundary_iou":        float(b_iou),
        "boundary_pixel_frac": float(total_boundary_px / (total_px + 1e-8)),
    }


def compute_boundary_metrics_per_class(
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
    kernel_size: int = 3,
) -> dict:
    """Compute per-class boundary metrics.

    For each class c, the boundary map is derived from the binary mask
    (label == c), so boundaries are class-specific (inner edges of each
    class region rather than all class transitions).

    Args:
        y_true_labels: Integer label array (N, H, W).
        y_pred_labels: Integer label array (N, H, W).
        num_classes: Total number of semantic classes.
        class_names: Optional list of class name strings.
        kernel_size: Structuring element size.

    Returns:
        dict mapping each class name to
        {'precision', 'recall', 'bf_score', 'boundary_iou', 'support'}.
    """
    class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
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
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        iou  = tp / (tp + fp + fn + 1e-8)
        results[name] = {
            "precision":    float(prec),
            "recall":       float(rec),
            "bf_score":     float(f1),
            "boundary_iou": float(iou),
            "support":      support,
        }

    return results


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def visualize_boundary_predictions(
    images: np.ndarray,
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    indices: list[int] | None = None,
    num_samples: int = 6,
    kernel_size: int = 3,
    save_path: str = "boundary_predictions.png",
    display_fn=None,
) -> None:
    """Save a figure showing input / GT boundary / pred boundary overlays.

    Each row: [Input image | GT boundary overlay | Pred boundary overlay]

    Args:
        images: Float image array (N, H, W, C).
        y_true_labels: Integer label array (N, H, W).
        y_pred_labels: Integer label array (N, H, W).
        indices: Explicit sample indices to display (overrides num_samples).
        num_samples: Number of random samples if indices is None.
        kernel_size: Structuring element size for boundary extraction.
        save_path: Output file path.
        display_fn: Optional callable to convert an image patch to RGB uint8.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if indices is None:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(images), min(num_samples, len(images)), replace=False).tolist()

    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Image", "GT Boundary", "Pred Boundary"]
    boundary_color = np.array([255, 50, 50], dtype=np.uint8)   # red overlay

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

        def _overlay(base_rgb, bnd_mask):
            out = base_rgb.copy()
            out[bnd_mask] = boundary_color
            return out

        gt_overlay   = _overlay(rgb, true_bnd)
        pred_overlay = _overlay(rgb, pred_bnd)

        for col, (title, data) in enumerate(
            zip(col_titles, [rgb, gt_overlay, pred_overlay])
        ):
            axes[row, col].imshow(data)
            axes[row, col].set_title(
                f"{title} (#{idx})" if col == 0 else title,
                fontsize=11, fontweight="bold"
            )
            axes[row, col].axis("off")

    patch = mpatches.Patch(color=(1.0, 50/255, 50/255), label="Boundary pixel")
    fig.legend(handles=[patch], loc="lower center", ncol=1, fontsize=11, frameon=True)
    plt.suptitle("Boundary Region Predictions", fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_boundary_metrics_chart(
    per_class_boundary: dict,
    class_colors: list[str],
    save_path: str = "boundary_metrics_chart.png",
) -> None:
    """Save a grouped bar chart of per-class boundary metrics.

    Args:
        per_class_boundary: Output of compute_boundary_metrics_per_class().
        class_colors: List of hex color strings for each class bar.
        save_path: Output file path.
    """
    import matplotlib.pyplot as plt

    class_names = list(per_class_boundary.keys())
    n = len(class_names)
    x = np.arange(n)
    width = 0.22

    metrics_keys = ["precision", "recall", "bf_score", "boundary_iou"]
    labels       = ["Precision", "Recall", "BF-score", "Boundary IoU"]
    bar_colors   = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

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
