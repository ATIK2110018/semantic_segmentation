import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def calculate_class_weights(labels, num_classes=None, ignore_label=None):
    """Calculate inverse-frequency class weights for one-hot segmentation loss."""
    if num_classes is None:
        num_classes = int(np.max(labels)) + 1

    cls_counts = np.bincount(labels.reshape(-1), minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = cls_counts > 0
    weights[present] = 1.0 / np.sqrt(cls_counts[present] / cls_counts[present].sum())

    if ignore_label is not None and 0 <= ignore_label < num_classes:
        weights[ignore_label] = 0.0

    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights.astype(np.float32)


def jacard_coef(y_true, y_pred):
    """Jaccard Coefficient (IoU)."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (
        K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0
    )


def get_masked_loss(class_weights, ignore_label=None, boundary_multiplier=2.0):
    """Return native TensorFlow Dice + Focal loss with optional ignored label masking and boundary weighting."""
    class_weights_tf = tf.constant(class_weights, dtype=tf.float32)

    def total_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

        valid_mask = tf.ones(tf.shape(y_true)[:-1], dtype=tf.float32)
        if ignore_label is not None:
            valid_mask = tf.cast(tf.not_equal(tf.argmax(y_true, axis=-1), ignore_label), tf.float32)

        pixel_weights = tf.reduce_sum(y_true * class_weights_tf, axis=-1)
        categorical_ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal = tf.pow(1.0 - pt, 2.0) * categorical_ce * pixel_weights

        # Boundary-weighted loss component (Path A)
        if boundary_multiplier > 0.0:
            # Dilation: Max pooling with 3x3 kernel on one-hot targets
            dilation = tf.nn.max_pool2d(
                y_true,
                ksize=3,
                strides=1,
                padding="SAME"
            )
            # Erosion: 1.0 - Max pooling of (1.0 - one-hot targets)
            erosion = 1.0 - tf.nn.max_pool2d(
                1.0 - y_true,
                ksize=3,
                strides=1,
                padding="SAME"
            )
            # Boundary map (1.0 at class transitions, 0.0 elsewhere)
            boundary_per_class = dilation - erosion
            boundary_map = tf.reduce_max(boundary_per_class, axis=-1)  # shape: (B, H, W)
            
            # Apply boundary multiplier to focus loss on edges
            boundary_weight = 1.0 + boundary_multiplier * boundary_map
            focal = focal * boundary_weight

        focal = tf.reduce_sum(focal * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-6)

        mask_expanded = tf.expand_dims(valid_mask, axis=-1)
        y_true_masked = y_true * mask_expanded
        y_pred_masked = y_pred * mask_expanded
        intersection = tf.reduce_sum(y_true_masked * y_pred_masked, axis=[0, 1, 2])
        denominator = tf.reduce_sum(y_true_masked + y_pred_masked, axis=[0, 1, 2])
        dice_per_class = (2.0 * intersection + 1.0) / (denominator + 1.0)
        dice_loss = 1.0 - tf.reduce_sum(dice_per_class * class_weights_tf)

        return 0.5 * dice_loss + 0.5 * focal

    return total_loss


def plot_history(history, save_path="training_history.png"):
    """Plot training and validation loss and IoU."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    iou_key = "iou_score" if "iou_score" in history.history else "iou"
    if iou_key in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history[iou_key], label="Train IoU")
        plt.plot(history.history[f"val_{iou_key}"], label="Val IoU")
        plt.title("IoU Score")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def display_image(image):
    """Return RGB channels suitable for matplotlib from RGB or Sentinel-2 B2/B3/B4/B8."""
    if image.shape[-1] >= 4:
        rgb = image[..., [2, 1, 0]]
    else:
        rgb = image[..., :3]
    return np.clip(rgb, 0.0, 1.0)


def visualize_prediction(
    model,
    x_test,
    y_test,
    num_samples=5,
    save_path="predictions.png",
    cmap=None,
    class_count=None,
):
    """Visualize predictions on random test samples and save to file."""
    import matplotlib.pyplot as plt

    class_count = class_count or y_test.shape[-1]
    indices = np.random.choice(len(x_test), min(num_samples, len(x_test)), replace=False)
    plt.figure(figsize=(15, 4 * len(indices)))

    for i, idx in enumerate(indices):
        test_img = x_test[idx]
        true_mask = np.argmax(y_test[idx], axis=-1)
        pred_mask = np.argmax(model.predict(np.expand_dims(test_img, 0), verbose=0)[0], axis=-1)

        plt.subplot(len(indices), 3, i * 3 + 1)
        plt.imshow(display_image(test_img))
        plt.title(f"Sample {idx}")
        plt.axis("off")

        plt.subplot(len(indices), 3, i * 3 + 2)
        plt.imshow(true_mask, cmap=cmap, vmin=0, vmax=class_count - 1, interpolation="nearest")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(len(indices), 3, i * 3 + 3)
        plt.imshow(pred_mask, cmap=cmap, vmin=0, vmax=class_count - 1, interpolation="nearest")
        plt.title("Model Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def evaluate_model_metrics(model, x_test, y_test, batch_size=16, class_names=None):
    """Calculate confusion matrix and per-class metrics efficiently in batches."""
    from sklearn.metrics import confusion_matrix

    num_classes = y_test.shape[-1]
    class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i : i + batch_size]
        y_batch = y_test[i : i + batch_size]
        y_pred_batch = model.predict(x_batch, verbose=0)
        cm += confusion_matrix(
            np.argmax(y_batch, axis=-1).flatten(),
            np.argmax(y_pred_batch, axis=-1).flatten(),
            labels=np.arange(num_classes),
        )

    ious = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        ious.append(iou)
        print(f"{name}: IoU={iou:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    print(f"Mean IoU: {np.mean(ious):.4f}")
    return cm, ious
