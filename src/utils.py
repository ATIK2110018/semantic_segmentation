import tensorflow as tf
import tensorflow.keras.backend as K
import segmentation_models as sm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define class weights globally or calculate them
def calculate_class_weights(labels):
    """Calculates class weights based on frequency"""
    cls_counts = np.bincount(labels.reshape(-1), minlength=6)
    freq = cls_counts / np.maximum(1, cls_counts.sum())
    cw = 1.0 / np.sqrt(freq + 1e-6)
    cw = cw / cw.sum()
    return cw.astype(np.float32)

def jacard_coef(y_true, y_pred):
    """Jaccard Coefficient (IoU)"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def get_masked_loss(class_weights):
    """Returns a combined Dice and Focal loss function that masks the Unlabeled class"""
    dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
    focal_loss = sm.losses.CategoricalFocalLoss()
    
    def masked_total_loss(y_true, y_pred):
        # Mask out Unlabeled class (index 0)
        mask = tf.not_equal(tf.argmax(y_true, axis=-1), 0)
        mask = tf.cast(mask, tf.float32)
        loss = 0.5 * dice_loss(y_true, y_pred) + 0.5 * focal_loss(y_true, y_pred)
        return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-6)
        
    return masked_total_loss

def plot_history(history, save_path='training_history.png'):
    """Plots training and validation loss and IoU"""
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot IoU
    iou_key = 'iou_score' if 'iou_score' in history.history else 'iou'
    if iou_key in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history[iou_key], label='Train IoU')
        plt.plot(history.history[f'val_{iou_key}'], label='Val IoU')
        plt.title('IoU Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def visualize_prediction(model, x_test, y_test, num_samples=4):
    """Visualizes predictions on random test samples"""
    custom_cmap = ListedColormap([
        (60/255, 16/255, 152/255),   # Building
        (132/255, 41/255, 246/255),  # Land
        (110/255, 193/255, 228/255), # Road
        (254/255, 221/255, 58/255),  # Vegetation
        (226/255, 169/255, 41/255),  # Water
        (155/255, 155/255, 155/255)  # Unlabeled
    ])
    
    y_pred = model.predict(x_test[:num_samples*2]) # Predict more than needed just in case
    
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        test_img = x_test[idx]
        true_mask = np.argmax(y_test[idx], axis=-1)
        pred_mask = np.argmax(model.predict(np.expand_dims(test_img, 0))[0], axis=-1)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(test_img)
        plt.title(f"Sample {idx}")
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(true_mask, cmap=custom_cmap, vmin=0, vmax=5)
        plt.title("True Mask")
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(pred_mask, cmap=custom_cmap, vmin=0, vmax=5)
        plt.title("Predicted Mask")
        plt.axis('off')
        plt.show()
