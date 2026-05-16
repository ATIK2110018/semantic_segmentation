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

from src.model import build_residual_attention_unet
from src.dataset import prepare_dataset
from src.utils import get_masked_loss, calculate_class_weights, plot_history, visualize_prediction

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
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
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

    # 7. Evaluation and Visualization
    plot_history(history)
    if args.visualize:
        visualize_prediction(model, x_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual Attention U-Net for Semantic Segmentation")
    parser.add_argument("--data_path", type=str, default="Semantic segmentation dataset", help="Path to dataset")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of patches")
    parser.add_argument("--patch_step", type=int, default=160, help="Step size for patching")
    parser.add_argument("--epochs", type=int, default=170, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=8e-6, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--model_save_path", type=str, default="model.keras", help="Path to save the model")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions after training")
    
    # Ablation Study Arguments
    parser.add_argument("--disable_attention", action="store_true", help="Disable attention gates for ablation study")
    parser.add_argument("--disable_residual", action="store_true", help="Disable residual connections for ablation study")
    
    args = parser.parse_args()
    main(args)
