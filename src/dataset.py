import os
import cv2
import numpy as np
from patchify import patchify
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Class RGB values
CLASSES = {
    "Building": [60, 16, 152],
    "Land": [132, 41, 246],
    "Road": [110, 193, 228],
    "Vegetation": [254, 221, 58],
    "Water": [226, 169, 41],
    "Unlabeled": [155, 155, 155]
}

def rgb_to_label(mask):
    """Converts RGB mask to 1D label mask"""
    if mask.ndim == 2 or mask.shape[-1] == 1:
        return mask.astype(np.uint8)
    
    label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    label_mask[np.all(mask == CLASSES["Unlabeled"],  axis=-1)] = 0
    label_mask[np.all(mask == CLASSES["Building"],   axis=-1)] = 1
    label_mask[np.all(mask == CLASSES["Land"],       axis=-1)] = 2
    label_mask[np.all(mask == CLASSES["Road"],       axis=-1)] = 3
    label_mask[np.all(mask == CLASSES["Vegetation"], axis=-1)] = 4
    label_mask[np.all(mask == CLASSES["Water"],      axis=-1)] = 5
    return label_mask

def load_data(data_path, patch_size=256, step=160):
    """Loads images and masks, resizes them, and creates patches"""
    images = []
    masks = []
    
    # The original code used range(1, 8) for tiles and range(1, 9) for parts
    # I'll check how many tiles are actually there in the directory structure
    # Based on session context, Tiles 1 to 8 exist.
    
    tiles = [f"Tile {i}" for i in range(1, 9)]
    for tile in tiles:
        tile_path = os.path.join(data_path, tile)
        if not os.path.exists(tile_path):
            continue
            
        img_dir = os.path.join(tile_path, 'images')
        msk_dir = os.path.join(tile_path, 'masks')
        
        # Sort files to ensure matching pairs
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        for img_file in img_files:
            mask_file = img_file.replace('.jpg', '.png')
            img_path = os.path.join(img_dir, img_file)
            msk_path = os.path.join(msk_dir, mask_file)
            
            if not os.path.exists(msk_path):
                print(f"Warning: Mask not found for {img_path}")
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            msk = cv2.imread(msk_path, cv2.IMREAD_COLOR)
            
            if img is not None and msk is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
                
                # Resize to be divisible by patch_size
                h, w = img.shape[:2]
                new_h = (h // patch_size) * patch_size
                new_w = (w // patch_size) * patch_size
                
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                msk = cv2.resize(msk, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
                label_msk = rgb_to_label(msk)
                
                # Patchify
                img_patches = patchify(img, (patch_size, patch_size, 3), step=step)
                msk_patches = patchify(label_msk, (patch_size, patch_size), step=step)
                
                for r in range(img_patches.shape[0]):
                    for c in range(img_patches.shape[1]):
                        images.append(img_patches[r, c, 0])
                        masks.append(msk_patches[r, c])
            else:
                print(f"Warning: Could not read {img_path} or {msk_path}")
                
    return np.array(images), np.array(masks)

def prepare_dataset(data_path, patch_size=256, step=160, test_size=0.25, random_state=42):
    """Loads and splits the dataset into train and test sets"""
    x, y = load_data(data_path, patch_size, step)
    
    n_classes = len(np.unique(y))
    y_cat = to_categorical(y, num_classes=n_classes)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y_cat, test_size=test_size, random_state=random_state)
    
    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    return x_train, x_test, y_train, y_test, n_classes
