import os
from pathlib import Path

import numpy as np
import rasterio


AERIAL_CLASSES = {
    "Building": [60, 16, 152],
    "Land": [132, 41, 246],
    "Road": [110, 193, 228],
    "Vegetation": [254, 221, 58],
    "Water": [226, 169, 41],
    "Unlabeled": [155, 155, 155],
}

AERIAL_CLASS_NAMES = ["Unlabeled", "Building", "Land", "Road", "Vegetation", "Water"]
AERIAL_CLASS_COLORS = ["#9B9B9B", "#3C1098", "#8429F6", "#6EC1E4", "#FEDD3A", "#E2A929"]

# Snow/Ice (original ID 6) and Clouds (original ID 7) are excluded because
# they are absent in the Sylhet study area.  Rangeland shifts to ID 6.
ESRI_CLASS_NAMES = [
    "Water",            # 0
    "Trees",            # 1
    "Flooded Vegetation",  # 2
    "Crops",            # 3
    "Built Area",       # 4
    "Bare Ground",      # 5
    "Rangeland",        # 6  (was 8 in the 9-class scheme)
]
ESRI_CLASS_COLORS = [
    "#1A5BAB",
    "#358221",
    "#87D19E",
    "#FFDB5C",
    "#ED022A",
    "#EDE9E4",
    "#C6AD8D",
]

# Pixel-level remap applied after loading the mask:
#   old 0-5  -> 0-5  (unchanged)
#   old 6    -> 255  (Snow/Ice  — mark as ignore)
#   old 7    -> 255  (Clouds    — mark as ignore)
#   old 8    -> 6    (Rangeland — shift down)
_ESRI_REMAP = np.array(
    [0, 1, 2, 3, 4, 5, 255, 255, 6], dtype=np.uint8
)  # index = old label, value = new label


def rgb_to_label(mask):
    """Convert an RGB aerial mask to integer labels."""
    if mask.ndim == 2 or mask.shape[-1] == 1:
        return mask.astype(np.uint8)

    label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    label_mask[np.all(mask == AERIAL_CLASSES["Unlabeled"], axis=-1)] = 0
    label_mask[np.all(mask == AERIAL_CLASSES["Building"], axis=-1)] = 1
    label_mask[np.all(mask == AERIAL_CLASSES["Land"], axis=-1)] = 2
    label_mask[np.all(mask == AERIAL_CLASSES["Road"], axis=-1)] = 3
    label_mask[np.all(mask == AERIAL_CLASSES["Vegetation"], axis=-1)] = 4
    label_mask[np.all(mask == AERIAL_CLASSES["Water"], axis=-1)] = 5
    return label_mask


def _patch_arrays(image, mask, patch_size, step, valid_pixel_threshold=0.0):
    h, w = image.shape[:2]
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    image = image[:new_h, :new_w]
    mask = mask[:new_h, :new_w]

    images = []
    masks = []
    for row in range(0, new_h - patch_size + 1, step):
        for col in range(0, new_w - patch_size + 1, step):
            image_patch = image[row : row + patch_size, col : col + patch_size, :]
            mask_patch = mask[row : row + patch_size, col : col + patch_size]

            if valid_pixel_threshold > 0:
                valid_fraction = np.isfinite(image_patch).mean()
                if valid_fraction < valid_pixel_threshold:
                    continue

            images.append(np.nan_to_num(image_patch, nan=0.0, posinf=0.0, neginf=0.0))
            masks.append(mask_patch)

    return images, masks


def load_aerial_data(data_path, patch_size=256, step=160):
    """Load the original JPG/PNG aerial imagery dataset."""
    import cv2

    images = []
    masks = []

    tiles = [f"Tile {i}" for i in range(1, 9)]
    for tile in tiles:
        tile_path = os.path.join(data_path, tile)
        if not os.path.exists(tile_path):
            continue

        img_dir = os.path.join(tile_path, "images")
        msk_dir = os.path.join(tile_path, "masks")
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

        for img_file in img_files:
            mask_file = img_file.replace(".jpg", ".png")
            img_path = os.path.join(img_dir, img_file)
            msk_path = os.path.join(msk_dir, mask_file)

            if not os.path.exists(msk_path):
                print(f"Warning: Mask not found for {img_path}")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            msk = cv2.imread(msk_path, cv2.IMREAD_COLOR)
            if img is None or msk is None:
                print(f"Warning: Could not read {img_path} or {msk_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            new_h = (h // patch_size) * patch_size
            new_w = (w // patch_size) * patch_size
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            img_list, mask_list = _patch_arrays(
                img, rgb_to_label(msk), patch_size, step
            )
            images.extend(img_list)
            masks.extend(mask_list)

    metadata = {
        "source": "aerial",
        "class_names": AERIAL_CLASS_NAMES,
        "class_colors": AERIAL_CLASS_COLORS,
        "ignore_label": 0,
    }
    return np.array(images), np.array(masks), metadata


def _find_tif(data_path, include_terms, exclude_terms=()):
    paths = sorted(Path(data_path).rglob("*.tif")) + sorted(Path(data_path).rglob("*.tiff"))
    for path in paths:
        name = path.name.lower()
        if all(term in name for term in include_terms) and not any(
            term in name for term in exclude_terms
        ):
            return path
    return None


def _normalize_sentinel_image(image):
    """Normalize Sentinel-2 channels into [0, 1]."""
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(image))
    if max_val > 1.5:
        image = image / 10000.0

    image = np.clip(image, 0.0, 1.0)
    return image


def load_geotiff_data(
    data_path,
    patch_size=256,
    step=160,
    valid_pixel_threshold=0.5,
    image_tif=None,
    mask_tif=None,
    compute_indices=True,
):
    """Load Sentinel-2 image and ESRI LULC mask GeoTIFFs exported from GEE."""
    image_path = Path(image_tif) if image_tif else _find_tif(data_path, ("sentinel2",), ("mask", "lulc"))
    mask_path = Path(mask_tif) if mask_tif else _find_tif(data_path, ("lulc", "mask"))

    if image_path is None or mask_path is None:
        raise FileNotFoundError(
            "Could not find Sentinel-2 image and ESRI LULC mask GeoTIFFs. "
            "Either pass explicit paths via --image_tif/--mask_tif or place files named like "
            "sylhet_sentinel2_30m_2023.tif and sylhet_esri_lulc_30m_mask_2023.tif under the data directory."
        )
    if not image_path.exists():
        raise FileNotFoundError(f"Sentinel-2 GeoTIFF not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"ESRI LULC mask GeoTIFF not found: {mask_path}")

    with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
        if image_src.width != mask_src.width or image_src.height != mask_src.height:
            raise ValueError(
                "Image and mask rasters must have the same width and height. "
                f"Got image {image_src.width}x{image_src.height}, "
                f"mask {mask_src.width}x{mask_src.height}."
            )
        if image_src.transform != mask_src.transform:
            raise ValueError("Image and mask rasters must have the same transform/grid.")

        image = image_src.read().astype(np.float32)
        mask = mask_src.read(1).astype(np.uint8)

    # Remap mask: collapse 9-class -> 7-class, set absent classes to 255
    mask = np.where(mask < len(_ESRI_REMAP), _ESRI_REMAP[mask], 255).astype(np.uint8)

    image = np.moveaxis(image, 0, -1)
    image = _normalize_sentinel_image(image)

    if compute_indices and image.shape[-1] >= 4:
        eps = 1e-8
        ndvi = (image[..., 3] - image[..., 2]) / (image[..., 3] + image[..., 2] + eps)
        ndwi = (image[..., 1] - image[..., 3]) / (image[..., 1] + image[..., 3] + eps)
        ndvi = np.clip((ndvi + 1.0) / 2.0, 0.0, 1.0)
        ndwi = np.clip((ndwi + 1.0) / 2.0, 0.0, 1.0)
        image = np.concatenate([image, ndvi[..., np.newaxis], ndwi[..., np.newaxis]], axis=-1)

    images, masks = _patch_arrays(
        image,
        mask,
        patch_size,
        step,
        valid_pixel_threshold=valid_pixel_threshold,
    )

    metadata = {
        "source": "geotiff",
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "class_names": ESRI_CLASS_NAMES,
        "class_colors": ESRI_CLASS_COLORS,
        "ignore_label": 255,  # pixels remapped to 255 (Snow/Ice, Clouds) are ignored
    }
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.uint8), metadata


def load_data(
    data_path,
    patch_size=256,
    step=160,
    valid_pixel_threshold=0.5,
    image_tif=None,
    mask_tif=None,
    compute_indices=True,
):
    """Auto-detect and load either the GeoTIFF dataset or original aerial dataset."""
    has_tifs = any(Path(data_path).rglob("*.tif")) or any(Path(data_path).rglob("*.tiff"))
    if has_tifs:
        return load_geotiff_data(
            data_path,
            patch_size,
            step,
            valid_pixel_threshold,
            image_tif=image_tif,
            mask_tif=mask_tif,
            compute_indices=compute_indices,
        )
    return load_aerial_data(data_path, patch_size, step)


def prepare_dataset(
    data_path,
    patch_size=256,
    step=160,
    test_size=0.25,
    random_state=42,
    valid_pixel_threshold=0.5,
    image_tif=None,
    mask_tif=None,
    return_metadata=False,
    compute_indices=True,
):
    """Load, one-hot encode, and split the dataset into train/test sets."""
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    x, y, metadata = load_data(
        data_path,
        patch_size,
        step,
        valid_pixel_threshold,
        image_tif=image_tif,
        mask_tif=mask_tif,
        compute_indices=compute_indices,
    )
    if len(x) == 0:
        raise ValueError("No valid image/mask patches were created from the dataset.")

    n_classes = len(metadata["class_names"])

    # Clamp ignore pixels (255) to 0 before one-hot encoding;
    # the loss function will zero them out via ignore_label masking.
    ignore = metadata.get("ignore_label")
    if ignore is not None:
        y = np.where(y == ignore, 0, y).astype(np.uint8)

    y_cat = to_categorical(y, num_classes=n_classes)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_cat, test_size=test_size, random_state=random_state
    )

    if metadata["source"] == "aerial":
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
    else:
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

    if return_metadata:
        return x_train, x_test, y_train, y_test, n_classes, metadata
    return x_train, x_test, y_train, y_test, n_classes
