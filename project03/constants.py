import os

DATA_DIR = "data"
IMAGES_DIR = "images"
MASKS_DIR = "masks"

RAW_DIR = os.path.join(DATA_DIR, "raw")
CROPPED_DIR = os.path.join(DATA_DIR, "cropped")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
SEGMENTED_DIR = os.path.join(DATA_DIR, "segmented")
SEGMENTED_SNAKES_DIR = os.path.join(SEGMENTED_DIR, "snakes")
SEGMENTED_KMEANS_DIR = os.path.join(SEGMENTED_DIR, "kmeans")
SEGMENTED_ANYTHING_DIR = os.path.join(SEGMENTED_DIR, "anything")
SIFT_FEATURES_DIR = os.path.join(DATA_DIR, "sift_features")

SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = os.path.join("models", "sam_vit_h_4b8939.pth")

def create_directories(dir_path):
    imgs_path = os.path.join(dir_path, IMAGES_DIR)
    masks_path = os.path.join(dir_path, MASKS_DIR)
    os.makedirs(imgs_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)   

    return imgs_path, masks_path