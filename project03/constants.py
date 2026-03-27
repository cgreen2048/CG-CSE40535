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