import os

DATA_DIR = "data"
IMAGES_DIR = "images"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CROPPED_DIR = os.path.join(DATA_DIR, "cropped")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
SEGMENTED_DIR = os.path.join(DATA_DIR, "segmented")
SEGMENTED_KMEANS_DIR = os.path.join(SEGMENTED_DIR, "kmeans")