import os
import numpy as np
from constants import IMAGES_DIR, PREPROCESSED_DIR, SEGMENTED_KMEANS_DIR
import cv2

KMEANS_CLASSES = 2

def load_image(image_src: str):
    img = cv2.imread(image_src)
    return img

def extract_hsv_features(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv_img).astype(np.float32)
    h = h.flatten()
    s = s.flatten()

    features = np.stack((h,s), axis=-1)
    return features

def run_kmeans(features):
    # https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.95)
    compactness, labels, center = cv2.kmeans(features, KMEANS_CLASSES, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return compactness, labels, center

def choose_foreground_cluster(labels, fraction_of_center: float = 0.3):
    # Choose the cluster that appears in the center of the image most often
    # we know that the hold itself will be in the center of the image, so we can use this as a heuristic to choose the foreground cluster
    h, w = labels.shape
    patch_height = max(1, int(h * fraction_of_center))
    patch_width = max(1, int(w * fraction_of_center))

    center_patch = labels[h//2 - patch_height//2:h//2 + patch_height//2, w//2 - patch_width//2:w//2 + patch_width//2]

    values, counts = np.unique(center_patch, return_counts=True)
    foreground_cluster = int(values[np.argmax(counts)])

    return foreground_cluster


if __name__ == "__main__":
    preprocessed_images_dir = os.path.join(PREPROCESSED_DIR, IMAGES_DIR)
    os.makedirs(os.path.join(SEGMENTED_KMEANS_DIR, IMAGES_DIR), exist_ok=True)

    for filename in os.listdir(preprocessed_images_dir):
        img_path = os.path.join(preprocessed_images_dir, filename)
        img = load_image(img_path)
        features = extract_hsv_features(img)

        compactness, labels, center = run_kmeans(features)

        labels_reshaped = labels.reshape(img.shape[0], img.shape[1], 1)
        