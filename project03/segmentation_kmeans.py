import os
import numpy as np
from constants import IMAGES_DIR, PREPROCESSED_DIR, SEGMENTED_KMEANS_DIR
import cv2

KMEANS_CLASSES = 3

def load_image(image_src: str):
    img = cv2.imread(image_src)
    return img

def extract_hsv_features(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv_img.astype(np.float32))
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
    # labels = (h, w, 1) where each pixel has a label corresponding to the cluster it belongs to
    h, w = labels.shape[:2]
    patch_height = max(1, int(h * fraction_of_center))
    patch_width = max(1, int(w * fraction_of_center))

    # Take the small center section of the image
    center_patch = labels[h//2 - patch_height//2:h//2 + patch_height//2, w//2 - patch_width//2:w//2 + patch_width//2]

    # Get all cluster classes that appear in the center patch and count how many times they appear
    values, counts = np.unique(center_patch, return_counts=True)

    # Get the max cluster index that appears in the center patch
    foreground_cluster = int(values[np.argmax(counts)])

    return foreground_cluster

def build_mask_from_cluster(img, labels, fg_cluster):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[labels == fg_cluster] = 255
    return mask

def clean_mask(mask):
    kernel = np.ones((5,5), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel=kernel)
    return closed

def build_segmented_image_from_mask(img, mask):
    result = img.copy()
    result[mask != 255] = (0, 0, 0)
    return result

if __name__ == "__main__":
    preprocessed_images_dir = os.path.join(PREPROCESSED_DIR, IMAGES_DIR)
    segmented_kmeans_images_dir = os.path.join(SEGMENTED_KMEANS_DIR, IMAGES_DIR)
    segmented_kmeans_masks_dir = os.path.join(SEGMENTED_KMEANS_DIR, "masks")
    os.makedirs(segmented_kmeans_images_dir, exist_ok=True)
    os.makedirs(segmented_kmeans_masks_dir, exist_ok=True)


    for filename in os.listdir(preprocessed_images_dir):
        img_path = os.path.join(preprocessed_images_dir, filename)
        img = load_image(img_path)
        features = extract_hsv_features(img)

        compactness, labels, center = run_kmeans(features)

        # resize to be (h,w) instead of (h*w, 1) so we know the exact coordinates of each pixel and corresponding class
        labels_reshaped = labels.reshape(img.shape[0], img.shape[1])
        fg_cluster = choose_foreground_cluster(labels_reshaped)
        mask = build_mask_from_cluster(img, labels_reshaped, fg_cluster)
        new_mask = clean_mask(mask)
        resulting_segmentation = build_segmented_image_from_mask(img, clean_mask)

        img_name_parts = filename.split("_")
        img_name_parts[2] = "segmentation_kmeans"
        img_name = "_".join(img_name_parts)
        img_save_path = os.path.join(segmented_kmeans_images_dir, img_name)
        mask_save_path = os.path.join(segmented_kmeans_masks_dir, img_name)

        cv2.imwrite(img_save_path, resulting_segmentation)
        cv2.imwrite(mask_save_path, new_mask)

        

        