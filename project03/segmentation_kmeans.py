import os
import numpy as np
from constants import IMAGES_DIR, MASKS_DIR, PREPROCESSED_DIR, SEGMENTED_KMEANS_DIR
import cv2

KMEANS_CLASSES = 3

def load_image(image_src: str):
    img = cv2.imread(image_src)
    return img

def extract_lab_features(roi):
    lab_img = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    l = lab_img[:,:,0].astype(np.float32)
    a = lab_img[:,:,1].astype(np.float32)
    b = lab_img[:,:,2].astype(np.float32)
    y, x = np.indices(l.shape)
    y = y.astype(np.float32) / l.shape[0]
    x = x.astype(np.float32) / l.shape[1]
    a = a / 255.0
    b = b / 255.0

    features = np.stack((a, b, 0.3 * x, 0.3 * y), axis=-1) # (H, W, 5)
    features = features.reshape((-1, 4))   # (H*W, 5)

    return features

def run_kmeans(features):
    # https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.98)
    compactness, labels, center = cv2.kmeans(features, KMEANS_CLASSES, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return compactness, labels, center

# def choose_foreground_cluster(labels, fraction_of_center: float = 0.3):
#     # Choose the cluster that appears in the center of the image most often
#     # we know that the hold itself will be in the center of the image, so we can use this as a heuristic to choose the foreground cluster
#     # labels = (h, w, 1) where each pixel has a label corresponding to the cluster it belongs to
#     h, w = labels.shape[:2]
#     patch_height = max(1, int(h * fraction_of_center))
#     patch_width = max(1, int(w * fraction_of_center))

#     # Take the small center section of the image
#     center_patch = labels[h//2 - patch_height//2:h//2 + patch_height//2, w//2 - patch_width//2:w//2 + patch_width//2]

#     # Get all cluster classes that appear in the center patch and count how many times they appear
#     values, counts = np.unique(center_patch, return_counts=True)

#     # Get the max cluster index that appears in the center patch
#     foreground_cluster = int(values[np.argmax(counts)])

#     return foreground_cluster

def choose_foreground_cluster(labels, fraction_of_center: float = 0.3):
    h, w = labels.shape[:2]
    patch_height = max(1, int(h * fraction_of_center))
    patch_width = max(1, int(w * fraction_of_center))

    center_patch = labels[
        h // 2 - patch_height // 2 : h // 2 + patch_height // 2,
        w // 2 - patch_width // 2  : w // 2 + patch_width // 2
    ]

    values, counts = np.unique(center_patch, return_counts=True)
    foreground_cluster = int(values[np.argmax(counts)])
    return foreground_cluster

def keep_component_containing_center(cluster_mask, fraction_of_center: float = 0.15):
    """
    cluster_mask: binary mask (uint8), 255 where the chosen cluster is present, 0 elsewhere

    Returns a new binary mask containing only the connected component
    that best matches the image center.
    """
    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(cluster_mask, connectivity=8)

    # No foreground components found
    if num_labels <= 1:
        return cluster_mask.copy()

    h, w = cluster_mask.shape
    cy, cx = h // 2, w // 2

    # Case 1: exact center lies inside a foreground connected component
    center_component = cc_labels[cy, cx]
    if center_component != 0:
        result = np.zeros_like(cluster_mask)
        result[cc_labels == center_component] = 255
        return result

    # Case 2: exact center is not inside foreground (e.g. bolt hole/shadow/hole in mask)
    # Look at a small patch around the center and choose the most common non-background component there
    patch_height = max(1, int(h * fraction_of_center))
    patch_width = max(1, int(w * fraction_of_center))

    y0 = max(0, cy - patch_height // 2)
    y1 = min(h, cy + patch_height // 2)
    x0 = max(0, cx - patch_width // 2)
    x1 = min(w, cx + patch_width // 2)

    center_patch = cc_labels[y0:y1, x0:x1]

    component_ids, counts = np.unique(center_patch, return_counts=True)

    # Remove background label 0
    nonzero = component_ids != 0
    component_ids = component_ids[nonzero]
    counts = counts[nonzero]

    # If no component appears near the center, fall back to largest connected component
    if len(component_ids) == 0:
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        result = np.zeros_like(cluster_mask)
        result[cc_labels == largest_component] = 255
        return result

    chosen_component = int(component_ids[np.argmax(counts)])

    result = np.zeros_like(cluster_mask)
    result[cc_labels == chosen_component] = 255
    return result

def build_mask_from_cluster(img, labels, fg_cluster):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[labels == fg_cluster] = 255
    return mask

def clean_mask(mask):
    kernel = np.ones((5,5), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel=kernel)
    return closed

def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    # skip label 0 because it's background
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result

def build_segmented_image_from_mask(img, mask):
    result = img.copy()
    result[mask != 255] = (0, 0, 0)
    return result

if __name__ == "__main__":
    preprocessed_images_dir = os.path.join(PREPROCESSED_DIR, IMAGES_DIR)
    segmented_kmeans_images_dir = os.path.join(SEGMENTED_KMEANS_DIR, IMAGES_DIR)
    segmented_kmeans_roi_dir = os.path.join(SEGMENTED_KMEANS_DIR, "roi")
    segmented_kmeans_masks_dir = os.path.join(SEGMENTED_KMEANS_DIR, MASKS_DIR)
    os.makedirs(segmented_kmeans_images_dir, exist_ok=True)
    os.makedirs(segmented_kmeans_masks_dir, exist_ok=True)
    os.makedirs(segmented_kmeans_roi_dir, exist_ok=True)


    for filename in os.listdir(preprocessed_images_dir):
        img_path = os.path.join(preprocessed_images_dir, filename)
        img = load_image(img_path)
        img_name_parts = filename.split("_")

        new_h, new_w, h_offset, w_offset = img_name_parts[3:7]
        roi = img[int(h_offset):int(h_offset)+int(new_h), int(w_offset):int(w_offset)+int(new_w)]
        roi_save_path = os.path.join(segmented_kmeans_roi_dir, filename)
        cv2.imwrite(roi_save_path, roi)
        features = extract_lab_features(roi)

        compactness, labels, center = run_kmeans(features)

        # resize to be (h,w) instead of (h*w, 1) so we know the exact coordinates of each pixel and corresponding class
        labels_reshaped = labels.reshape(roi.shape[0], roi.shape[1])
        fg_cluster = choose_foreground_cluster(labels_reshaped)
        roi_mask = build_mask_from_cluster(roi, labels_reshaped, fg_cluster)
        roi_mask = clean_mask(roi_mask)
        new_mask = keep_component_containing_center(roi_mask)
        result = np.zeros_like(img)
        segmented_roi = build_segmented_image_from_mask(roi, new_mask)
        result[int(h_offset):int(h_offset)+int(new_h), int(w_offset):int(w_offset)+int(new_w)] = segmented_roi

        img_name_parts = filename.split("_")
        img_name_parts[2] = "segmentation_kmeans"
        img_name = "_".join(img_name_parts)
        img_save_path = os.path.join(segmented_kmeans_images_dir, img_name)
        mask_save_path = os.path.join(segmented_kmeans_masks_dir, img_name)

        cv2.imwrite(img_save_path, result)
        cv2.imwrite(mask_save_path, new_mask)
    print("Finished segmenting")

        

        