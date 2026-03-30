import os
import cv2
import numpy as np
from constants import (
    create_directories, 
    IMAGES_DIR, 
    MASKS_DIR, 
    SEGMENTED_ANYTHING_DIR,
    SIFT_FEATURES_DIR
)
from skimage import measure

def load_img(img_src):
    img = cv2.imread(img_src)
    return img

def construct_sift_feature_vector(descriptors):
    desc_mean = np.mean(descriptors, axis=0)
    desc_std_dev = np.std(descriptors, axis=0)
    desc_num_keypoints = np.array([descriptors.shape[0]], dtype=np.float32)

    # 257-dim feature vector
    return np.concatenate([desc_mean, desc_std_dev, desc_num_keypoints]).astype(np.float32)

def extract_sift_features_one_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=200,      
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )

    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    img_sift_features = construct_sift_feature_vector(descriptors)
    return keypoints, descriptors, img_sift_features

def extract_sift_features():
    segmented_anything_images_dir = os.path.join(SEGMENTED_ANYTHING_DIR, IMAGES_DIR)
    sift_images_dir, _ = create_directories(SIFT_FEATURES_DIR)

    sift_features = []

    for filename in os.listdir(segmented_anything_images_dir):
        img_path = os.path.join(segmented_anything_images_dir, filename)
        img = load_img(img_path)
        keypoints, _, features = extract_sift_features_one_image(img)
        sift_features.append(features)

        keypoints_img = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_name_parts = filename.split('_')
        img_name_parts[2] = "sift"
        img_save_path = os.path.join(sift_images_dir, '_'.join(img_name_parts))
        cv2.imwrite(img_save_path, keypoints_img)

    print(np.vstack(sift_features).shape)
    return np.vstack(sift_features)


if __name__ == "__main__":
    extract_sift_features()