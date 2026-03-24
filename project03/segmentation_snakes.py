import numpy as np
import os
import cv2
from skimage.color import rgb2gray
from skimage import data, draw
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from constants import IMAGES_DIR, PREPROCESSED_DIR, SEGMENTED_DIR


def get_curve_from_bbox(new_h, new_w, h_offset, w_offset):
    sinusoidal = np.linspace(0, 2 * np.pi, 400)
    
    mid_x = w_offset + new_w / 2
    mid_y = h_offset + new_h / 2

    rad_x = new_w / 2
    rad_y = new_h / 2 

    r = mid_y + rad_y * np.sin(sinusoidal)
    c = mid_x + rad_x * np.cos(sinusoidal)

    return np.array([r,c]).T

def run_active_contour(img_gray, init_curve):
    snake = active_contour(
        gaussian(img_gray, sigma=3, preserve_range=False),
        init_curve,
        alpha=0.015,
        beta=10,
        gamma=0.001,
    )

    # shape (N, 2) where each row is one point [row, col] part of the snake
    return snake

if __name__ == "__main__":
    preprocessed_images_dir = os.path.join(PREPROCESSED_DIR, IMAGES_DIR)
    segmented_images_dir = os.path.join(SEGMENTED_DIR, IMAGES_DIR)
    os.makedirs(segmented_images_dir, exist_ok=True)
    os.makedirs(os.path.join(SEGMENTED_DIR, "snakes"), exist_ok=True)

    for filename in os.listdir(preprocessed_images_dir):
        img_path = os.path.join(preprocessed_images_dir, filename)
        img_name_parts = filename.split("_")
        

        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        h, w = img_gray.shape[:2]
        new_h, new_w, h_offset, w_offset = img_name_parts[3:7]
        initial_curve = get_curve_from_bbox(int(new_h), int(new_w), int(h_offset), int(w_offset))
        snake = run_active_contour(img_gray, initial_curve)

        rows = np.round(snake[:, 0]).astype(int)
        cols = np.round(snake[:, 1]).astype(int)
        rows = np.clip(rows, 0, h-1)
        cols = np.clip(cols, 0, w-1)
        img[rows, cols] = (0,0,255)

        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        fill_row_coords, fill_col_coords = draw.polygon(snake[:, 0], snake[:, 1], img_gray.shape)
        mask[fill_row_coords, fill_col_coords] = 255


        img_name_parts[2] = "segmented_snakes"
        img_name = "_".join(img_name_parts)
        img_path = os.path.join(segmented_images_dir, img_name)
        img_snake_path = os.path.join(SEGMENTED_DIR, "snakes", img_name)
        cv2.imwrite(img_path, mask)
        cv2.imwrite(img_snake_path, img)
    print("Finished segmenting")
