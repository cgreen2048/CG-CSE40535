import os
import cv2
import numpy as np
from constants import IMAGES_DIR, MASKS_DIR, PREPROCESSED_DIR, SEGMENTED_DIR


GRABCUT_ITERATIONS = 8


def load_image(image_src: str):
    img = cv2.imread(image_src)
    return img


def parse_roi_from_filename(filename: str):
    """
    Expected filename format similar to:
    IMG_5078_preprocessed_256_225_0_15_Crimp.jpg

    where:
      parts[3] = new_h
      parts[4] = new_w
      parts[5] = h_offset
      parts[6] = w_offset
    """
    parts = filename.split("_")
    new_h, new_w, h_offset, w_offset = map(int, parts[3:7])
    return new_h, new_w, h_offset, w_offset


def extract_roi(img, new_h: int, new_w: int, h_offset: int, w_offset: int):
    return img[h_offset:h_offset + new_h, w_offset:w_offset + new_w]


def run_grabcut_on_roi(roi, iterations: int = GRABCUT_ITERATIONS):
    """
    Runs GrabCut on the ROI using a mask-based initialization.
    Assumes the climbing hold is roughly centered in the ROI.
    """
    h, w = roi.shape[:2]

    if h < 2 or w < 2:
        return np.zeros((h, w), dtype=np.uint8)

    # GrabCut mask labels:
    # 0 = cv2.GC_BGD       definite background
    # 1 = cv2.GC_FGD       definite foreground
    # 2 = cv2.GC_PR_BGD    probable background
    # 3 = cv2.GC_PR_FGD    probable foreground
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Start with a centered rectangle as probable foreground.
    # Since your ROI is already a crop around the hold, this is a reasonable prior.
    margin_y = max(1, int(0.12 * h))
    margin_x = max(1, int(0.12 * w))

    inner_y0 = margin_y
    inner_y1 = h - margin_y
    inner_x0 = margin_x
    inner_x1 = w - margin_x

    # If the ROI is tiny, fall back to the full ROI as probable foreground
    if inner_y1 <= inner_y0 or inner_x1 <= inner_x0:
        inner_y0, inner_y1 = 0, h
        inner_x0, inner_x1 = 0, w

    gc_mask[inner_y0:inner_y1, inner_x0:inner_x1] = cv2.GC_PR_FGD

    # Strengthen the very center as definite foreground.
    center_margin_y = max(1, int(0.30 * h))
    center_margin_x = max(1, int(0.30 * w))

    cy0 = center_margin_y
    cy1 = h - center_margin_y
    cx0 = center_margin_x
    cx1 = w - center_margin_x

    if cy1 > cy0 and cx1 > cx0:
        gc_mask[cy0:cy1, cx0:cx1] = cv2.GC_FGD

    # Keep the outer border as definite background.
    gc_mask[0, :] = cv2.GC_BGD
    gc_mask[-1, :] = cv2.GC_BGD
    gc_mask[:, 0] = cv2.GC_BGD
    gc_mask[:, -1] = cv2.GC_BGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        roi,
        gc_mask,
        None,
        bg_model,
        fg_model,
        iterations,
        cv2.GC_INIT_WITH_MASK
    )

    # Final binary mask: treat definite/probable foreground as foreground
    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return mask


def clean_mask(mask):
    kernel_open = np.ones((3, 3), dtype=np.uint8)
    kernel_close = np.ones((5, 5), dtype=np.uint8)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    return closed


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    # Skip component 0 because it's the background
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

    segmented_grabcut_dir = os.path.join(SEGMENTED_DIR, "grabcut")
    segmented_grabcut_images_dir = os.path.join(segmented_grabcut_dir, "images")
    segmented_grabcut_masks_dir = os.path.join(segmented_grabcut_dir, "masks")

    os.makedirs(segmented_grabcut_images_dir, exist_ok=True)
    os.makedirs(segmented_grabcut_masks_dir, exist_ok=True)

    for filename in os.listdir(preprocessed_images_dir):
        img_path = os.path.join(preprocessed_images_dir, filename)

        if not os.path.isfile(img_path):
            continue

        img = load_image(img_path)
        if img is None:
            print(f"Skipping unreadable image: {filename}")
            continue

        try:
            new_h, new_w, h_offset, w_offset = parse_roi_from_filename(filename)
        except Exception as e:
            print(f"Skipping {filename}: could not parse ROI metadata ({e})")
            continue

        roi = extract_roi(img, new_h, new_w, h_offset, w_offset)
        if roi.size == 0:
            print(f"Skipping {filename}: empty ROI")
            continue

        roi_mask = run_grabcut_on_roi(roi, iterations=GRABCUT_ITERATIONS)
        roi_mask = clean_mask(roi_mask)
        roi_mask = keep_largest_component(roi_mask)

        segmented_roi = build_segmented_image_from_mask(roi, roi_mask)

        # Reconstruct into full preprocessed image size
        result = np.zeros_like(img)
        result[h_offset:h_offset + new_h, w_offset:w_offset + new_w] = segmented_roi

        # Optional: also place ROI mask into full image-sized mask for easier debugging
        full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        full_mask[h_offset:h_offset + new_h, w_offset:w_offset + new_w] = roi_mask

        img_name_parts = filename.split("_")
        if len(img_name_parts) > 2:
            img_name_parts[2] = "segmentation_grabcut"
            save_name = "_".join(img_name_parts)
        else:
            save_name = filename

        img_save_path = os.path.join(segmented_grabcut_images_dir, save_name)
        mask_save_path = os.path.join(segmented_grabcut_masks_dir, save_name)

        cv2.imwrite(img_save_path, result)
        cv2.imwrite(mask_save_path, full_mask)

        print(f"Saved: {img_save_path}")