from segment_anything import sam_model_registry, SamPredictor
import cv2
import os
import numpy as np
from skimage import measure
from constants import (
    create_directories, 
    SAM_MODEL_TYPE, 
    SAM_CHECKPOINT, 
    SEGMENTED_ANYTHING_DIR, 
    IMAGES_DIR, 
    PREPROCESSED_DIR
)

def load_img(img_src):
    img = cv2.imread(img_src)
    return img

def get_roi_dimensions(img_name):
    img_name_parts = img_name.split('_')
    new_h, new_w, h_offset, w_offset = map(int, img_name_parts[3:7])
    return new_h, new_w, h_offset, w_offset

def get_roi_from_preprocessed(img_src):
    img = load_img(img_src)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    filename = os.path.basename(img_src)
    new_h, new_w, h_offset, w_offset = get_roi_dimensions(filename)
    roi = img_rgb[h_offset:h_offset+new_h, w_offset:w_offset+new_w]

    return img_rgb, roi, h_offset, w_offset

def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask
    
    largest_label = 1 + np.argmax(stats[1: , cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255

def fill_holes(mask):
    mask_floodfilled = mask.copy()
    cv2.floodFill(mask_floodfilled, None, (0,0), 255)

    mask_floodfilled_inv = cv2.bitwise_not(mask_floodfilled)
    return cv2.bitwise_or(mask, mask_floodfilled_inv)

def get_image_segmentation(img, mask, h_offset, w_offset):
    result_img = img.copy()
    result_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask_h, mask_w = mask.shape

    roi = result_img[h_offset:h_offset+mask_h, w_offset:w_offset+mask_w]
    roi[mask == 0] = (0,0,0) 
    result_mask[h_offset:h_offset+mask_h, w_offset:w_offset+mask_w] = mask

    return result_img, result_mask

def get_output_paths(filename, images_dir, masks_dir):
    img_name_parts = filename.split('_')
    img_name = "_".join(img_name_parts[:2]) + "_segmented_" + img_name_parts[-1]
    segmented_img_save_path = os.path.join(images_dir, img_name)
    segmented_mask_save_path = os.path.join(masks_dir, img_name)
    return segmented_img_save_path, segmented_mask_save_path

def get_local_features(mask):
    mask_labels = measure.label(mask, connectivity=2)
    props = measure.regionprops(mask_labels)
    if len(props) != 0:
        largest_region = max(props, key=lambda x: x.area)
        features = np.array([
            largest_region.area,
            largest_region.perimeter,
            largest_region.eccentricity,
            largest_region.solidity,
            largest_region.extent,
            largest_region.axis_major_length,
            largest_region.axis_minor_length
        ], dtype=np.float32)
        return features
    else:
        return np.zeros(7, dtype=np.float32)

def segment_one_image(img_name, predictor, pad=5):
    preprocessed_images_dir = os.path.join(PREPROCESSED_DIR, IMAGES_DIR)
    img_rgb, roi, h_offset, w_offset = get_roi_from_preprocessed(os.path.join(preprocessed_images_dir, img_name))

    predictor.set_image(roi)

    roi_h, roi_w = roi.shape[:2]
    input_box = np.array([0,0, roi_w - 1, roi_h - 1], dtype=np.float32)

    point_coords = np.array([
        [roi_w // 2, roi_h // 2],
        [roi_w // 2 + pad, roi_h // 2],
        [roi_w // 2 - pad, roi_h // 2],
        [pad, pad],
        [roi_w - 1 - pad, pad],
        [pad, roi_h - 1 - pad],
        [roi_w - 1 - pad, roi_h - 1 - pad]
    ], dtype=np.float32)
    point_labels = np.array([1,1,1,0,0,0,0])

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box = input_box[None, :],
        multimask_output=True
    )

    max_score_index = int(np.argmax(scores))
    max_score_mask = masks[max_score_index]

    mask = (max_score_mask > 0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    # mask = fill_holes(mask)

    segmented_img, resized_mask = get_image_segmentation(img_rgb, mask, h_offset, w_offset)
    final_img = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)
    return final_img, resized_mask, get_local_features(mask)

def segment_and_save_images():
    preprocessed_images_dir = os.path.join(PREPROCESSED_DIR, IMAGES_DIR)
    segment_anything_images_dir, segment_anything_masks_dir = create_directories(SEGMENTED_ANYTHING_DIR)
    
    device = "cuda"
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    pad = 5
    local_features = []
    counter = 0

    for filename in os.listdir(preprocessed_images_dir):
        img_path = os.path.join(preprocessed_images_dir, filename)
        final_img, resized_mask, local_feature = segment_one_image(filename, predictor, pad)
    
        local_features.append(local_feature)

        segmented_img_save_path, segmented_mask_save_path = get_output_paths(filename, segment_anything_images_dir, segment_anything_masks_dir)
        cv2.imwrite(segmented_img_save_path, final_img)
        cv2.imwrite(segmented_mask_save_path, resized_mask)

        print(f"Processed image {counter}")
        counter += 1

    print("Finished segmenting & saving images.")

    return np.vstack(local_features)



if __name__ == "__main__":
    segment_and_save_images()