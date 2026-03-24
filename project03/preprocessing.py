import json
import os
import cv2
from PIL import Image
import numpy as np

DATA_DIR = "data"
IMAGES_DIR = "images"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CROPPED_DIR = os.path.join(DATA_DIR, "cropped")
PRE_PROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

def read_json_data(filename):
    with open(filename) as f:
        data = json.load(f)
        return data
    
def get_img_name(imgSrc: str):
    return os.path.basename(imgSrc).split(".")[0]

def clamp_bbox(left: int, top: int, width: int, height: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """
    Clamp a bbox to image bounds.
    """
    left = max(0, min(left, img_w))
    top = max(0, min(top, img_h))
    right = max(left, min(left + width, img_w))
    bottom = max(top, min(top + height, img_h))
    return left, top, right - left, bottom - top

def pad_and_clamp_bbox(left: int, top: int, width: int, height: int, pad: int, img_w: int, img_h: int):
    new_left = max(0, left - pad)
    new_top = max(0, top - pad)
    new_right = min(img_w, left + width + pad)
    new_bottom = min(img_h, top + height + pad)
    return new_left, new_top, new_right - new_left, new_bottom - new_top
    
def crop_image(imgSrc: str, bboxLeft: int, bboxTop: int, bboxWidth: int, bboxHeight: int, rotation: int, labelValue: str): 
    img = Image.open(imgSrc) 
    
    rawW, rawH = img.size
    left = int(bboxLeft) 
    top = int(bboxTop) 
    width = int(bboxWidth)
    height = int(bboxHeight)
    left, top, width, height = clamp_bbox(left, top, width, height, rawW, rawH)
    left, top, width, height = pad_and_clamp_bbox(left, top, width, height, 4, rawW, rawH)
    

    cropped = img.crop((left,top,left+width,top+height)) 

    if rotation == 3:
        cropped = cropped.rotate(180, expand=True)
    elif rotation == 6:
        cropped = cropped.rotate(-90, expand=True)
    elif rotation == 1:
        pass
    else:
        raise ValueError(f"Unsupported exif_rotation: {rotation}")

    if cropped.mode == "RGBA":
        cropped = cropped.convert("RGB")

    imgName = get_img_name(imgSrc) 
    
    imgPath = f"{os.path.join(CROPPED_DIR, IMAGES_DIR)}/{imgName}_cropped_{bboxLeft}_{bboxTop}_{labelValue}.jpg" 
    cropped.save(imgPath) 
    return cv2.imread(imgPath)

def normalize_image_size(img, size=256):
    h, w = img.shape[:2]

    # Scale down the width and height so neither dimension exceeds 256x256
    # i.e. whichever of two dims is larger will become exactly 256, other will be scaled down by same scale below 256
    scale = min(size / w, size / h)
    new_w = int(round(w*scale))
    new_h = int(round(h*scale))

    # fit inside the new bounds computed above Ax256 or 256xB
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # our initial 256x256 img
    canvas = np.zeros((size,size,3), dtype=np.uint8)

    # shift so we know where to start placing pixels from whichever dimension was scaled down
    # i.e. dim that became 256 has offest = 0 to take up whole image
    # but the other dim will center the image along that axis (padded by black)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas, new_h, new_w, y_offset, x_offset

def clahe_lighting_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_l = clahe.apply(l)
    lab_merged = cv2.merge((clahe_l, a, b))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)



if __name__ == "__main__":
    data_path = os.path.join(RAW_DIR, "annotations.json")
    data = read_json_data(data_path)
    
    os.makedirs(os.path.join(CROPPED_DIR, IMAGES_DIR), exist_ok=True) 
    os.makedirs(os.path.join(PRE_PROCESSED_DIR, IMAGES_DIR), exist_ok=True)
    for record in data:
        imgSrc = os.path.join(RAW_DIR, IMAGES_DIR, record["filename"])

        left = int(record["bbox_left"])
        top = int(record["bbox_top"])
        width = int(record["bbox_width"])
        height = int(record["bbox_height"])
        label_name = record["label_name"]
        cropped = crop_image(
            imgSrc, 
            left, 
            top, 
            width, 
            height, 
            record["exif_rotation"],
            label_name
        )
        resized, new_h, new_w, h_offset, w_offset = normalize_image_size(cropped)
        equalized = clahe_lighting_bgr(resized)

        preprocessed_images_dir = os.path.join(PRE_PROCESSED_DIR, IMAGES_DIR)
        img_name = f"{get_img_name(imgSrc)}_preprocessed_{new_h}_{new_w}_{h_offset}_{w_offset}_{label_name}.jpg"
        img_path = os.path.join(preprocessed_images_dir, img_name)
        cv2.imwrite(img_path, equalized)

    print("Finished preprocessing")