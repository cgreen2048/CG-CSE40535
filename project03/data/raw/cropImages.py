import json
import os
import cv2
from PIL import Image, ImageOps
import numpy as np

IMAGES_DIR = "images"
CROPPED_DIR = "../cropped"

def read_json_data(filename):
    with open(filename) as f:
        data = json.load(f)
        return data
    
def get_img_name(imgSrc: str):
    return os.path.basename(imgSrc).split(".")[0]
    
def crop_image(imgSrc: str, bboxLeft: int, bboxTop: int, bboxWidth: int, bboxHeight: int, labelValue: str): 
    img =Image.open(imgSrc) 
    img = ImageOps.exif_transpose(img) 
    
    left = int(bboxLeft) 
    top = int(bboxTop) 
    right = int(bboxLeft + bboxWidth) 
    bottom = int(bboxTop + bboxHeight) 
    cropped = img.crop((left,top,right,bottom)) 
    if cropped.mode == "RGBA":
        cropped = cropped.convert("RGB")

    imgName = get_img_name(imgSrc) 
    os.makedirs(f"{CROPPED_DIR}/{IMAGES_DIR}", exist_ok = True) 
    
    imgPath = f"{os.path.join(CROPPED_DIR, IMAGES_DIR)}/{imgName}_cropped_{bboxLeft}_{bboxTop}_{labelValue}.jpg" 
    cropped.save(imgPath) 
    return cv2.imread(imgPath)


if __name__ == "__main__":
    data = read_json_data("annotations.json")

    for record in data:
        imgSrc = os.path.join(IMAGES_DIR, record["filename"])
        cropped = crop_image(
            imgSrc, 
            record["bbox_left"], 
            record["bbox_top"], 
            record["bbox_width"], 
            record["bbox_height"], 
            record["label_name"]
        )

    print("Finished cropping")