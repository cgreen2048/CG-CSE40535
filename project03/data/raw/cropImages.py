import json
import os
import cv2

IMAGES_DIR = "images"

def read_json_data(filename):
    with open(filename) as f:
        data = json.load(f)
        return data
    
def get_img_name(imgSrc: str):
    return os.path.basename(imgSrc).split(".")[0]
    
def crop_image(imgSrc: str, bboxLeft: int, bboxTop: int, bboxWidth: int, bboxHeight: int, labelValue: str):
    img = cv2.imread(imgSrc)
    if img is None:
        raise ValueError(f"Could not read image: {imgSrc}")
    h,w = img.shape[:2]
    left = min(0, int(bboxLeft))
    top = min(0, int(bboxTop))
    right = min(w, int(bboxLeft + bboxWidth))
    bottom = min(h, int(bboxTop + bboxHeight))

    cropped = img[top:bottom, left:right]
    imgName = get_img_name(imgSrc)
    cv2.imwrite(f"../cropped/images/{imgName}_cropped_{bboxLeft}_{bboxTop}_{labelValue}.jpg", cropped)
    return cropped


if __name__ == "__main__":
    data = read_json_data("annotations.json")

    for record in data:
        imgSrc = os.path.join(IMAGES_DIR, record["filename"])
        cropped = crop_image(imgSrc, record["bboxLeft"], record["bboxTop"], record["bboxWidth"], record["bboxHeight"], record["label_name"])

    print("Finished cropping")