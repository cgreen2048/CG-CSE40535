import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from preprocessing import read_json_data, preprocess_one_image
from segmentation_anything import segment_one_image
from feature_extraction_sift import extract_sift_features_one_image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from segment_anything import sam_model_registry, SamPredictor
from torchvision import models
from constants import (
    IMAGES_DIR,
    RAW_DIR,
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
    SIFT_FEATURES_DIR
)

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

def crop_image(
    img_src: str,
    bbox_left: int,
    bbox_top: int,
    bbox_width: int,
    bbox_height: int, 
    rotation: int,
    label_name: str,
    label_value: int,
    pad: int = 4
):
    img = Image.open(img_src)

    if img.mode == "RGBA":
        img = img.convert("RGB")

    raw_w, raw_h = img.size
    left, top, width, height = clamp_bbox(bbox_left, bbox_top, bbox_width, bbox_height, raw_w, raw_h)
    left, top, width, height = pad_and_clamp_bbox(left, top, width, height, 4, raw_w, raw_h)
    out = {
        "bbox_left": left,
        "bbox_top": top, 
        "bbox_width": width,
        "bbox_height": height,
        "raw_w": raw_w,
        "raw_h": raw_h,
        "rotation": rotation,
        "label_name": label_name,
        "label_value": label_value
    }

    cropped = img.crop((left,top,left+width,top+height)) 

    if rotation == 3:
        cropped = cropped.rotate(180, expand=True)
    elif rotation == 6:
        cropped = cropped.rotate(-90, expand=True)
    elif rotation == 1:
        pass
    else:
        raise ValueError(f"Unsupported exif_rotation: {rotation}")
    
    return cv2.imread(cropped), out

def normalize_img_size(img, size=256):
    h, w = img.shape[:2]

    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    resize_meta = {
        canvas_size=size,
        new_h=new_h,
        new_w=new_w,
        y_offset=y_offset,
        x_offset=x_offset,
    }
    return canvas, resize_meta


def clahe_lighting_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_l = clahe.apply(l)
    lab_merged = cv2.merge((clahe_l, a, b))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

def preprocess_one_image(
    img_src: str,
    bbox_left: int,
    bbox_top: int,
    bbox_width: int,
    bbox_height: int,
    rotation: int,
    label_name: str,
    label_value: int,
    output_size: int = 256,
    pad: int = 4,
):
    cropped, metadata = crop_image(
        img_src,
        bbox_left,
        bbox_top,
        bbox_width,
        bbox_height, 
        rotation,
        label_name,
        label_value,
        pad
    )
    normalized, resize_meta = normalize_img_size(cropped, size=output_size)
    equalized = clahe_lighting_bgr(normalized)

    merged_metadata = metadata | resize_meta
    return equalized, merged_metadata

def get_roi_from_preprocessed(img, new_h, new_w, h_offset, w_offset):
    return img[h_offset:h_offset+new_h, w_offset:w_offset+new_w]

def get_image_segmentation(img, mask, h_offset, w_offset):
    result_img = img.copy()
    result_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask_h, mask_w = mask.shape

    roi = result_img[h_offset:h_offset+mask_h, w_offset:w_offset+mask_w]
    roi[mask == 0] = (0,0,0) 
    result_mask[h_offset:h_offset+mask_h, w_offset:w_offset+mask_w] = mask

def segment_one_image(img, predictor, new_h, new_w, h_offset, w_offset, pad=5):
    roi = get_roi_from_preprocessed(img, new_h, new_w, h_offset, w_offset)

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

    segmented_img, resized_mask = get_image_segmentation(img, mask, h_offset, w_offset)
    final_img = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)
    return final_img, resized_mask


# add train CNN function later


def get_cnn_features_SVM(model, dataloader, device):
    model.eval()
    model.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model(images)
            features = features.flatten(start_dim=1)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_features), np.concatenate(all_labels)








def build_feature_extractor_RESNET():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])  # remove final FC
    return model


def run_svm_classification(X, Y):
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y,
        test_size=0.3,
        random_state=42,
        stratify=Y   
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp,
        test_size=0.5,
        random_state=42,
        stratify=Y_temp
    )

    X_test = scaler.fit_transform(X_test)

    # Change to linear to see other results
    clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_val)

    accuracy = np.mean(Y_pred == Y_val)
    print(f"SVM Classification Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")

    # UNCOMMENT ONLY WHEN DOING FINAL TEST ON CHOSEN MODEL
    # Y_test_pred = clf.predict(X_test)
    # test_accuracy = np.mean(Y_test_pred == Y_test)
    # print(f"Test Accuracy: {test_accuracy:.4f}")