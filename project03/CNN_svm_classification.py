import os
import cv2
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from preprocessing import read_json_data, preprocess_one_image
from segmentation_anything import segment_one_image
from feature_extraction_sift import extract_sift_features_one_image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from segment_anything import sam_model_registry, SamPredictor
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from constants import (
    IMAGES_DIR,
    RAW_DIR,
    CROPPED_DIR, 
    PREPROCESSED_DIR,
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
)

class HoldPreprocessedDataset(Dataset):
    def __init__(self, samples, use_segmentation=False, predictor=None):
        self.samples = samples
        self.use_segmentation = use_segmentation
        self.predictor = predictor
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = sample["img"]
        metadata = sample["metadata"]
        label = sample["label_value"]

        if self.use_segmentation:
            img, _ = segment_one_image(
                img,
                self.predictor,
                metadata["new_h"],
                metadata["new_w"],
                metadata["y_offset"],
                metadata["x_offset"],
            )

        # OpenCV is BGR; torchvision expects RGB-like ordering
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ToTensor gives float tensor in [0,1], shape [C,H,W]
        img_tensor = self.to_tensor(img)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv4_drop = nn.Dropout2d(p=0.25)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = self.conv4_drop(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)
    


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
    left, top, width, height = pad_and_clamp_bbox(left, top, width, height, pad, raw_w, raw_h)
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
    
    cropped = np.array(cropped)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    return cropped, out

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
        "canvas_size": size,
        "new_h": new_h,
        "new_w":new_w,
        "y_offset": y_offset,
        "x_offset": x_offset,
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
    return roi, result_mask

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


# Get CNN features portion
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

def train_model(model, train_loader, val_loader, device, epochs=10):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    train_losses = []
    train_accs = []
    val_accs = []
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model.forward(images)                     # returns (batch_size, 10) holding the output prob for every class in every image
                predictions = torch.argmax(outputs, dim=1)          # grabs the max prob reducing along each row to get (batch_size, 1)
                correct += torch.sum(predictions == labels).float() # sums the correct predictions and turns to float for calculation later
                total += images.shape[0]                             # grabs the total number of images in this batch

        val_acc = 100. * correct / total

        train_losses.append(epoch_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, '
                f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    return train_losses, train_accs, val_accs


def run_svm_classification(X_train, Y_train, X_val, Y_val, kernel="rbf"):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)

    # Change to linear to see other results
    clf = svm.SVC(kernel=kernel, C=1.0, gamma='scale')
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_val)

    accuracy = np.mean(Y_pred == Y_val)
    print(f"SVM Classification Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(Y_val, Y_pred)}")

    # UNCOMMENT ONLY WHEN DOING FINAL TEST ON CHOSEN MODEL
    # Y_test_pred = clf.predict(X_test)
    # test_accuracy = np.mean(Y_test_pred == Y_test)
    # print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    num_classes = 7
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    data_path = os.path.join(RAW_DIR, "annotations.json")
    data = read_json_data(data_path)

    os.makedirs(os.path.join(CROPPED_DIR, IMAGES_DIR), exist_ok=True)
    os.makedirs(os.path.join(PREPROCESSED_DIR, IMAGES_DIR), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessed_data = []


    is_sam_model = args.model in ["1", "2"]  
    predictor = None
    if is_sam_model:
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(device=device)
        predictor = SamPredictor(sam)

    for record in data:
        img_src = os.path.join(RAW_DIR, IMAGES_DIR, record["filename"])

        img, metadata = preprocess_one_image(
            img_src=img_src,
            bbox_left=int(record["bbox_left"]),
            bbox_top=int(record["bbox_top"]),
            bbox_height=int(record["bbox_height"]),
            bbox_width=int(record["bbox_width"]),
            rotation=int(record["exif_rotation"]),
            label_name=record["label_name"],
            label_value=int(record["label_value"])
        )

        preprocessed_data.append({
            "img": img,
            "metadata": metadata,
            "label_value": metadata["label_value"]
        })

    print("Preprocessed data")

    labels = [sample["label_value"] for sample in preprocessed_data]
    train_samples, temp_samples = train_test_split(
        preprocessed_data,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )
    temp_labels = [sample["label_value"] for sample in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels
    )

    train_dataset = HoldPreprocessedDataset(
        samples=train_samples,
        use_segmentation=is_sam_model,
        predictor=predictor
    )
    val_dataset = HoldPreprocessedDataset(
        samples=val_samples,
        use_segmentation=is_sam_model,
        predictor=predictor
    )
    test_dataset = HoldPreprocessedDataset(
        samples=test_samples,
        use_segmentation=is_sam_model,
        predictor=predictor
    )
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Created datasets and sampled")

    # SVM based classification
    if args.model == "1":
        model = build_feature_extractor_RESNET()
        X_train, Y_train = get_cnn_features_SVM(model, train_loader, device=device)
        X_val, Y_val = get_cnn_features_SVM(model, val_loader, device=device)

        clf, Y_pred = run_svm_classification(X_train, Y_train, X_val, Y_val, kernel="linear")
    
    if args.model == "2":
        model = build_feature_extractor_RESNET()
        X_train, Y_train = get_cnn_features_SVM(model, train_loader, device=device)
        X_val, Y_val = get_cnn_features_SVM(model, val_loader, device=device)

        clf, Y_pred = run_svm_classification(X_train, Y_train, X_val, Y_val, kernel="rbf")


    if args.model == "3":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        train_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, device, epochs=10)
