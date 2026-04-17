import os
import cv2
import numpy as np
from preprocessing import read_json_data, preprocess_one_image
from segmentation_anything import segment_one_image
from feature_extraction_sift import extract_sift_features_one_image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from segment_anything import sam_model_registry, SamPredictor
from constants import (
    IMAGES_DIR,
    RAW_DIR,
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
    SIFT_FEATURES_DIR
)




def run_svm_classification():
    feature_vectors_X = []
    feature_vectors_Y = []
    data_path = os.path.join(RAW_DIR, "annotations.json")

    device = "cuda"
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    pad = 5

    data = read_json_data(data_path)
    for record in data:
        img_path = os.path.join(RAW_DIR, IMAGES_DIR, record["filename"])
        bboxLeft = int(record["bbox_left"])
        bboxTop = int(record["bbox_top"])
        bboxWidth = int(record["bbox_width"])
        bboxHeight = int(record["bbox_height"])
        rotation = int(record["exif_rotation"])
        labelValue = int(record["label_value"])

        preprocessed_img_name, preprocessed_img = preprocess_one_image(img_path, bboxLeft, bboxTop, bboxWidth, bboxHeight, rotation, labelValue)
        segmented_img, _, local_feature = segment_one_image(preprocessed_img_name, predictor, pad)
        _, _, sift_feature = extract_sift_features_one_image(segmented_img)

        feature_vectors_X.append(np.concatenate([local_feature, sift_feature]))
        feature_vectors_Y.append(labelValue)

        print(f"Processed {record['filename']} with label {labelValue}")

    X = np.vstack(feature_vectors_X)
    Y = np.array(feature_vectors_Y, dtype=np.int32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y,
        test_size=0.2,
        random_state=42,
        stratify=Y   # important if classes are imbalanced
    )

    # Change to linear to see other results
    clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    out = []
    for i in range(len(Y_test)):
        out.append(f"True: {Y_test[i]} | Pred: {Y_pred[i]}")
    print(out)

    accuracy = np.mean(Y_pred == Y_test)
    print(f"SVM Classification Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")

    return clf, scaler, X, Y







if __name__ == "__main__":
    run_svm_classification()