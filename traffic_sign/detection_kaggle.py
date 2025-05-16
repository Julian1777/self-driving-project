import os
import glob
import json
import shutil
import random
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from datetime import datetime

BASE_DIR = "/kaggle/working/traffic_sign_detection_yolo"
ORIGINAL_DS_DIR = "/kaggle/input/mapillary-sign-dataset"
DATASET_DIR = f"{BASE_DIR}/dataset"
ANNOTATIONS_DIR = f"{ORIGINAL_DS_DIR}/annotations"  # Directory containing JSON annotations
IMAGES_DIR = f"{ORIGINAL_DS_DIR}/images"  # Directory containing images
TRAIN_RATIO = 0.9
EPOCHS = 30
BATCH_SIZE = 16
WORKERS = 8
IMG_SIZE = (640, 640)
YOLO_MODEL_SIZE = "yolov8l.pt"
CONF_THRESHOLD = 0.2

os.makedirs(os.path.join(DATASET_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "labels", "val"), exist_ok=True)

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = '0'
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("No CUDA devices available, using CPU")
    device = 'cpu'

def load_class_mapping():
    """Create a mapping from sign labels to class IDs"""
    unique_labels = set()

    print("Scanning for unique labels...")
    json_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json"))
    for json_file in tqdm(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
            for obj in data.get("objects", []):
                label = obj.get("label")
                if label and label != "other-sign":
                    unique_labels.add(label)

    sorted_labels = sorted(list(unique_labels))
    label_to_id = {label: i for i, label in enumerate(sorted_labels)}

    print(f"Found {len(sorted_labels)} unique traffic sign classes")
    return label_to_id, sorted_labels

def convert_annotations(json_file, label_to_id):
    """Convert a JSON annotation to YOLO format"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        img_width = data.get("width", 0)
        img_height = data.get("height", 0)

        if img_width <= 0 or img_height <= 0:
            return None

        yolo_annotations = []

        for obj in data.get("objects", []):
            label = obj.get("label")
            if label == "other-sign" or label not in label_to_id:
                continue

            bbox = obj.get("bbox", {})
            if not all(k in bbox for k in ["xmin", "ymin", "xmax", "ymax"]):
                continue

            xmin = bbox["xmin"]
            ymin = bbox["ymin"]
            xmax = bbox["xmax"]
            ymax = bbox["ymax"]

            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            if not all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                continue

            class_id = label_to_id[label]
            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

        return yolo_annotations
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None

def prepare_dataset():
    print("Loading class mappings...")
    label_to_id, class_names = load_class_mapping()

    print("Finding all annotation files...")
    json_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json"))

    file_pairs = []
    for json_file in json_files:
        base_name = os.path.basename(json_file)
        image_id = os.path.splitext(base_name)[0]

        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(IMAGES_DIR, f"{image_id}{ext}")
            if os.path.exists(img_path):
                file_pairs.append((json_file, img_path))
                break

    print(f"Found {len(file_pairs)} image-annotation pairs")

    random.seed(42)
    random.shuffle(file_pairs)

    split_idx = int(len(file_pairs) * TRAIN_RATIO)
    train_pairs = file_pairs[:split_idx]
    val_pairs = file_pairs[split_idx:]

    print(f"Split dataset: {len(train_pairs)} training, {len(val_pairs)} validation")

    print("Processing training set...")
    for json_file, img_path in tqdm(train_pairs):
        base_name = os.path.basename(json_file)
        image_id = os.path.splitext(base_name)[0]

        img_ext = os.path.splitext(img_path)[1]
        dest_img = os.path.join(DATASET_DIR, "images", "train", f"{image_id}{img_ext}")
        shutil.copy2(img_path, dest_img)

        yolo_annotations = convert_annotations(json_file, label_to_id)
        if yolo_annotations:
            label_path = os.path.join(DATASET_DIR, "labels", "train", f"{image_id}.txt")
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_annotations))

    print("Processing validation set...")
    for json_file, img_path in tqdm(val_pairs):
        base_name = os.path.basename(json_file)
        image_id = os.path.splitext(base_name)[0]

        img_ext = os.path.splitext(img_path)[1]
        dest_img = os.path.join(DATASET_DIR, "images", "val", f"{image_id}{img_ext}")
        shutil.copy2(img_path, dest_img)

        yolo_annotations = convert_annotations(json_file, label_to_id)
        if yolo_annotations:
            label_path = os.path.join(DATASET_DIR, "labels", "val", f"{image_id}.txt")
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_annotations))

    print("Creating dataset.yaml...")
    dataset_yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    with open(dataset_yaml_path, "w") as f:
        f.write(f"train: ./images/train\n")
        f.write(f"val: ./images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    train_imgs = glob.glob(os.path.join(DATASET_DIR, "images", "train", "*.*"))
    val_imgs = glob.glob(os.path.join(DATASET_DIR, "images", "val", "*.*"))
    train_labels = glob.glob(os.path.join(DATASET_DIR, "labels", "train", "*.txt"))
    val_labels = glob.glob(os.path.join(DATASET_DIR, "labels", "val", "*.txt"))

    print("\nDataset prepared:")
    print(f"  Training images: {len(train_imgs)}")
    print(f"  Training labels: {len(train_labels)}")
    print(f"  Validation images: {len(val_imgs)}")
    print(f"  Validation labels: {len(val_labels)}")
    print(f"  Classes: {len(class_names)}")

    return dataset_yaml_path

def train_model(dataset_yaml_path):
    print("\nInitializing YOLOv8 model...")
    try:
        model = YOLO(YOLO_MODEL_SIZE)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Downloading model using torch hub...")
        torch.hub.load('ultralytics/yolov8', 'yolov8l', pretrained=True)
        model = YOLO(YOLO_MODEL_SIZE)

    os.chdir(DATASET_DIR)

    results = model.train(
        data=dataset_yaml_path,
        epochs=EPOCHS,
        workers=WORKERS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='yolo_traffic_sign_detector',
        patience=15,
        save=True,
        device=device,
        cache=False,
        amp=True,
        rect=True,
        plots=True,
        augment=True,
        close_mosaic=10,
        overlap_mask=True,
        cos_lr=True,
        pretrained=True,
        seed=42,
        profile=True,
        verbose=True,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        scale=0.5,
        save_period=10
    )

    print("\nValidating trained model...")
    metrics = model.val(
        data=dataset_yaml_path,
        conf=CONF_THRESHOLD
    )

    best_model_path = os.path.join("runs", "detect", "yolo_traffic_sign_detector", "weights", "best.pt")
    kaggle_output_path = "/kaggle/working/best_model.pt"

    print("\nEvaluating model performance...")
    evaluate_detection_model(
        model_path=best_model_path,
        data_yaml=dataset_yaml_path,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=0.5,
        save_dir="/kaggle/working/detection_metrics"
    )

    shutil.copy2(best_model_path, kaggle_output_path)
    print(f"Model saved to Kaggle working directory: {kaggle_output_path}")
    
    # Also save the results images
    results_img = f"runs/detect/yolo_traffic_sign_detector/results.png"
    confusion_matrix = f"runs/detect/yolo_traffic_sign_detector/confusion_matrix.png"
    
    if os.path.exists(results_img):
        shutil.copy2(results_img, "/kaggle/working/results.png")
    if os.path.exists(confusion_matrix):
        shutil.copy2(confusion_matrix, "/kaggle/working/confusion_matrix.png")

    print("\nTraining and evaluation complete!")

def evaluate_detection_model(model_path, data_yaml, conf_threshold=0.15, iou_threshold=0.5, save_dir="metrics"):
    os.makedirs(save_dir, exist_ok=True)

    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=True,
        save_json=True,
        save_hybrid=True,
        plots=True
    )

    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-10)),
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    per_class_ap = {}
    if hasattr(metrics.box, 'ap_class_index') and hasattr(metrics.box, 'ap50'):
        class_names = model.names
        for i, class_idx in enumerate(metrics.box.ap_class_index):
            class_name = class_names[int(class_idx)]
            per_class_ap[class_name] = float(metrics.box.ap50[i])
        results["per_class_ap50"] = per_class_ap

    with open(os.path.join(save_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), 'w') as f:
        json.dump(results, f, indent=4)

    print("\n===== DETECTION MODEL EVALUATION =====")
    print(f"mAP@0.5: {results['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP50-95']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

    if "per_class_ap50" in results:
        print("\nPer-class AP@0.5:")
        for class_name, ap in results["per_class_ap50"].items():
            print(f"  {class_name}: {ap:.4f}")

    return results

if __name__ == "__main__":
    dataset_yaml_path = prepare_dataset()
    train_model(dataset_yaml_path)