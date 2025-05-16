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
from PIL import Image

# BDD100K Dataset paths
BASE_DIR = "/kaggle/working/vehicle_pedestrian_detection_yolo"
ORIGINAL_DS_DIR = "/kaggle/input/bdd_dataset100k"  # Base BDD100K directory
IMAGES_DIR = os.path.join(ORIGINAL_DS_DIR, "images", "100k")
TRAIN_IMAGES = os.path.join(IMAGES_DIR, "train")
VAL_IMAGES = os.path.join(IMAGES_DIR, "val")
ANNOTATIONS_FILE = os.path.join(ORIGINAL_DS_DIR, "labels", "bdd100k_labels_images_train.json")
VAL_ANNOTATIONS_FILE = os.path.join(ORIGINAL_DS_DIR, "labels", "bdd100k_labels_images_val.json")

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

EPOCHS = 30
BATCH_SIZE = 32
WORKERS = 8
IMG_SIZE = (640, 640)
YOLO_MODEL_SIZE = "yolov8l.pt"
CONF_THRESHOLD = 0.25

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

def load_bdd100k_annotations(annotation_file):
    print(f"Loading annotations from {annotation_file}")
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        print(f"Loaded annotations for {len(annotations)} images")
        return annotations
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return []

def extract_class_mapping(annotations):
    unique_categories = set()
    
    print("Scanning for unique object categories...")
    for img_anno in tqdm(annotations):
        for label in img_anno.get("labels", []):
            category = label.get("category")
            if category:
                unique_categories.add(category)
    
    sorted_categories = sorted(list(unique_categories))
    category_to_id = {category: i for i, category in enumerate(sorted_categories)}
    
    print(f"Found {len(sorted_categories)} unique categories: {sorted_categories}")
    return category_to_id, sorted_categories

def convert_bdd100k_to_yolo(image_file, annotations, category_to_id):
    img_name = os.path.basename(image_file)
    
    img_anno = None
    for anno in annotations:
        if anno.get("name") == img_name:
            img_anno = anno
            break
    
    if not img_anno:
        return []

    try:
        with Image.open(image_file) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Error reading image {image_file}: {e}")
        return []
    
    yolo_annotations = []
    
    for label in img_anno.get("labels", []):
        category = label.get("category")
        if category not in category_to_id:
            continue
        
        box = label.get("box2d")
        if not box:
            continue
        
        xmin = float(box["x1"])
        ymin = float(box["y1"])
        xmax = float(box["x2"])
        ymax = float(box["y2"])
        
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        if width <= 0 or height <= 0 or not all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
            continue
        
        class_id = category_to_id[category]
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_annotations

def find_all_training_images():
    all_images = []
    
    subdirs = [d for d in os.listdir(TRAIN_IMAGES) if os.path.isdir(os.path.join(TRAIN_IMAGES, d))]
    
    if subdirs and subdirs[0].startswith('train'):
        print(f"Found training subdirectories: {subdirs}")
        for subdir in subdirs:
            subdir_path = os.path.join(TRAIN_IMAGES, subdir)
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                all_images.extend(glob.glob(os.path.join(subdir_path, ext)))
    else:
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            all_images.extend(glob.glob(os.path.join(TRAIN_IMAGES, ext)))
    
    print(f"Found {len(all_images)} training images")
    return all_images

def find_all_validation_images():
    all_images = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        all_images.extend(glob.glob(os.path.join(VAL_IMAGES, ext)))
    
    print(f"Found {len(all_images)} validation images")
    return all_images

def prepare_dataset():
    print("Loading annotations...")
    train_annotations = load_bdd100k_annotations(ANNOTATIONS_FILE)
    val_annotations = load_bdd100k_annotations(VAL_ANNOTATIONS_FILE)
    
    if not train_annotations:
        print("ERROR: No training annotations found!")
        return None
    
    print("Extracting class mapping...")
    category_to_id, class_names = extract_class_mapping(train_annotations + val_annotations)
    
    print("Finding training images...")
    train_images = find_all_training_images()
    
    print("Finding validation images...")
    val_images = find_all_validation_images()
    
    print("Processing training set...")
    train_with_labels = 0
    for img_path in tqdm(train_images):
        img_name = os.path.basename(img_path)
        dest_img = os.path.join(DATASET_DIR, "images", "train", img_name)
        
        shutil.copy2(img_path, dest_img)
        
        yolo_annotations = convert_bdd100k_to_yolo(img_path, train_annotations, category_to_id)
        if yolo_annotations:
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(DATASET_DIR, "labels", "train", label_name)
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_annotations))
            train_with_labels += 1
    
    print("Processing validation set...")
    val_with_labels = 0
    for img_path in tqdm(val_images):
        img_name = os.path.basename(img_path)
        dest_img = os.path.join(DATASET_DIR, "images", "val", img_name)
        
        shutil.copy2(img_path, dest_img)
        
        yolo_annotations = convert_bdd100k_to_yolo(img_path, val_annotations, category_to_id)
        if yolo_annotations:
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(DATASET_DIR, "labels", "val", label_name)
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_annotations))
            val_with_labels += 1
    
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
    print(f"  Training images: {len(train_imgs)} ({train_with_labels} with labels)")
    print(f"  Training labels: {len(train_labels)}")
    print(f"  Validation images: {len(val_imgs)} ({val_with_labels} with labels)")
    print(f"  Validation labels: {len(val_labels)}")
    print(f"  Categories: {len(class_names)}")
    print(f"  Classes: {', '.join(class_names)}")
    
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
    
    print("\nStarting training...")
    os.chdir(DATASET_DIR)
    
    results = model.train(
        data=dataset_yaml_path,
        epochs=EPOCHS,
        workers=WORKERS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='yolo_bdd100k_detector',
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
    best_model_path = os.path.join("runs", "detect", "yolo_bdd100k_detector", "weights", "best.pt")
    
    metrics = model.val(
        data=dataset_yaml_path,
        conf=CONF_THRESHOLD
    )
    
    kaggle_output_path = "/kaggle/working/bdd100k_detector_best.pt"
    
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
    
    # Save the results images
    results_img = f"runs/detect/yolo_bdd100k_detector/results.png"
    confusion_matrix = f"runs/detect/yolo_bdd100k_detector/confusion_matrix.png"
    
    if os.path.exists(results_img):
        shutil.copy2(results_img, "/kaggle/working/results.png")
    if os.path.exists(confusion_matrix):
        shutil.copy2(confusion_matrix, "/kaggle/working/confusion_matrix.png")
    
    print("\nTraining and evaluation complete!")

def evaluate_detection_model(model_path, data_yaml, conf_threshold=0.25, iou_threshold=0.5, save_dir="metrics"):
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
    
    print("\n===== BDD100K DETECTION MODEL EVALUATION =====")
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
    if dataset_yaml_path:
        train_model(dataset_yaml_path)
    else:
        print("Dataset preparation failed. Exiting.")