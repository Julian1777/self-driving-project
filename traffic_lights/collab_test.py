import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import random
from datetime import datetime
import torch
import cv2 as cv
from ultralytics import YOLO
from google.colab import drive, files
import gc
from tqdm.notebook import tqdm

# Mount Google Drive
drive.mount('/content/drive')

# Set paths and parameters
TARGET_DIR = "/content/traffic_light_detection_yolo/dataset"
SAMPLED_DIR = "/content/traffic_light_detection_yolo/sampled_dataset"
SAMPLE_SIZE = 5000  # ⬆️ INCREASED from 1000 to ensure better class representation
EPOCHS = 50  # ⬆️ INCREASED from 30 for better training
BATCH_SIZE = 16  # ⬇️ REDUCED from 32 to improve gradient updates
WORKERS = 8
IMG_SIZE = (640, 640)  # ⬆️ INCREASED from 416 for better small object detection
YOLO_MODEL_SIZE = "yolov8l.pt"  # UPGRADED from medium model
STATES = ["red", "yellow", "green"]
CONF_THRESHOLD = 0.15  # ⬇️ LOWERED from 0.25 to improve recall

# Create directories for sampled dataset
os.makedirs(os.path.join(SAMPLED_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(SAMPLED_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(SAMPLED_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(SAMPLED_DIR, "labels", "val"), exist_ok=True)

# GPU settings
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = '0'
    # ✅ ADDED memory optimization
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("No CUDA devices available, using CPU")
    device = 'cpu'

# Verify original dataset structure
print("\nVerifying original dataset structure...")
train_imgs = glob.glob(os.path.join(TARGET_DIR, "images", "train", "*.jpg"))
val_imgs = glob.glob(os.path.join(TARGET_DIR, "images", "val", "*.jpg"))
train_labels = glob.glob(os.path.join(TARGET_DIR, "labels", "train", "*.txt"))
val_labels = glob.glob(os.path.join(TARGET_DIR, "labels", "val", "*.txt"))

print(f"Found {len(train_imgs)} training images and {len(train_labels)} labels")
print(f"Found {len(val_imgs)} validation images and {len(val_labels)} labels")

# ✅ NEW FUNCTION: Balance training data across classes
def get_class_from_label(label_path):
    """Extract traffic light classes from label file"""
    classes = set()
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    classes.add(class_id)
    except:
        pass
    return classes

# ✅ NEW FUNCTION: Sample balanced dataset
def balance_sample(train_imgs, sample_size):
    """Create a balanced sample across red, yellow, green classes"""
    red_imgs = []
    yellow_imgs = []
    green_imgs = []
    other_imgs = []

    print("Analyzing class distribution...")
    for img_path in tqdm(train_imgs):
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]
        label_path = os.path.join(os.path.dirname(os.path.dirname(img_path)),
                                 "labels", "train", f"{base_name}.txt")

        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            other_imgs.append(img_path)
            continue

        classes = get_class_from_label(label_path)

        if 1 in classes:  # Yellow (priority since it's underrepresented)
            yellow_imgs.append(img_path)
        elif 0 in classes:  # Red
            red_imgs.append(img_path)
        elif 2 in classes:  # Green
            green_imgs.append(img_path)
        else:
            other_imgs.append(img_path)

    print(f"Found {len(red_imgs)} red, {len(yellow_imgs)} yellow, {len(green_imgs)} green, {len(other_imgs)} other images")

    # Calculate sample sizes with priority on yellow
    yellow_target = min(len(yellow_imgs), int(sample_size * 0.4))  # 40% yellow images
    remaining = sample_size - yellow_target
    red_target = min(len(red_imgs), int(remaining * 0.5))  # Half of remaining for red
    remaining -= red_target
    green_target = min(len(green_imgs), int(remaining * 0.8))  # Most of remaining for green
    remaining -= green_target
    other_target = min(len(other_imgs), remaining)

    # Sample with fixed random seed
    random.seed(42)
    sampled_yellow = random.sample(yellow_imgs, yellow_target) if yellow_target > 0 else []
    sampled_red = random.sample(red_imgs, red_target) if red_target > 0 else []
    sampled_green = random.sample(green_imgs, green_target) if green_target > 0 else []
    sampled_other = random.sample(other_imgs, other_target) if other_target > 0 else []

    # Combine samples
    balanced_sample = sampled_yellow + sampled_red + sampled_green + sampled_other
    random.shuffle(balanced_sample)  # Shuffle again

    print(f"Created balanced sample with {len(sampled_yellow)} yellow, {len(sampled_red)} red, "
          f"{len(sampled_green)} green, {len(sampled_other)} other images")

    return balanced_sample

# ✅ MODIFIED: Sample images with class balance
print(f"\nSampling {SAMPLE_SIZE} images with class balance...")
if len(train_imgs) > SAMPLE_SIZE:
    sampled_train_imgs = balance_sample(train_imgs, SAMPLE_SIZE)
else:
    print(f"Warning: Requested {SAMPLE_SIZE} samples but only {len(train_imgs)} are available.")
    sampled_train_imgs = train_imgs

# Copy sampled training images and their labels
print("Copying sampled training images and labels...")
for img_path in tqdm(sampled_train_imgs):
    # Copy image
    img_filename = os.path.basename(img_path)
    dest_img_path = os.path.join(SAMPLED_DIR, "images", "train", img_filename)
    shutil.copy2(img_path, dest_img_path)

    # Copy corresponding label
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    src_label_path = os.path.join(TARGET_DIR, "labels", "train", label_filename)
    if os.path.exists(src_label_path):
        dest_label_path = os.path.join(SAMPLED_DIR, "labels", "train", label_filename)
        shutil.copy2(src_label_path, dest_label_path)

# Copy all validation images and labels
print("Copying validation images and labels...")
for img_path in tqdm(val_imgs):
    # Copy image
    img_filename = os.path.basename(img_path)
    dest_img_path = os.path.join(SAMPLED_DIR, "images", "val", img_filename)
    shutil.copy2(img_path, dest_img_path)

    # Copy corresponding label
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    src_label_path = os.path.join(TARGET_DIR, "labels", "val", label_filename)
    if os.path.exists(src_label_path):
        dest_label_path = os.path.join(SAMPLED_DIR, "labels", "val", label_filename)
        shutil.copy2(src_label_path, dest_label_path)

# Verify sampled dataset
sampled_train_imgs = glob.glob(os.path.join(SAMPLED_DIR, "images", "train", "*.jpg"))
sampled_val_imgs = glob.glob(os.path.join(SAMPLED_DIR, "images", "val", "*.jpg"))
sampled_train_labels = glob.glob(os.path.join(SAMPLED_DIR, "labels", "train", "*.txt"))
sampled_val_labels = glob.glob(os.path.join(SAMPLED_DIR, "labels", "val", "*.txt"))

print(f"\nSampled dataset created with {len(sampled_train_imgs)} training images and {len(sampled_val_imgs)} validation images")
print(f"Sampled training labels: {len(sampled_train_labels)}")
print(f"Sampled validation labels: {len(sampled_val_labels)}")

# Create dataset.yaml for the sampled dataset
dataset_yaml_path = os.path.join(SAMPLED_DIR, "dataset.yaml")
with open(dataset_yaml_path, "w") as f:
    f.write(f"train: ./images/train\n")
    f.write(f"val: ./images/val\n")
    f.write(f"nc: 3\n")
    f.write(f"names: {STATES}\n")

print(f"Created dataset.yaml at {dataset_yaml_path}")

# Initialize YOLOv8 model
print("\nInitializing YOLOv8 model...")
model = YOLO(YOLO_MODEL_SIZE)

# Keep Colab from disconnecting
from IPython.display import display, Javascript
def keep_alive():
    display(Javascript('''
    function ClickConnect(){
        console.log("Clicking connect button");
        document.querySelector("colab-connect-button").click()
    }
    setInterval(ClickConnect, 60000)
    '''))

# Start training with the sampled dataset
print("\nStarting training on sampled dataset...")
keep_alive()

# Change working directory to sampled dataset
os.chdir(SAMPLED_DIR)

# ✅ IMPROVED training parameters
results = model.train(
    data=dataset_yaml_path,
    epochs=EPOCHS,
    workers=WORKERS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name='yolo_traffic_light_detector',
    patience=15,  # More patience before stopping
    save=True,
    device=device,
    cache=False,
    amp=True,
    rect=True,  # ✅ ADDED for better batch efficiency
    plots=True,
    augment=True,  # ✅ ENABLED augmentation (was False before)
    close_mosaic=10,
    overlap_mask=True,  # ✅ ADDED for better mask overlaps
    cos_lr=True,  # ✅ ADDED cosine learning rate
    pretrained=True,
    seed=42,
    profile=True,
    verbose=True,
    mosaic=0.8,  # ✅ ADDED explicit mosaic augmentation
    mixup=0.1,  # ✅ ADDED mixup augmentation
    copy_paste=0.1,  # ✅ ADDED copy-paste augmentation
    degrees=10.0,  # ✅ ADDED rotation augmentation
    scale=0.5,  # ✅ ADDED scale augmentation
    save_period=10
)

# Validate the model with lower confidence threshold to improve recall
print("\nValidating trained model...")
metrics = model.val(
    data=dataset_yaml_path,
    conf=CONF_THRESHOLD  # ⬇️ LOWERED from 0.25
)

# Get best model path
best_model_path = os.path.join("runs", "detect", "yolo_traffic_light_detector", "weights", "best.pt")

# ✅ FIXED F1 calculation
def evaluate_detection_model(model_path, data_yaml, conf_threshold=CONF_THRESHOLD, iou_threshold=0.5, save_dir="metrics"):
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
        # ✅ FIXED F1 calculation to use the proper precision/recall formula
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

# Evaluate model performance with lower confidence threshold
print("\nEvaluating model performance...")
evaluation_results = evaluate_detection_model(
    model_path=best_model_path,
    data_yaml=dataset_yaml_path,
    conf_threshold=CONF_THRESHOLD,  # ⬇️ LOWERED
    iou_threshold=0.5,
    save_dir="/content/detection_metrics"
)

# Save to Google Drive
try:
    drive_model_path = "/content/drive/MyDrive/traffic_light_detection_yolo/best_model.pt"
    os.makedirs(os.path.dirname(drive_model_path), exist_ok=True)
    os.system(f"cp {best_model_path} {drive_model_path}")
    print(f"Model saved to Google Drive at: {drive_model_path}")
except Exception as e:
    print(f"Could not save to Drive: {e}")

# Download model and results
print("\nDownloading model to your computer...")
files.download(best_model_path)

results_img = f"runs/detect/yolo_traffic_light_detector/results.png"
confusion_matrix = f"runs/detect/yolo_traffic_light_detector/confusion_matrix.png"

if os.path.exists(results_img):
    files.download(results_img)
if os.path.exists(confusion_matrix):
    files.download(confusion_matrix)

print("\nTraining and evaluation complete!")