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

drive.mount('/content/drive')

TARGET_DIR = "/content/traffic_light_detection_yolo/dataset"
SAMPLED_DIR = "/content/traffic_light_detection_yolo/sampled_dataset"
SAMPLE_SIZE = 10000
EPOCHS = 50
BATCH_SIZE = 16
WORKERS = 8
IMG_SIZE = (640, 640)
YOLO_MODEL_SIZE = "yolov8l.pt"
STATES = ["red", "yellow", "green"]
CONF_THRESHOLD = 0.2

os.makedirs(os.path.join(SAMPLED_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(SAMPLED_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(SAMPLED_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(SAMPLED_DIR, "labels", "val"), exist_ok=True)

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

print("\nVerifying original dataset structure...")
train_imgs = glob.glob(os.path.join(TARGET_DIR, "images", "train", "*.jpg"))
val_imgs = glob.glob(os.path.join(TARGET_DIR, "images", "val", "*.jpg"))
train_labels = glob.glob(os.path.join(TARGET_DIR, "labels", "train", "*.txt"))
val_labels = glob.glob(os.path.join(TARGET_DIR, "labels", "val", "*.txt"))

print(f"Found {len(train_imgs)} training images and {len(train_labels)} labels")
print(f"Found {len(val_imgs)} validation images and {len(val_labels)} labels")

def get_class_from_label(label_path):
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

def balance_sample(train_imgs, sample_size):
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

    yellow_target = min(len(yellow_imgs), int(sample_size * 0.4))  # 40% yellow images
    remaining = sample_size - yellow_target
    red_target = min(len(red_imgs), int(remaining * 0.5))  # Half of remaining for red
    remaining -= red_target
    green_target = min(len(green_imgs), int(remaining * 0.8))  # Most of remaining for green
    remaining -= green_target
    other_target = min(len(other_imgs), remaining)

    random.seed(42)
    sampled_yellow = random.sample(yellow_imgs, yellow_target) if yellow_target > 0 else []
    sampled_red = random.sample(red_imgs, red_target) if red_target > 0 else []
    sampled_green = random.sample(green_imgs, green_target) if green_target > 0 else []
    sampled_other = random.sample(other_imgs, other_target) if other_target > 0 else []

    balanced_sample = sampled_yellow + sampled_red + sampled_green + sampled_other
    random.shuffle(balanced_sample)  # Shuffle again

    print(f"Created balanced sample with {len(sampled_yellow)} yellow, {len(sampled_red)} red, "
          f"{len(sampled_green)} green, {len(sampled_other)} other images")

    return balanced_sample

def crop_around_traffic_lights(img_path, label_path, output_img_path, output_label_path, 
                               padding_factor=3.0, min_crop_size=224, max_crop_size=640):
    try:
        img = cv.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return False
            
        img_height, img_width = img.shape[:2]
        
        with open(label_path, 'r') as f:
            labels = [line.strip().split() for line in f if line.strip()]
        
        if not labels:
            print(f"Warning: No labels found in {label_path}")
            return False
            
        traffic_light_labels = [label for label in labels if int(label[0]) in [0, 1, 2]]
        
        if not traffic_light_labels:
            return False
            
        areas = []
        for label in traffic_light_labels:
            class_id, x_center, y_center, width, height = map(float, label)
            area = width * height
            areas.append((area, label))
        
        areas.sort(reverse=True)
        
        class_id, x_center, y_center, width, height = map(float, areas[0][1])
        
        x_center_px = int(x_center * img_width)
        y_center_px = int(y_center * img_height)
        width_px = int(width * img_width)
        height_px = int(height * img_height)
        
        crop_width = max(min_crop_size, int(width_px * padding_factor))
        crop_height = max(min_crop_size, int(height_px * padding_factor))
        
        crop_width = min(crop_width, max_crop_size)
        crop_height = min(crop_height, max_crop_size)
        
        x_min = max(0, x_center_px - crop_width // 2)
        y_min = max(0, y_center_px - crop_height // 2)
        
        if x_min + crop_width > img_width:
            x_min = max(0, img_width - crop_width)
        if y_min + crop_height > img_height:
            y_min = max(0, img_height - crop_height)
        
        x_max = min(img_width, x_min + crop_width)
        y_max = min(img_height, y_min + crop_height)
        
        actual_width = x_max - x_min
        actual_height = y_max - y_min
        
        cropped_img = img[y_min:y_max, x_min:x_max]
        
        new_labels = []
        for label in labels:
            class_id, x_center, y_center, width, height = map(float, label)
            
            orig_x_center_px = int(x_center * img_width)
            orig_y_center_px = int(y_center * img_height)
            orig_width_px = int(width * img_width)
            orig_height_px = int(height * img_height)
            
            new_x_center_px = orig_x_center_px - x_min
            new_y_center_px = orig_y_center_px - y_min
            
            if (new_x_center_px + orig_width_px / 2 <= 0 or 
                new_x_center_px - orig_width_px / 2 >= actual_width or
                new_y_center_px + orig_height_px / 2 <= 0 or
                new_y_center_px - orig_height_px / 2 >= actual_height):
                continue
            
            new_x_center = new_x_center_px / actual_width
            new_y_center = new_y_center_px / actual_height
            new_width = min(orig_width_px / actual_width, 1.0)  # Cap at 1.0
            new_height = min(orig_height_px / actual_height, 1.0)  # Cap at 1.0
            
            new_x_center = max(0, min(1, new_x_center))
            new_y_center = max(0, min(1, new_y_center))
            
            new_labels.append(f"{int(class_id)} {new_x_center} {new_y_center} {new_width} {new_height}")
        
        cv.imwrite(output_img_path, cropped_img)
        
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(new_labels))
        
        return True
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


print(f"\nSampling {SAMPLE_SIZE} images with class balance...")
if len(train_imgs) > SAMPLE_SIZE:
    sampled_train_imgs = balance_sample(train_imgs, SAMPLE_SIZE)
else:
    print(f"Warning: Requested {SAMPLE_SIZE} samples but only {len(train_imgs)} are available.")
    sampled_train_imgs = train_imgs

print("Processing and cropping training images around traffic lights...")
successful_crops = 0
failed_crops = 0

for img_path in tqdm(sampled_train_imgs):
    img_filename = os.path.basename(img_path)
    base_name = os.path.splitext(img_filename)[0]
    
    label_path = os.path.join(TARGET_DIR, "labels", "train", f"{base_name}.txt")
    dest_img_path = os.path.join(SAMPLED_DIR, "images", "train", img_filename)
    dest_label_path = os.path.join(SAMPLED_DIR, "labels", "train", f"{base_name}.txt")
    
    if not os.path.exists(label_path):
        failed_crops += 1
        continue
    
    success = crop_around_traffic_lights(
        img_path, 
        label_path, 
        dest_img_path, 
        dest_label_path,
        padding_factor=4.0,
        min_crop_size=320,
        max_crop_size=640
    )
    
    if success:
        successful_crops += 1
    else:
        shutil.copy2(img_path, dest_img_path)
        if os.path.exists(label_path):
            shutil.copy2(label_path, dest_label_path)
        failed_crops += 1

print(f"Training data processing complete: {successful_crops} successful crops, {failed_crops} fallbacks to original images")

print("Processing and cropping validation images around traffic lights...")
val_successful_crops = 0
val_failed_crops = 0

for img_path in tqdm(val_imgs):
    img_filename = os.path.basename(img_path)
    base_name = os.path.splitext(img_filename)[0]
    
    label_path = os.path.join(TARGET_DIR, "labels", "val", f"{base_name}.txt")
    dest_img_path = os.path.join(SAMPLED_DIR, "images", "val", img_filename)
    dest_label_path = os.path.join(SAMPLED_DIR, "labels", "val", f"{base_name}.txt")
    
    if not os.path.exists(label_path):
        val_failed_crops += 1
        continue
    
    success = crop_around_traffic_lights(
        img_path, 
        label_path, 
        dest_img_path, 
        dest_label_path,
        padding_factor=4.0,
        min_crop_size=320,
        max_crop_size=640
    )
    
    if success:
        val_successful_crops += 1
    else:
        shutil.copy2(img_path, dest_img_path)
        if os.path.exists(label_path):
            shutil.copy2(label_path, dest_label_path)
        val_failed_crops += 1

print(f"Validation data processing complete: {val_successful_crops} successful crops, {val_failed_crops} fallbacks to original images")

sampled_train_imgs = glob.glob(os.path.join(SAMPLED_DIR, "images", "train", "*.jpg"))
sampled_val_imgs = glob.glob(os.path.join(SAMPLED_DIR, "images", "val", "*.jpg"))
sampled_train_labels = glob.glob(os.path.join(SAMPLED_DIR, "labels", "train", "*.txt"))
sampled_val_labels = glob.glob(os.path.join(SAMPLED_DIR, "labels", "val", "*.txt"))

print(f"\nSampled dataset created with {len(sampled_train_imgs)} training images and {len(sampled_val_imgs)} validation images")
print(f"Sampled training labels: {len(sampled_train_labels)}")
print(f"Sampled validation labels: {len(sampled_val_labels)}")

dataset_yaml_path = os.path.join(SAMPLED_DIR, "dataset.yaml")
with open(dataset_yaml_path, "w") as f:
    f.write(f"train: ./images/train\n")
    f.write(f"val: ./images/val\n")
    f.write(f"nc: 3\n")
    f.write(f"names: {STATES}\n")

print(f"Created dataset.yaml at {dataset_yaml_path}")

print("\nInitializing YOLOv8 model...")
model = YOLO(YOLO_MODEL_SIZE)

from IPython.display import display, Javascript
def keep_alive():
    display(Javascript('''
    function ClickConnect(){
        console.log("Clicking connect button");
        document.querySelector("colab-connect-button").click()
    }
    setInterval(ClickConnect, 60000)
    '''))

print("\nStarting training on sampled dataset...")
keep_alive()

os.chdir(SAMPLED_DIR)

results = model.train(
    data=dataset_yaml_path,
    epochs=EPOCHS,
    workers=WORKERS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name='yolo_traffic_light_detector',
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

best_model_path = os.path.join("runs", "detect", "yolo_traffic_light_detector", "weights", "best.pt")

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

print("\nEvaluating model performance...")
evaluation_results = evaluate_detection_model(
    model_path=best_model_path,
    data_yaml=dataset_yaml_path,
    conf_threshold=CONF_THRESHOLD,
    iou_threshold=0.5,
    save_dir="/content/detection_metrics"
)

try:
    drive_model_path = "/content/drive/MyDrive/traffic_light_detection_yolo/best_model.pt"
    os.makedirs(os.path.dirname(drive_model_path), exist_ok=True)
    os.system(f"cp {best_model_path} {drive_model_path}")
    print(f"Model saved to Google Drive at: {drive_model_path}")
except Exception as e:
    print(f"Could not save to Drive: {e}")

print("\nDownloading model to your computer...")
files.download(best_model_path)

results_img = f"runs/detect/yolo_traffic_light_detector/results.png"
confusion_matrix = f"runs/detect/yolo_traffic_light_detector/confusion_matrix.png"

if os.path.exists(results_img):
    files.download(results_img)
if os.path.exists(confusion_matrix):
    files.download(confusion_matrix)

print("\nTraining and evaluation complete!")