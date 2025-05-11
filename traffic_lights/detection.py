import os
import pandas as pd
import shutil
import random
import glob
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import json
import csv
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import torch
import cv2 as cv


torch.backends.cudnn.benchmark = True



SOURCE_DIR = "dataset"
TARGET_DIR = "yolo_dataset"
TRAIN_RATIO = 0.9
EPOCHS = 20
BATCH_SIZE = 16
WORKERS = 0
GRID_SIZE = 7
BOXES_PER_CELL = 2
IMG_SIZE = (416,416)
YOLO_MODEL_SIZE = "yolov8m.pt"
DTLD_DIR = "dtld_dataset"
LISA_DIR = "lisa_dataset"
STATES = ["red", "yellow", "green"]
ANNOTATION = os.path.join(DTLD_DIR, "Berlin.json")
sequences = ["daySequence1", "daySequence2", "dayTrain", "nightSequence1", "nightSequence2", "nightTrain"]
DTLD_CITIES = ["Berlin", "Bochum", "Dortmund", "Bremen", "Koeln"]

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.2),
])

def prepare_yolo_dataset():
    print("Preparing YOLO dataset from LISA and DTLD datasets...")
    
    temp_dir = os.path.join(TARGET_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    collected_images = []
    
    if os.path.exists(DTLD_DIR):
        print(f"Collecting images from DTLD dataset...")
        dtld_images = []
        
        # Process all cities instead of just Berlin
        city_found = False
        for city in DTLD_CITIES:
            city_annotation = os.path.join(DTLD_DIR, f"{city}.json")
            if os.path.exists(city_annotation):
                city_found = True
                print(f"Processing city: {city}")
                try:
                    with open(city_annotation, 'r') as f:
                        data = json.load(f)
                    
                    if "images" in data and isinstance(data["images"], list):
                        for image_entry in tqdm(data["images"], desc=f"Finding {city} images"):
                            rel_path = image_entry.get("image_path", "")
                            if rel_path:
                                img_path = os.path.join(DTLD_DIR, rel_path)
                                if os.path.exists(img_path):
                                    dtld_images.append(img_path)
                except Exception as e:
                    print(f"Error loading DTLD {city} annotations: {e}")
        
        if not city_found:
            print("No city annotation files found. Searching for images recursively in DTLD directory...")
            for ext in ['.jpg', '.jpeg', '.png']:
                dtld_images.extend(glob.glob(os.path.join(DTLD_DIR, '**', f'*{ext}'), recursive=True))
        
        print(f"Found {len(dtld_images)} images in DTLD dataset")
        collected_images.extend(dtld_images)
    
    if os.path.exists(LISA_DIR):
        print(f"Collecting images from LISA dataset...")
        lisa_images = []
        
        for seq in sequences:
            seq_dir = os.path.join(LISA_DIR, seq)
            if os.path.exists(seq_dir):
                for ext in ['.jpg', '.jpeg', '.png']:
                    lisa_images.extend(glob.glob(os.path.join(seq_dir, f'*{ext}')))
                
                frames_dir = os.path.join(seq_dir, "frames")
                if os.path.exists(frames_dir):
                    for ext in ['.jpg', '.jpeg', '.png']:
                        lisa_images.extend(glob.glob(os.path.join(frames_dir, f'*{ext}')))
                
                nested_dir = os.path.join(seq_dir, seq)
                if os.path.exists(nested_dir):
                    for ext in ['.jpg', '.jpeg', '.png']:
                        lisa_images.extend(glob.glob(os.path.join(nested_dir, f'*{ext}')))
                    
                    nested_frames = os.path.join(nested_dir, "frames")
                    if os.path.exists(nested_frames):
                        for ext in ['.jpg', '.jpeg', '.png']:
                            lisa_images.extend(glob.glob(os.path.join(nested_frames, f'*{ext}')))
        
        print(f"Found {len(lisa_images)} images in LISA dataset")
        collected_images.extend(lisa_images)
    
    collected_images = list(set(collected_images))
    print(f"Total unique images collected: {len(collected_images)}")
    
    print("Copying images to temporary directory...")
    image_mapping = {}
    
    for img_path in tqdm(collected_images):
        original_name = os.path.basename(img_path)
        
        if img_path.startswith(DTLD_DIR):
            dataset_prefix = "dtld_"
            try:
                pil_image = Image.open(img_path)
                np_img = np.array(pil_image)
                
                if np_img.dtype == np.uint16:
                    np_img = ((np_img - np_img.min()) * 255.0 / (np_img.max() - np_img.min())).astype(np.uint8)
                
                if len(np_img.shape) == 2:
                    np_img = np.stack([np_img] * 3, axis=-1)

                base_name, _ = os.path.splitext(original_name)    
                dest_path = os.path.join(temp_dir, f"{dataset_prefix}{base_name}.jpg")
                cv.imwrite(dest_path, np_img)
                image_mapping[f"{dataset_prefix}{base_name}.jpg"] = img_path
                continue
            except Exception as e:
                print(f"Error processing DTLD image {img_path}: {e}")
        else:
            dataset_prefix = "lisa_"
            
        dest_path = os.path.join(temp_dir, original_name)
        
        if os.path.exists(dest_path):
            prefixed_name = f"{dataset_prefix}{original_name}"
            dest_path = os.path.join(temp_dir, prefixed_name)
            
            counter = 1
            while os.path.exists(dest_path):
                base, ext = os.path.splitext(original_name)
                prefixed_name = f"{dataset_prefix}{base}_{counter}{ext}"
                dest_path = os.path.join(temp_dir, prefixed_name)
                counter += 1
                
            used_name = prefixed_name
        else:
            used_name = original_name
            
        shutil.copy2(img_path, dest_path)
        image_mapping[used_name] = img_path
    
    temp_images = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                  if os.path.isfile(os.path.join(temp_dir, f)) and 
                  os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    
    random.seed(42)  # For reproducibility
    random.shuffle(temp_images)
    
    split_idx = int(len(temp_images) * TRAIN_RATIO)
    train_images = temp_images[:split_idx]
    val_images = temp_images[split_idx:]
    
    print(f"Splitting into {len(train_images)} training and {len(val_images)} validation images")
    
    print("Copying to train directory...")
    for img_path in tqdm(train_images):
        file_name = os.path.basename(img_path)
        dest_path = os.path.join(TARGET_DIR, "images", "train", file_name)
        shutil.copy2(img_path, dest_path)
    
    print("Copying to validation directory...")
    for img_path in tqdm(val_images):
        file_name = os.path.basename(img_path)
        dest_path = os.path.join(TARGET_DIR, "images", "val", file_name)
        shutil.copy2(img_path, dest_path)
    
    dataset_yaml = {
        'train': os.path.join(TARGET_DIR, "images", "train"),
        'val': os.path.join(TARGET_DIR, "images", "val"),
        'nc': 1,  # Number of classes
        'names': ['traffic_light']
    }
    
    with open(os.path.join(TARGET_DIR, "dataset.yaml"), 'w') as f:
        for key, value in dataset_yaml.items():
            f.write(f"{key}: {value}\n")
    
    print("Copying annotation files...")
    
    lisa_annotations_dir = os.path.join(LISA_DIR, "Annotations")
    if os.path.exists(lisa_annotations_dir):
        print("Copying LISA annotations directory...")
        yolo_lisa_annotations_dir = os.path.join(TARGET_DIR, "Annotations", "LISA")
        
        if os.path.exists(yolo_lisa_annotations_dir):
            shutil.rmtree(yolo_lisa_annotations_dir)
            
        shutil.copytree(lisa_annotations_dir, yolo_lisa_annotations_dir)
        print(f"LISA annotations copied to {yolo_lisa_annotations_dir}")
    else:
        print(f"Warning: LISA annotations directory not found at {lisa_annotations_dir}")
    
    dtld_annotations_dir = os.path.join(TARGET_DIR, "Annotations", "DTLD")
    os.makedirs(dtld_annotations_dir, exist_ok=True)
    
    for city in DTLD_CITIES:
        city_annotation = os.path.join(DTLD_DIR, f"{city}.json")
        if os.path.exists(city_annotation):
            dest_path = os.path.join(dtld_annotations_dir, f"{city}.json")
            shutil.copy2(city_annotation, dest_path)
            print(f"DTLD {city}.json copied to {dest_path}")
        else:
            print(f"Warning: DTLD annotation file not found at {city_annotation}")
    
    print("Dataset preparation complete!")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    print("Cleaning up temporary directory...")
    shutil.rmtree(temp_dir)

def generate_yolo_labels():                
    print("Generating YOLO format labels...")
    
    dtld_annotations = {}
    dtld_annotations_dir = os.path.join(TARGET_DIR, "Annotations", "DTLD")

    for city in DTLD_CITIES:
        city_annotation = os.path.join(dtld_annotations_dir, f"{city}.json")
        if os.path.exists(city_annotation):
            with open(city_annotation, 'r') as f:
                data = json.load(f)
                if "images" in data and len(data["images"]) > 0:
                    print("SAMPLE JSON ENTRY:")
                    print(f"Image path: {data['images'][0].get('image_path', 'unknown')}")
                    print(f"File basename: {os.path.basename(data['images'][0].get('image_path', 'unknown'))}")
                    if "labels" in data["images"][0]:
                        print(f"Sample label: {data['images'][0]['labels'][0]}")
                    break
    
    if os.path.exists(dtld_annotations_dir):
        print("Processing DTLD annotations...")
        
        for city in DTLD_CITIES:
            city_annotation = os.path.join(dtld_annotations_dir, f"{city}.json")
            if os.path.exists(city_annotation):
                print(f"Processing {city} annotations...")
                try:
                    with open(city_annotation, 'r') as f:
                        data = json.load(f)
                        
                    # Insert this debug code here:
                    total_images = 0
                    total_traffic_lights = 0
                    failed_images = 0

                    if "images" in data and isinstance(data["images"], list):
                        for image_entry in tqdm(data["images"], desc=f"Loading {city} annotations"):
                            total_images += 1
                            rel_path = image_entry.get("image_path", "")
                            if not rel_path:
                                continue
                                
                            img_filename = os.path.basename(rel_path)
                            traffic_lights = []
                            
                            img_width = image_entry.get("width", 0)
                            img_height = image_entry.get("height", 0)
                            
                            if img_width <= 0 or img_height <= 0:
                                try:
                                    # Construct the full path to the image
                                    img_path = os.path.join(DTLD_DIR, rel_path)
                                    if os.path.exists(img_path):
                                        with Image.open(img_path) as img:
                                            img_width, img_height = img.size
                                            print(f"Loaded dimensions for {img_filename}: {img_width}x{img_height}")
                                    else:
                                        # Try alternate path formats
                                        base_dir = os.path.dirname(rel_path)
                                        alt_path = os.path.join(DTLD_DIR, base_dir, img_filename)
                                        if os.path.exists(alt_path):
                                            with Image.open(alt_path) as img:
                                                img_width, img_height = img.size
                                except Exception as e:
                                    print(f"Error opening image {img_filename}: {e}")

                            if img_width <= 0 or img_height <= 0:
                                failed_images += 1
                                print(f"Still invalid dimensions for {img_filename}: {img_width}x{img_height}")
                                continue
                            
                            label_counter = 0
                            for label in image_entry.get("labels", []):
                                label_counter += 1
                                attr = label.get("attributes", {})
                                state = attr.get("state", "")
                                
                                if state == "red":
                                    class_id = 0
                                elif state == "yellow" or state == "amber":
                                    class_id = 1
                                elif state == "green":
                                    class_id = 2
                                else:
                                    class_id = 0
                                
                                x = label.get("x", 0)
                                y = label.get("y", 0)
                                w = label.get("w", 0)
                                h = label.get("h", 0)
                                
                                if w > 0 and h > 0:
                                    x_center = (x + w/2) / img_width
                                    y_center = (y + h/2) / img_height
                                    width = w / img_width
                                    height = h / img_height
                                    
                                    traffic_lights.append((x_center, y_center, width, height, class_id))
                            
                            if not traffic_lights:
                                print(f"Image {img_filename} has {label_counter} labels but no valid traffic lights")
                            else:
                                total_traffic_lights += len(traffic_lights)
                                
                                dtld_annotations[img_filename] = traffic_lights
                                
                except Exception as e:
                    print(f"Error processing {city} annotations: {e}")
                    
    print(f"{city}: Processed {total_images} images, found {total_traffic_lights} traffic lights, failed {failed_images} images")
        
    lisa_annotations = {}
    lisa_annotations_dir = os.path.join(TARGET_DIR, "Annotations", "LISA", "Annotations")
    if os.path.exists(lisa_annotations_dir):
        print("Processing LISA annotations...")
        
        for seq in os.listdir(lisa_annotations_dir):
            seq_dir = os.path.join(lisa_annotations_dir, seq)
            if not os.path.isdir(seq_dir):
                continue
                
            for ann_file in ["frameAnnotationsBOX.csv", "frameAnnotationsBULB.csv"]:
                ann_path = os.path.join(seq_dir, ann_file)
                if not os.path.exists(ann_path):
                    continue
                    
                try:
                    with open(ann_path, 'r') as f:
                        reader = csv.reader(f, delimiter=';')
                        next(reader)  # Skip header
                        
                        for row in tqdm(reader, desc=f"Processing {seq}/{ann_file}"):
                            if len(row) >= 6:
                                image_path = row[0]
                                image_name = os.path.basename(image_path)

                                light_type = row[1].lower() if len(row) > 1 else ""

                                if "red" in light_type:
                                    class_id = 0
                                elif "yellow" in light_type or "amber" in light_type:
                                    class_id = 1
                                elif "green" in light_type or "go" in light_type:
                                    class_id = 2
                                else:
                                    class_id = 0
                                
                                try:
                                    x1 = float(row[2])
                                    y1 = float(row[3])
                                    x2 = float(row[4])
                                    y2 = float(row[5])
                                    
                                    if image_name not in lisa_annotations:
                                        lisa_annotations[image_name] = []
                                        
                                    lisa_annotations[image_name].append((x1, y1, x2, y2, class_id))
                                except:
                                    continue
                except Exception as e:
                    print(f"Error processing {ann_path}: {e}")
    
        print(f"Loaded annotations for {len(lisa_annotations)} LISA images")
    
    generate_labels_for_dir("train", dtld_annotations, lisa_annotations)
    
    generate_labels_for_dir("val", dtld_annotations, lisa_annotations)
    
    print("YOLO label generation complete!")

def generate_labels_for_dir(split, dtld_annotations, lisa_annotations):
    print(f"Sample DTLD annotation keys: {list(dtld_annotations.keys())[:5]}")

    image_dir = os.path.join(TARGET_DIR, "images", split)
    label_dir = os.path.join(TARGET_DIR, "labels", split)
    
    if not os.path.exists(image_dir):
        print(f"Image directory {image_dir} not found")
        return
        
    os.makedirs(label_dir, exist_ok=True)
    
    print(f"Generating labels for {split} images...")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    total = len(image_files)
    matched = 0
    unmatched = 0

    print(f"Sample DTLD image filenames: {[f for f in image_files if f.startswith('dtld_')][:5]}")

    
    for img_file in tqdm(image_files, desc=f"Processing {split} images"):
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        img_path = os.path.join(image_dir, img_file)
        
        original_name = find_original_image_name(img_path, img_file, dtld_annotations)
        
        boxes = None
        img_width, img_height = None, None
        
        if original_name in dtld_annotations:
            boxes = dtld_annotations[original_name]
        
        elif original_name in lisa_annotations:
            unnormalized_boxes = lisa_annotations[original_name]
            
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                    
                if img_width > 0 and img_height > 0:
                    boxes = []
                    for x1, y1, x2, y2, class_id in unnormalized_boxes:
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        x_center = (x1 + (x2 - x1) / 2) / img_width
                        y_center = (y1 + (y2 - y1) / 2) / img_height
                        boxes.append((x_center, y_center, width, height, class_id))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if boxes:
            with open(label_path, 'w') as f:
                for box in boxes:
                    class_id = box[4] if len(box) > 4 else 0
                    x_center = max(0, min(1, box[0]))
                    y_center = max(0, min(1, box[1]))
                    width = max(0, min(1, box[2]))
                    height = max(0, min(1, box[3]))
                    
                    f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            matched += 1
        else:
            with open(label_path, 'w') as f:
                pass
            unmatched += 1
    
    print(f"{split} labels generated: {matched} with annotations, {unmatched} without annotations")

def find_original_image_name(img_path, img_file, dtld_annotations):
    if img_file.startswith("dtld_"):
        base_name, ext = os.path.splitext(img_file[5:])
        
        if base_name.startswith("DE_"):
            without_de = base_name[3:]
        else:
            without_de = base_name
        
        potential_matches = [
            img_file[5:],
            base_name,
            without_de,
            f"{without_de}.jpg",
            f"{base_name}.tiff",
            f"{base_name}.jpg",
            f"{base_name}.pgm",
            f"{base_name}.png"
        ]
        
        for city in DTLD_CITIES:
            city_lower = city.lower()
            potential_matches.append(f"{city_lower}_{without_de}{ext}")
            potential_matches.append(f"{city_lower}_{without_de}.jpg")
            potential_matches.append(f"{city_lower}_{base_name}{ext}")
            potential_matches.append(f"{city_lower}_{base_name}.jpg")
        
        for name in potential_matches:
            if name in dtld_annotations:
                return name
        
        print(f"Failed to match DTLD image: {img_file}")
        if len(dtld_annotations) > 0:
            similar_keys = []
            for key in dtld_annotations.keys():
                if without_de in key:
                    similar_keys.append(key)
                    if len(similar_keys) >= 3:
                        break
            print(f"Available keys similar to this: {similar_keys}")
        
        return img_file[5:]
        
    elif img_file.startswith("lisa_"):
        original_name = '_'.join(img_file.split('_')[1:])
        return original_name
    
    return img_file

def crop_around_traffic_lights(padding=30):
    global TARGET_DIR
    
    print(f"Creating cropped traffic light dataset with {padding}px padding...")
    
    crops_dir = TARGET_DIR
    crops_images_train = os.path.join(crops_dir, "images", "train")
    crops_images_val = os.path.join(crops_dir, "images", "val") 
    crops_labels_train = os.path.join(crops_dir, "labels", "train")
    crops_labels_val = os.path.join(crops_dir, "labels", "val")
    
    for dir_path in [crops_images_train, crops_images_val, crops_labels_train, crops_labels_val]:
        os.makedirs(dir_path, exist_ok=True)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    for split in ["train", "val"]:
        image_dir = os.path.join(TARGET_DIR, "images", split)
        label_dir = os.path.join(TARGET_DIR, "labels", split)
        crops_image_dir = os.path.join(crops_dir, "images", split)
        crops_label_dir = os.path.join(crops_dir, "labels", split)
        
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_crops = 0
        
        for img_file in tqdm(image_files, desc=f"Cropping {split} images"):
            img_path = os.path.join(image_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)
            
            if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
                continue
            
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                bounding_boxes = []
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            x_min = int((x_center - width/2) * img_width)
                            y_min = int((y_center - height/2) * img_height)
                            x_max = int((x_center + width/2) * img_width)
                            y_max = int((y_center + height/2) * img_height)
                            
                            bounding_boxes.append((class_id, x_min, y_min, x_max, y_max))
                
                if not bounding_boxes:
                    continue
                
                for i, (class_id, x_min, y_min, x_max, y_max) in enumerate(bounding_boxes):
                    crop_x_min = max(0, x_min - padding)
                    crop_y_min = max(0, y_min - padding)
                    crop_x_max = min(img_width, x_max + padding)
                    crop_y_max = min(img_height, y_max + padding)
                    
                    if crop_x_max - crop_x_min < 10 or crop_y_max - crop_y_min < 10:
                        continue
                    
                    crop_img = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                    crop_width, crop_height = crop_img.size
                    
                    base_name = os.path.splitext(img_file)[0]
                    crop_img_file = f"{base_name}_crop{i}.jpg"
                    crop_img_path = os.path.join(crops_image_dir, crop_img_file)
                    
                    crop_img.save(crop_img_path, "JPEG")
                    
                    new_x_min = x_min - crop_x_min
                    new_y_min = y_min - crop_y_min
                    new_x_max = x_max - crop_x_min
                    new_y_max = y_max - crop_y_min
                    
                    new_x_center = (new_x_min + new_x_max) / (2 * crop_width)
                    new_y_center = (new_y_min + new_y_max) / (2 * crop_height)
                    new_width = (new_x_max - new_x_min) / crop_width
                    new_height = (new_y_max - new_y_min) / crop_height
                    
                    new_x_center = max(0, min(1, new_x_center))
                    new_y_center = max(0, min(1, new_y_center))
                    new_width = max(0, min(1, new_width))
                    new_height = max(0, min(1, new_height))
                    
                    crop_label_file = f"{base_name}_crop{i}.txt"
                    crop_label_path = os.path.join(crops_label_dir, crop_label_file)
                    
                    with open(crop_label_path, 'w') as f:
                        f.write(f"{int(class_id)} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")
                    
                    total_crops += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"Created {total_crops} crops for {split} set")
    
    dataset_yaml = {
        'train': "./images/train",
        'val': "./images/val",
        'nc': 3,  # Number of classes
        'names': STATES
    }
    
    with open(os.path.join(crops_dir, "dataset.yaml"), 'w') as f:
        for key, value in dataset_yaml.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Cropped dataset created in {crops_dir}")
    return crops_dir

def detect_traffic_lights(model, image_path):
    results = model.predict(image_path, conf=0.25)
    
    traffic_lights = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = confs[i]
            traffic_lights.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf)
            })
    
    return traffic_lights, results

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
        "mAP50": float(metrics.box.map50),  # mAP at IoU=0.5
        "mAP50-95": float(metrics.box.map),  # mAP at IoU=0.5:0.95
        "precision": float(metrics.box.mp),  # mean precision
        "recall": float(metrics.box.mr),     # mean recall
        "f1": float(metrics.box.map50 * 2 / (metrics.box.mp + metrics.box.mr + 1e-10)),  # F1 score
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    create_metrics_visualizations(results, save_dir, timestamp)
    
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

def create_metrics_visualizations(results, save_dir, timestamp):
    plt.figure(figsize=(12, 8))
    
    main_metrics = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
    metric_values = [results[m] for m in main_metrics]
    
    plt.subplot(2, 1, 1)
    bars = plt.bar(main_metrics, metric_values, color='skyblue')
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('YOLO Detection Model Performance Metrics')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom', rotation=0)
    
    if "per_class_ap50" in results:
        plt.subplot(2, 1, 2)
        classes = list(results["per_class_ap50"].keys())
        ap_values = list(results["per_class_ap50"].values())
        
        sorted_indices = np.argsort(ap_values)
        classes = [classes[i] for i in sorted_indices]
        ap_values = [ap_values[i] for i in sorted_indices]
        
        bars = plt.barh(classes, ap_values, color='lightgreen')
        plt.xlim(0, 1.0)
        plt.xlabel('AP@0.5')
        plt.title('Per-class Average Precision (IoU=0.5)')
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                     f'{width:.4f}',
                     ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"detection_metrics_{timestamp}.png"))
    
    metrics_files = [f for f in os.listdir(save_dir) if f.endswith('.json')]
    if len(metrics_files) > 1:
        all_metrics = []
        for file in metrics_files:
            with open(os.path.join(save_dir, file), 'r') as f:
                data = json.load(f)
                all_metrics.append(data)
        
        all_metrics = sorted(all_metrics, key=lambda x: x.get('timestamp', ''))
        
        plt.figure(figsize=(10, 6))
        timestamps = [m.get('timestamp', 'Unknown') for m in all_metrics]
        map50_values = [m.get('mAP50', 0) for m in all_metrics]
        map_values = [m.get('mAP50-95', 0) for m in all_metrics]
        
        plt.plot(timestamps, map50_values, 'o-', label='mAP@0.5')
        plt.plot(timestamps, map_values, 's-', label='mAP@0.5:0.95')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)
        plt.title('Detection Performance Trend')
        plt.ylabel('mAP Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "detection_metric_trends.png"))


if __name__ == '__main__':

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        device = '0'
    else:
        print("No CUDA devices available, using CPU")
        device = 'cpu'

    os.makedirs(f"{TARGET_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{TARGET_DIR}/images/val", exist_ok=True)
    os.makedirs(f"{TARGET_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{TARGET_DIR}/labels/val", exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "Annotations"), exist_ok=True)


    train_img_dir = os.path.join(TARGET_DIR, "images", "train")
    if not os.path.exists(train_img_dir) or len(os.listdir(train_img_dir)) == 0:
        print("No training images found. Running dataset preparation...")
        prepare_yolo_dataset()
        generate_yolo_labels()
    else:
        label_dir = os.path.join(TARGET_DIR, "labels", "train")
        if not os.path.exists(label_dir) or len(os.listdir(label_dir)) == 0:
            print("Labels not found. Generating labels...")
            generate_yolo_labels()
        else:
            print("Dataset and labels already exist. Skipping preparation.")

            
    if not os.path.exists(os.path.join(TARGET_DIR, "images")):
        print("Creating cropped dataset...")
        crop_around_traffic_lights(padding=30)


    model = YOLO(YOLO_MODEL_SIZE)

    dataset_yaml_path = os.path.abspath(os.path.join(TARGET_DIR, "dataset.yaml"))

    print(f"Using dataset YAML at: {dataset_yaml_path}")
    print(f"Dataset YAML exists: {os.path.exists(dataset_yaml_path)}")

    with open(dataset_yaml_path, "w") as f:
        f.write(f"train: ./images/train\n")
        f.write(f"val: ./images/val\n")
        f.write(f"nc: 3\n")
        f.write(f"names: {STATES}\n")

    print("Updated YAML file with proper paths")
    with open(dataset_yaml_path, "r") as f:
        print(f.read())

    results = model.train(
        data=dataset_yaml_path,
        epochs=EPOCHS,
        workers=WORKERS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='yolo_traffic_light_detector',
        patience=10,
        save=True,
        device=device,
        cache=False,
        amp=True,
        augment = False,
        close_mosaic=10,
        plots=True
    )


    metrics = model.val(data=dataset_yaml_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_model_path = os.path.join("runs", "detect", "yolo_traffic_light_detector", "weights", "best.pt")

    evaluation_results = evaluate_detection_model(
        model_path=best_model_path,
        data_yaml=dataset_yaml_path,
        conf_threshold=0.25,
        iou_threshold=0.5,
        save_dir="detection_metrics"
    )