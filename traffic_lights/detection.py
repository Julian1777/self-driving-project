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


# Config
SOURCE_DIR = "dataset"
TARGET_DIR = "yolo_dataset"
TRAIN_RATIO = 0.9
CLASS_ID = 0
EPOCHS = 20
GRID_SIZE = 7
BOXES_PER_CELL = 2
DTLD_DIR = "dtld_dataset"
LISA_DIR = "lisa_dataset"
STATES = ["green", "red", "yellow", "off"]
ANNOTATION = os.path.join(DTLD_DIR, "Berlin.json")
sequences = ["daySequence1", "daySequence2", "dayTrain", "nightSequence1", "nightSequence2", "nightTrain"]
DTLD_CITIES = ["Berlin", "Bochum", "Dortmund", "Bremen", "Koeln"]



os.makedirs(f"{TARGET_DIR}/images/train", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/images/val", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/labels/val", exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "Annotations"), exist_ok=True)

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
    
    # Process all city annotation files
    dtld_annotations_dir = os.path.join(TARGET_DIR, "Annotations", "DTLD")
    if os.path.exists(dtld_annotations_dir):
        print("Processing DTLD annotations...")
        
        for city in DTLD_CITIES:
            city_annotation = os.path.join(dtld_annotations_dir, f"{city}.json")
            if os.path.exists(city_annotation):
                print(f"Processing {city} annotations...")
                try:
                    with open(city_annotation, 'r') as f:
                        data = json.load(f)
                        
                    if "images" in data and isinstance(data["images"], list):
                        for image_entry in tqdm(data["images"], desc=f"Loading {city} annotations"):
                            rel_path = image_entry.get("image_path", "")
                            if not rel_path:
                                continue
                                
                            img_filename = os.path.basename(rel_path)
                            traffic_lights = []
                            
                            img_width = image_entry.get("width", 0)
                            img_height = image_entry.get("height", 0)
                            
                            if img_width <= 0 or img_height <= 0:
                                continue
                            
                            for label in image_entry.get("labels", []):
                                attr = label.get("attributes", {})
                                
                                x = label.get("x", 0)
                                y = label.get("y", 0)
                                w = label.get("w", 0)
                                h = label.get("h", 0)
                                
                                if w > 0 and h > 0:
                                    x_center = (x + w/2) / img_width
                                    y_center = (y + h/2) / img_height
                                    width = w / img_width
                                    height = h / img_height
                                    
                                    traffic_lights.append((x_center, y_center, width, height))
                            
                            if traffic_lights:
                                if img_filename in dtld_annotations:
                                    city_img_filename = f"{city.lower()}_{img_filename}"
                                    dtld_annotations[city_img_filename] = traffic_lights
                                else:
                                    dtld_annotations[img_filename] = traffic_lights
                                
                except Exception as e:
                    print(f"Error processing {city} annotations: {e}")
                    
        print(f"Loaded annotations for {len(dtld_annotations)} DTLD images")
    
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
                                
                                try:
                                    x1 = float(row[2])
                                    y1 = float(row[3])
                                    x2 = float(row[4])
                                    y2 = float(row[5])
                                    
                                    if image_name not in lisa_annotations:
                                        lisa_annotations[image_name] = []
                                        
                                    lisa_annotations[image_name].append((x1, y1, x2, y2))
                                except:
                                    continue
                except Exception as e:
                    print(f"Error processing {ann_path}: {e}")
    
        print(f"Loaded annotations for {len(lisa_annotations)} LISA images")
    
    generate_labels_for_dir("train", dtld_annotations, lisa_annotations)
    
    generate_labels_for_dir("val", dtld_annotations, lisa_annotations)
    
    print("YOLO label generation complete!")

def generate_labels_for_dir(split, dtld_annotations, lisa_annotations):
    """Generate labels for a specific directory (train or val)."""
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
    
    for img_file in tqdm(image_files, desc=f"Processing {split} images"):
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        img_path = os.path.join(image_dir, img_file)
        
        original_name = find_original_image_name(img_path, img_file)
        
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
                    for x1, y1, x2, y2 in unnormalized_boxes:
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        x_center = (x1 + (x2 - x1) / 2) / img_width
                        y_center = (y1 + (y2 - y1) / 2) / img_height
                        boxes.append((x_center, y_center, width, height))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if boxes:
            with open(label_path, 'w') as f:
                for box in boxes:
                    x_center = max(0, min(1, box[0]))
                    y_center = max(0, min(1, box[1]))
                    width = max(0, min(1, box[2]))
                    height = max(0, min(1, box[3]))
                    
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            matched += 1
        else:
            with open(label_path, 'w') as f:
                pass
            unmatched += 1
    
    print(f"{split} labels generated: {matched} with annotations, {unmatched} without annotations")

def find_original_image_name(img_path, img_file):
    """Find the original image name for annotation matching."""
    if img_file.startswith("dtld_") or img_file.startswith("lisa_"):
        parts = img_file.split('_')
        if img_file.startswith("dtld_"):
            for city in DTLD_CITIES:
                city_lower = city.lower()
                if city_lower in img_file:
                    city_index = img_file.find(city_lower)
                    if city_index > 0:
                        original_name = img_file[city_index + len(city_lower) + 1:]
                        return f"{city_lower}_{original_name}"
            
            # For cases without city prefix in filename
            prefix_count = img_file.count('_', 0, img_file.find("dtld_") + 5)
            original_name = '_'.join(parts[prefix_count:])
            return original_name
        else:
            original_name = '_'.join(parts[2:])
            return original_name
    
    return img_file

def crop_around_traffic_lights(padding=30):
    print(f"Creating cropped traffic light dataset with {padding}px padding...")
    
    crops_dir = os.path.join(TARGET_DIR, "crops")
    crops_images_train = os.path.join(crops_dir, "images", "train")
    crops_images_val = os.path.join(crops_dir, "images", "val")
    crops_labels_train = os.path.join(crops_dir, "labels", "train")
    crops_labels_val = os.path.join(crops_dir, "labels", "val")
    
    os.makedirs(crops_images_train, exist_ok=True)
    os.makedirs(crops_images_val, exist_ok=True)
    os.makedirs(crops_labels_train, exist_ok=True)
    os.makedirs(crops_labels_val, exist_ok=True)
    
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
        'train': os.path.join(crops_dir, "images", "train"),
        'val': os.path.join(crops_dir, "images", "val"),
        'nc': 1,  # Number of classes
        'names': ['traffic_light']
    }
    
    with open(os.path.join(crops_dir, "dataset.yaml"), 'w') as f:
        for key, value in dataset_yaml.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Cropped dataset created in {crops_dir}")
    
    global TARGET_DIR
    TARGET_DIR = crops_dir
    
    return crops_dir

def yolo_model(input_shape=(224,224,3), grid_size=7, boxes_per_cell=2):
    outputs_per_box = 5
    output_size = grid_size * grid_size * boxes_per_cell * outputs_per_box

    inputs = tf.keras.Input(shape=input_shape)

    x = data_augmentation(inputs)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(1024, 3, padding='same', activation='relu')(x)

    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(output_size, activation='sigmoid')(x)


    outputs = layers.Reshape((grid_size, grid_size, boxes_per_cell, outputs_per_box))(x)

    return models.Model(inputs, outputs)

def load_dataset(image_dir, label_dir, image_size=(224, 224), grid_size=7, boxes_per_cell=2):
    def load_example(img_path):
        img_path_str = img_path.numpy().decode()
        base_name = os.path.splitext(os.path.basename(img_path_str))[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        img = tf.io.read_file(img_path_str)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = img / 255.0
        
        outputs_per_box = 5  # x, y, w, h, confidence
        target = np.zeros((grid_size, grid_size, boxes_per_cell, outputs_per_box))
        
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        grid_x = int(x_center * grid_size)
                        grid_y = int(y_center * grid_size)
                        
                        grid_x = max(0, min(grid_size - 1, grid_x))
                        grid_y = max(0, min(grid_size - 1, grid_y))
                        
                        cell_x = x_center * grid_size - grid_x
                        cell_y = y_center * grid_size - grid_y
                        
                        min_conf_idx = np.argmin(target[grid_y, grid_x, :, 4])
                        
                        target[grid_y, grid_x, min_conf_idx, 0] = cell_x
                        target[grid_y, grid_x, min_conf_idx, 1] = cell_y
                        target[grid_y, grid_x, min_conf_idx, 2] = width
                        target[grid_y, grid_x, min_conf_idx, 3] = height
                        target[grid_y, grid_x, min_conf_idx, 4] = 1.0  # confidence
        
        return img, tf.convert_to_tensor(target, dtype=tf.float32)
    def wrapper(img_path):
        img, label = tf.py_function(load_example, [img_path], [tf.float32, tf.float32])
        img.set_shape((image_size[0], image_size[1], 3))
        label.set_shape((5,))
        return img, label

    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    print(f"Found {len(img_paths)} images in {image_dir}")
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.map(wrapper)
    dataset = dataset.shuffle(500).batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        
        remaining = indices[1:]
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in remaining])
        
        indices = remaining[ious <= iou_threshold]
    
    return keep

def evaluate_detection_model(model, dataset, iou_thresholds=[0.5]):
    all_predictions = []
    all_ground_truths = []
    
    for batch_images, batch_targets in dataset:
        predictions = model.predict(batch_images, verbose=0)
        
        for i in range(len(batch_images)):
            pred = predictions[i]
            target = batch_targets[i]
            
            pred_boxes = []
            pred_scores = []
            
            grid_size = pred.shape[0]
            boxes_per_cell = pred.shape[2]
            
            for grid_y in range(grid_size):
                for grid_x in range(grid_size):
                    for b in range(boxes_per_cell):
                        cell_x = pred[grid_y, grid_x, b, 0]
                        cell_y = pred[grid_y, grid_x, b, 1]
                        cell_w = pred[grid_y, grid_x, b, 2]
                        cell_h = pred[grid_y, grid_x, b, 3]
                        confidence = pred[grid_y, grid_x, b, 4]
                        
                        if confidence < 0.1:
                            continue
                        
                        x_center = (grid_x + cell_x) / grid_size
                        y_center = (grid_y + cell_y) / grid_size
                        
                        x_min = x_center - cell_w/2
                        y_min = y_center - cell_h/2
                        x_max = x_center + cell_w/2
                        y_max = y_center + cell_h/2
                        
                        pred_boxes.append([x_min, y_min, x_max, y_max])
                        pred_scores.append(confidence)
            
            # Apply NMS
            if pred_boxes:
                keep_indices = non_max_suppression(pred_boxes, pred_scores)
                pred_boxes = [pred_boxes[i] for i in keep_indices]
                pred_scores = [pred_scores[i] for i in keep_indices]
            
            gt_boxes = []
            
            for grid_y in range(grid_size):
                for grid_x in range(grid_size):
                    for b in range(boxes_per_cell):
                        confidence = target[grid_y, grid_x, b, 4]
                        
                        if confidence > 0.5:
                            cell_x = target[grid_y, grid_x, b, 0]
                            cell_y = target[grid_y, grid_x, b, 1]
                            cell_w = target[grid_y, grid_x, b, 2]
                            cell_h = target[grid_y, grid_x, b, 3]
                            
                            x_center = (grid_x + cell_x) / grid_size
                            y_center = (grid_y + cell_y) / grid_size
                            
                            x_min = x_center - cell_w/2
                            y_min = y_center - cell_h/2
                            x_max = x_center + cell_w/2
                            y_max = y_center + cell_h/2
                            
                            gt_boxes.append([x_min, y_min, x_max, y_max])
            
            all_predictions.append((pred_boxes, pred_scores))
            all_ground_truths.append(gt_boxes)
    
    map_values = {}
    
    for iou_threshold in iou_thresholds:
        precisions = []
        recalls = []
        
        for (pred_boxes, pred_scores), gt_boxes in zip(all_predictions, all_ground_truths):
            if len(gt_boxes) == 0:
                if len(pred_boxes) == 0:
                    precisions.append(1.0)
                    recalls.append(1.0)
                else:
                    precisions.append(0.0)
                    recalls.append(1.0)
                continue
            
            if len(pred_boxes) == 0:
                precisions.append(1.0)
                recalls.append(0.0)
                continue
            
            matches = np.zeros(len(gt_boxes), dtype=bool)
            
            tp = 0
            fp = 0
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    if matches[i]:
                        continue
                    
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                
                if best_idx >= 0 and best_iou >= iou_threshold:
                    tp += 1
                    matches[best_idx] = True
                else:
                    fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / len(gt_boxes)
            
            precisions.append(precision)
            recalls.append(recall)
        
        if precisions:
            map_values[iou_threshold] = sum(precisions) / len(precisions)
        else:
            map_values[iou_threshold] = 0.0
    
    map_values['mean'] = sum(map_values.values()) / len(map_values)
    
    return map_values

def yolo_loss(y_true, y_pred):
    pred_xy = y_pred[..., 0:2]  # center x, y
    pred_wh = y_pred[..., 2:4]  # width, height
    pred_conf = y_pred[..., 4:5]  # confidence
    
    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]
    true_conf = y_true[..., 4:5]
    
    object_mask = tf.cast(true_conf > 0.5, tf.float32)
    
    lambda_coord = 5.0
    lambda_noobj = 0.5
    
    xy_loss = lambda_coord * tf.reduce_sum(
        object_mask * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1, keepdims=True)
    )
    
    true_wh_sqrt = tf.sqrt(tf.maximum(true_wh, 1e-7))
    pred_wh_sqrt = tf.sqrt(tf.maximum(pred_wh, 1e-7))
    
    wh_loss = lambda_coord * tf.reduce_sum(
        object_mask * tf.reduce_sum(tf.square(true_wh_sqrt - pred_wh_sqrt), axis=-1, keepdims=True)
    )
    
    conf_obj_loss = tf.reduce_sum(
        object_mask * tf.square(true_conf - pred_conf)
    )
    
    conf_noobj_loss = lambda_noobj * tf.reduce_sum(
        (1 - object_mask) * tf.square(true_conf - pred_conf)
    )
    
    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    total_loss = (xy_loss + wh_loss + conf_obj_loss + conf_noobj_loss) / batch_size
    
    return total_loss

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area <= 0:
        return 0
    
    return intersection_area / union_area


if not os.path.exists(os.path.join(TARGET_DIR, "dataset.yaml")):
    prepare_yolo_dataset()
    generate_yolo_labels()
else:
    if not os.listdir(os.path.join(TARGET_DIR, "labels", "train")):
        print("Labels not found. Generating labels...")
        generate_yolo_labels()
    else:
        print("Dataset and labels already exist. Skipping preparation.")

        
if not os.path.exists(os.path.join(TARGET_DIR, "crops")):
    print("Creating cropped dataset...")
    crop_around_traffic_lights(padding=30)
    TARGET_DIR = os.path.join(TARGET_DIR, "crops")

train_dataset = load_dataset(
    os.path.join(TARGET_DIR, "images", "train"),
    os.path.join(TARGET_DIR, "labels", "train"),
    grid_size=GRID_SIZE, 
    boxes_per_cell=BOXES_PER_CELL
)

val_dataset = load_dataset(
    os.path.join(TARGET_DIR, "images", "val"),
    os.path.join(TARGET_DIR, "labels", "val"),
    grid_size=GRID_SIZE, 
    boxes_per_cell=BOXES_PER_CELL
)

model = yolo_model(grid_size=GRID_SIZE, boxes_per_cell=BOXES_PER_CELL)

if os.path.exists("traffic_light_detection.h5"):
    print("Model already exists. Loading model...")
    model = tf.keras.models.load_model("traffic_light_classification.h5")
    print("Model loaded successfully.")
elif os.path.exists("traffic_light_detection_checkpoint.h5"):
    print("Using Checkpoint model...")
    model = tf.keras.models.load_model("best_light_detection_model.h5") 
    print("Model loaded successfully.")
else:

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "traffic_light_detection_checkpoint.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    model.compile(
        optimizer='adam',
        loss=yolo_loss,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        batch_size=None,
        verbose=1,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    model.save('traffic_light_detection.h5')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title('Training Loss')


