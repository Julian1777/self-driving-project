import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate, Lambda
from tensorflow.keras.models import Model
import json
import cv2 as cv
from PIL import Image
import csv
import shutil
import sys

BATCH_SIZE = 64
IMG_SIZE = (64,64)
INPUT_SHAPE = (64, 64, 3)
SEED = 123
EPOCHS = 20
PADDING = 5

DTLD_DIR = "dtld_dataset"
LISA_DIR = "lisa_dataset"
STATES = ["green", "red", "yellow", "off"]
ANNOTATION = os.path.join(DTLD_DIR, "Berlin.json")
CROP_DS_DTLD = os.path.join(DTLD_DIR, "cropped_dataset")
CROP_DS_LISA = os.path.join(LISA_DIR, "cropped_dataset")
STATE_DIRS_LISA = {state: os.path.join(CROP_DS_LISA, state) for state in STATES}
VALID_DIRECTIONS = ["front"]
VALID_RELEVANCE = ["relevant"]
MERGED_DS = "merged_dataset"
sequences = ["daySequence1", "daySequence2", "dayTrain", "nightSequence1", "nightSequence2", "nightTrain"]
DTLD_CITIES = ["Berlin", "Bochum", "Dortmund", "Bremen", "Koeln"]

def process_lisa_dataset(force_overwrite=False):
    print(f"Processing LISA dataset from {LISA_DIR}")
    processed_images = 0
    skipped_images = 0
    total_images = 0
    state_dirs = STATE_DIRS_LISA

    if force_overwrite:
        print("Force overwrite enabled - cleaning destination directories")
        for state_dir in state_dirs.values():
            if os.path.exists(state_dir):
                for file in os.listdir(state_dir):
                    file_path = os.path.join(state_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print(f"Cleaned directory: {state_dir}")
    

    for state_dir in state_dirs.values():
        os.makedirs(state_dir, exist_ok=True)

    lisa_to_standard = {
        "go": "green",
        "stop": "red",
        "warning": "yellow",
        "goLeft": "green",
        "stopLeft": "red",
        "warningLeft": "yellow",
        "redLeft": "red",
        "greenLeft": "green"
    }
    
    available_sequences = []
    for seq in sequences:
        seq_path = os.path.join(LISA_DIR, seq)
        if os.path.exists(seq_path) and os.path.isdir(seq_path):
            available_sequences.append(seq)
        else:
            print(f"Sequence folder not found: {seq_path}")
    
    if not available_sequences:
        print("No sequence folders found in LISA dataset")
        return
    
    print(f"Found {len(available_sequences)} sequences in LISA dataset")
    
    for seq in tqdm(available_sequences, desc="Processing sequences"):
        seq_dir = os.path.join(LISA_DIR, seq, seq)

        print(f"  - {seq}: {os.path.exists(seq_path)}")
        if os.path.exists(seq_path):
            try:
                contents = os.listdir(seq_path)
                print(f"Contents: {contents[:5]}{'...' if len(contents) > 5 else ''}")
                
                frames_path = os.path.join(seq_path, "frames")
                if os.path.exists(frames_path):
                    print(f"Found frames directory with {len(os.listdir(frames_path))} files")
                
                nested_path = os.path.join(seq_path, seq)
                if os.path.exists(nested_path):
                    print(f"    Found nested sequence directory: {nested_path}")
                    print(f"    Contents: {os.listdir(nested_path)[:5]}{'...' if len(os.listdir(nested_path)) > 5 else ''}")
            except Exception as e:
                print(f"Error listing directory: {e}")

        
        annotation_dir = os.path.join(LISA_DIR, "Annotations", "Annotations", seq)
        
        if not os.path.exists(annotation_dir):
            print(f"Skipping {seq} - annotation directory not found")
            continue
            
        annotation_file = None
        for ann_file in ["frameAnnotationsBOX.csv", "frameAnnotationsBULB.csv"]:
            ann_path = os.path.join(annotation_dir, ann_file)
            if os.path.exists(ann_path):
                annotation_file = ann_path
                break
                
        if not annotation_file:
            print(f"Skipping {seq} - no annotation file found")
            continue
            
        frames_dir = os.path.join(seq_dir, "frames")
        if not os.path.exists(frames_dir):
            frames_dir = seq_dir

        clip_dirs = []
        if os.path.exists(seq_dir):
            for item in os.listdir(seq_dir):
                clip_path = os.path.join(seq_dir, item)
                if os.path.isdir(clip_path) and (item.endswith("Clip") or "Clip" in item):
                    # Found a clip subfolder
                    clip_dirs.append(clip_path)
                    # Also check for frames directory within clip
                    clip_frames = os.path.join(clip_path, "frames")
                    if os.path.exists(clip_frames):
                        clip_dirs.append(clip_frames)
            
        try:
            with open(annotation_file, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                header = next(reader, None)  # Skip header
                
                for row in tqdm(reader, desc=f"Processing {seq} annotations"):
                    if len(row) >= 6:
                        image_path = row[0]  # Filename
                        image_name = os.path.basename(image_path)
                        light_state = row[1] # Traffic light state

                        standard_state = lisa_to_standard.get(light_state)
                        
                        if standard_state in state_dirs:
                            img_src_path = os.path.join(frames_dir, image_name)
                            if not os.path.exists(img_src_path):
                                alt_paths = [
                                    os.path.join(seq_dir, image_name),
                                    os.path.join(seq_dir, "frames", image_name),
                                    image_path if os.path.isabs(image_path) else None
                                ]
                                for clip_dir in clip_dirs:
                                    alt_paths.append(os.path.join(clip_dir, image_name))
                                
                                # Try all alternative paths
                                for path in alt_paths:
                                    if path and os.path.exists(path):
                                        img_src_path = path
                                        break
                            
                            if os.path.exists(img_src_path):
                                img_dest_path = os.path.join(
                                    state_dirs[standard_state], 
                                    f"{seq}_{image_name}"
                                )

                                if skipped_images < 5 and os.path.exists(img_dest_path):
                                    print(f"DEBUG - Path check:")
                                    print(f"  Destination exists: {img_dest_path}")
                                    print(f"  Source image: {img_src_path}")
                                    print(f"  Is directory empty? {len(os.listdir(state_dirs[standard_state])) == 0}")
                                
                                if not os.path.exists(img_dest_path):
                                    try:
                                        img = cv.imread(img_src_path)
                                        if img is not None:
                                            if len(row) >= 6:
                                                try:
                                                    x1 = int(row[2])
                                                    y1 = int(row[3]) 
                                                    x2 = int(row[4])
                                                    y2 = int(row[5])

                                                    w = x2 - x1
                                                    h = y2 - y1
                                                    
                                                    x0 = max(int(x1 - PADDING), 0)
                                                    y0 = max(int(y1 - PADDING), 0)
                                                    x1_padded = min(int(x2 + PADDING), img.shape[1])
                                                    y1_padded = min(int(y2 + PADDING), img.shape[0])
                                                    
                                                    if x1_padded > x0 and y1_padded > y0:
                                                        crop = img[y0:y1_padded, x0:x1_padded]
                                                        crop_resized = cv.resize(
                                                            crop, 
                                                            IMG_SIZE,
                                                            interpolation=cv.INTER_CUBIC
                                                        )
                                                        cv.imwrite(img_dest_path, crop_resized)
                                                        processed_images += 1
                                                    else:
                                                        full_img_resized = cv.resize(img, IMG_SIZE, interpolation=cv.INTER_CUBIC)
                                                        cv.imwrite(img_dest_path, full_img_resized)
                                                        processed_images += 1
                                                except:
                                                    full_img_resized = cv.resize(img, IMG_SIZE, interpolation=cv.INTER_CUBIC)
                                                    cv.imwrite(img_dest_path, full_img_resized)
                                                    processed_images += 1
                                            else:
                                                full_img_resized = cv.resize(img, IMG_SIZE, interpolation=cv.INTER_CUBIC)
                                                cv.imwrite(img_dest_path, full_img_resized)
                                                processed_images += 1
                                    except Exception as e:
                                        print(f"Error processing {img_src_path}: {e}")
                                else:
                                    print(f"Image already exists: {img_dest_path}")
                                    skipped_images += 1
                                    if skipped_images % 20 == 0:
                                        print(f"Skipped {skipped_images} existing images")
                            else:
                                print(f"Image not found: {img_src_path}")
                        else:
                            if processed_images % 500 == 0:
                                print(f"Skipping unknown state: {standard_state}")
        except Exception as e:
            print(f"Error processing annotation file {annotation_file}: {e}")
    
    print("\n--- LISA Dataset Processing Complete ---")
    for state in STATES:
        state_dir = os.path.join(CROP_DS_LISA, state)
        if os.path.exists(state_dir):
            count = len([f for f in os.listdir(state_dir) if os.path.isfile(os.path.join(state_dir, f))])
            total_images += count
            print(f"  - {state}: {count} images")
    
    print(f"Total: {total_images} traffic light images")
    print(f"Results saved to {CROP_DS_LISA}")

def process_dtld_dataset():
    print(f"Processing DTLD dataset from {DTLD_DIR}")
    
    for state in STATES:
        os.makedirs(os.path.join(CROP_DS_DTLD, state), exist_ok=True)
    
    if not os.path.exists(ANNOTATION):
        print(f"Error: Annotation file not found: {ANNOTATION}")
        return
    
    print(f"Loading annotation file: {ANNOTATION}")
    
    processed_images = 0
    cropped_lights = {state: 0 for state in STATES}
    city_stats = {}

    for city in DTLD_CITIES:
        annotation_file = os.path.join(DTLD_DIR, f"{city}.json")
        
        if not os.path.exists(annotation_file):
            print(f"Error: Annotation file not found: {annotation_file}")
            continue
        
        print(f"\nProcessing city: {city}")
        print(f"Loading annotation file: {annotation_file}")
        
        city_processed = 0
        city_cropped = {state: 0 for state in STATES}

        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            if "images" in data and isinstance(data["images"], list):
                images = data["images"]
                print(f"Found {len(images)} image entries")
                
                if images:
                    print("\nSample image entry structure:")
                    for key, value in images[0].items():
                        print(f"  - {key}: {type(value)}")
                    
                    if "labels" in images[0] and images[0]["labels"]:
                        print("\nSample label structure:")
                        for key, value in images[0]["labels"][0].items():
                            print(f"  - {key}: {type(value)}")
                        
                        if "attributes" in images[0]["labels"][0]:
                            print("\nSample attributes structure:")
                            for key, value in images[0]["labels"][0]["attributes"].items():
                                print(f"  - {key}: {value}")
                
                for image_entry in tqdm(images, desc="Processing images"):
                    rel_path = image_entry.get("image_path", "")
                    if not rel_path:
                        continue
                    
                    img_path = os.path.join(DTLD_DIR, rel_path)
                    
                    if not os.path.exists(img_path):
                        print(f"Warning: Image not found: {img_path}")
                        continue
                    
                    try:
                        pil_image = Image.open(img_path)
                        
                        if processed_images % 50 == 0:
                            print(f"\nProcessing image {processed_images}: {img_path}")
                            print(f"  Format: {pil_image.format}, Mode: {pil_image.mode}, Size: {pil_image.size}")
                        
                        np_img = np.array(pil_image)
                        
                        if np_img.dtype == np.uint16:
                            print(f"Converting 16-bit image: min={np_img.min()}, max={np_img.max()}")
                            np_img = ((np_img - np_img.min()) * 255.0 / (np_img.max() - np_img.min())).astype(np.uint8)
                        
                        if len(np_img.shape) == 2:
                            np_img = np.stack([np_img] * 3, axis=-1)
                        
                        h_img, w_img = np_img.shape[:2]
                        processed_images += 1
                        
                        for label in image_entry.get("labels", []):
                            attr = label.get("attributes", {})
                            
                            if (attr.get("relevance") not in VALID_RELEVANCE or 
                                attr.get("direction") not in VALID_DIRECTIONS):
                                continue
                            
                            state = attr.get("state", "")
                            if state not in STATES:
                                continue
                                
                            x, y, w, h = label.get("x", 0), label.get("y", 0), label.get("w", 0), label.get("h", 0)
                            
                            x0 = max(int(x - PADDING), 0)
                            y0 = max(int(y - PADDING), 0)
                            x1 = min(int(x + w + PADDING), w_img)
                            y1 = min(int(y + h + PADDING), h_img)
                            
                            if x1 <= x0 or y1 <= y0:
                                continue
                            
                            crop = np_img[y0:y1, x0:x1]
                            if crop.size == 0:
                                continue
                            
                            crop_resized = cv.resize(
                                crop,
                                IMG_SIZE,
                                interpolation=cv.INTER_CUBIC
                            )
                            
                            if pil_image.mode == 'RGB':
                                crop_resized = cv.cvtColor(crop_resized, cv.COLOR_RGB2BGR)
                            
                            base = os.path.splitext(os.path.basename(rel_path))[0]
                            out_name = f"{city}_{base}_{int(x)}_{int(y)}.jpg"
                            out_path = os.path.join(CROP_DS_DTLD, state, out_name)
                            
                            cv.imwrite(out_path, crop_resized)
                            cropped_lights[state] += 1
                            city_cropped[state] += 1
                            
                            if sum(cropped_lights.values()) % 100 == 0:
                                print(f"Saved {state} crop: {out_path}")
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

                city_stats[city] = {
                "processed_images": city_processed,
                "cropped_lights": city_cropped.copy()
                }
                        
            else:
                print("No 'images' list found in the annotation file")
                
        except Exception as e:
            print(f"Error processing annotation file: {e}")
        
        # Print final statistics
    print("\n--- Processing Complete ---")
    print(f"Processed {processed_images} images total")
    
    print("\nCropped traffic lights by city:")
    for city, stats in city_stats.items():
        print(f"\n{city}:")
        for state in STATES:
            print(f"  - {state}: {stats['cropped_lights'][state]} images")
        city_total = sum(stats['cropped_lights'].values())
        print(f"  Total for {city}: {city_total} traffic lights")
    
    print("\nOverall traffic lights by state:")
    for state in STATES:
        print(f"  - {state}: {cropped_lights[state]} images")
    print(f"Total: {sum(cropped_lights.values())} cropped traffic lights")
    print(f"Results saved to {CROP_DS_DTLD}")

def create_merged_dataset():
    print("Creating merged dataset...")
    
    for state in STATES:
        os.makedirs(os.path.join(MERGED_DS, state), exist_ok=True)
        
    for state in STATES:
        dtld_state_dir = os.path.join(CROP_DS_DTLD, state)
        merged_state_dir = os.path.join(MERGED_DS, state)
        
        if os.path.exists(dtld_state_dir):
            files = [f for f in os.listdir(dtld_state_dir) if os.path.isfile(os.path.join(dtld_state_dir, f))]
            print(f"Copying {len(files)} images from DTLD {state}")
            
            for file in tqdm(files, desc=f"DTLD {state}"):
                src_path = os.path.join(dtld_state_dir, file)
                dst_path = os.path.join(merged_state_dir, f"dtld_{file}")
                
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
    
    for state in STATES:
        lisa_state_dir = os.path.join(CROP_DS_LISA, state)
        merged_state_dir = os.path.join(MERGED_DS, state)
        
        if os.path.exists(lisa_state_dir):
            files = [f for f in os.listdir(lisa_state_dir) if os.path.isfile(os.path.join(lisa_state_dir, f))]
            print(f"Copying {len(files)} images from LISA {state}")
            
            for file in tqdm(files, desc=f"LISA {state}"):
                src_path = os.path.join(lisa_state_dir, file)
                dst_path = os.path.join(merged_state_dir, f"lisa_{file}")
                
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)

    print("Copying annotation files...")
    annotations_dir = os.path.join(CROP_DS_LISA, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    for seq in sequences:
        seq_ann_dir = os.path.join(LISA_DIR, "Annotations", "Annotations", seq)
        if os.path.exists(seq_ann_dir):
            for ann_file in ["frameAnnotationsBOX.csv", "frameAnnotationsBULB.csv"]:
                ann_path = os.path.join(seq_ann_dir, ann_file)
                if os.path.exists(ann_path):
                    dst_path = os.path.join(annotations_dir, f"{seq}_{ann_file}")
                    shutil.copy2(ann_path, dst_path)
                    print(f"Copied annotation file: {seq}_{ann_file}")
    
    total_images = 0
    print("\n--- Merged Dataset Summary ---")
    for state in STATES:
        state_dir = os.path.join(MERGED_DS, state)
        if os.path.exists(state_dir):
            count = len([f for f in os.listdir(state_dir) if os.path.isfile(os.path.join(state_dir, f))])
            total_images += count
            print(f"  - {state}: {count} images")
    
    print(f"Total: {total_images} images in merged dataset")

def visualize_sample_annotations(dataset_type, num_samples=5):
    """
    Visualize sample images with bounding boxes to debug crop positions.
    
    Args:
        dataset_type: 'dtld' or 'lisa' to specify which dataset to visualize
        num_samples: Number of sample images to visualize
    """
    print(f"Visualizing sample {dataset_type.upper()} images with bounding boxes...")
    
    debug_dir = os.path.join("debug_visualizations", dataset_type)
    os.makedirs(debug_dir, exist_ok=True)
    
    if dataset_type.lower() == 'dtld':
        # Visualize DTLD dataset
        try:
            with open(ANNOTATION, 'r') as f:
                data = json.load(f)
                
            if "images" not in data or not isinstance(data["images"], list):
                print("No valid images found in DTLD annotation file")
                return
                
            samples = data["images"][:num_samples] if len(data["images"]) > num_samples else data["images"]
            
            for i, image_entry in enumerate(samples):
                rel_path = image_entry.get("image_path", "")
                if not rel_path:
                    continue
                    
                img_path = os.path.join(DTLD_DIR, rel_path)
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    continue
                
                try:
                    # Load image
                    pil_image = Image.open(img_path)
                    np_img = np.array(pil_image)
                    
                    # Convert 16-bit to 8-bit if needed
                    if np_img.dtype == np.uint16:
                        np_img = ((np_img - np_img.min()) * 255.0 / (np_img.max() - np_img.min())).astype(np.uint8)
                    
                    # Convert grayscale to BGR
                    if len(np_img.shape) == 2:
                        np_img = cv.cvtColor(np_img, cv.COLOR_GRAY2BGR)
                    elif np_img.shape[2] == 3 and pil_image.mode == 'RGB':
                        np_img = cv.cvtColor(np_img, cv.COLOR_RGB2BGR)
                    
                    # Draw bounding boxes for traffic lights
                    for label in image_entry.get("labels", []):
                        attr = label.get("attributes", {})
                        
                        if (attr.get("relevance") in VALID_RELEVANCE and 
                            attr.get("direction") in VALID_DIRECTIONS):
                            
                            state = attr.get("state", "")
                            if state in STATES:
                                x, y, w, h = label.get("x", 0), label.get("y", 0), label.get("w", 0), label.get("h", 0)
                                
                                # Draw rectangle with color based on state
                                color = (0, 0, 255)  # Red (BGR)
                                if state == "green":
                                    color = (0, 255, 0)  # Green
                                elif state == "yellow":
                                    color = (0, 255, 255)  # Yellow
                                
                                # Draw bounding box and padding area
                                x0 = max(int(x - PADDING), 0)
                                y0 = max(int(y - PADDING), 0)
                                x1 = min(int(x + w + PADDING), np_img.shape[1])
                                y1 = min(int(y + h + PADDING), np_img.shape[0])
                                
                                # Draw outer padded rectangle
                                cv.rectangle(np_img, (x0, y0), (x1, y1), color, 1)
                                
                                # Draw inner exact rectangle
                                cv.rectangle(np_img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 255, 255), 1)
                                
                                # Add text with state
                                cv.putText(np_img, state, (int(x), int(y-5)), 
                                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Save the visualization
                    base = os.path.splitext(os.path.basename(rel_path))[0]
                    out_path = os.path.join(debug_dir, f"{base}_annotated.jpg")
                    cv.imwrite(out_path, np_img)
                    print(f"Saved visualization: {out_path}")
                    
                except Exception as e:
                    print(f"Error visualizing {img_path}: {e}")
                
        except Exception as e:
            print(f"Error processing DTLD annotations: {e}")
            
    elif dataset_type.lower() == 'lisa':
        # Visualize LISA dataset
        for seq in sequences[:min(len(sequences), 2)]:  # Check first 2 sequences
            seq_dir = os.path.join(LISA_DIR, seq)
            if not os.path.exists(seq_dir):
                continue
                
            annotation_dir = os.path.join(LISA_DIR, "Annotations", "Annotations", seq)
            if not os.path.exists(annotation_dir):
                continue
                
            annotation_file = None
            for ann_file in ["frameAnnotationsBOX.csv", "frameAnnotationsBULB.csv"]:
                ann_path = os.path.join(annotation_dir, ann_file)
                if os.path.exists(ann_path):
                    annotation_file = ann_path
                    break
                    
            if not annotation_file:
                continue
                
            # Find potential image sources
            frames_dir = os.path.join(seq_dir, "frames")
            if not os.path.exists(frames_dir):
                frames_dir = seq_dir
            
            # Find recursive directories that might contain images
            clip_dirs = []
            if os.path.exists(seq_dir):
                for item in os.listdir(seq_dir):
                    clip_path = os.path.join(seq_dir, item)
                    if os.path.isdir(clip_path):
                        clip_dirs.append(clip_path)
                        clip_frames = os.path.join(clip_path, "frames")
                        if os.path.exists(clip_frames):
                            clip_dirs.append(clip_frames)
            
            # Process annotation file
            try:
                with open(annotation_file, 'r') as f:
                    reader = csv.reader(f, delimiter=';')
                    next(reader)  # Skip header
                    
                    samples_count = 0
                    for row in reader:
                        if samples_count >= num_samples:
                            break
                            
                        if len(row) >= 6:  # Has bounding box
                            image_path = row[0]  # Filename
                            image_name = os.path.basename(image_path)
                            light_state = row[1]  # Traffic light state
                            
                            # Find image in possible directories
                            img_src_path = os.path.join(frames_dir, image_name)
                            if not os.path.exists(img_src_path):
                                # Try alternatives
                                for dir_path in [seq_dir] + clip_dirs:
                                    alt_path = os.path.join(dir_path, image_name)
                                    if os.path.exists(alt_path):
                                        img_src_path = alt_path
                                        break
                            
                            if os.path.exists(img_src_path):
                                try:
                                    # Load image
                                    img = cv.imread(img_src_path)
                                    if img is None:
                                        continue
                                    
                                    lisa_to_standard = {
                                        "go": "green",
                                        "stop": "red",
                                        "warning": "yellow",
                                        "goLeft": "green",
                                        "stopLeft": "red",
                                        "warningLeft": "yellow",
                                        "redLeft": "red",
                                        "greenLeft": "green"
                                    }
                                    state = lisa_to_standard.get(light_state, light_state)
                                    
                                    color = (0, 0, 255)  # Red (BGR)
                                    if state == "green":
                                        color = (0, 255, 0)  # Green
                                    elif state == "yellow":
                                        color = (0, 255, 255)  # Yellow

                                    x1 = int(row[2])  # Upper left X
                                    y1 = int(row[3])  # Upper left Y
                                    x2 = int(row[4])  # Lower right X
                                    y2 = int(row[5])  # Lower right Y

                                    w = x2 - x1
                                    h = y2 - y1
                                    
                                    x0 = max(int(x1 - PADDING), 0)
                                    y0 = max(int(y1 - PADDING), 0)
                                    x1_padded = min(int(x2 + PADDING), img.shape[1])
                                    y1_padded = min(int(y2 + PADDING), img.shape[0])
                                    
                                    cv.rectangle(img, (x0, y0), (x1_padded, y1_padded), color, 1)
                                    
                                    cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                                    
                                    cv.putText(img, state, (x1, y1-5), 
                                              cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                    
                                    out_path = os.path.join(debug_dir, f"{seq}_{image_name}_annotated.jpg")
                                    cv.imwrite(out_path, img)
                                    print(f"Saved visualization: {out_path}")
                                    
                                    samples_count += 1
                                except Exception as e:
                                    print(f"Error visualizing {img_src_path}: {e}")
            except Exception as e:
                print(f"Error processing LISA annotations: {e}")
    
    print(f"Visualization complete. Check the {debug_dir} directory for results.")

def visualize_predictions(model, dataset, num_batches=1, show_ground_truth=True):
    class_names = ['green', 'red', 'yellow', 'off']
    
    for batch_images, batch_labels in dataset.take(num_batches):
        predictions = model.predict(batch_images, verbose=0)
        pred_classes = tf.argmax(predictions, axis=1)
        
        num_images = min(len(batch_images), 12)
        columns = 3
        rows = (num_images + columns - 1) // columns
        
        plt.figure(figsize=(12, 4 * rows))
        
        for i in range(num_images):
            plt.subplot(rows, columns, i+1)
            plt.imshow(batch_images[i].numpy().astype("uint8"))
            
            pred_idx = pred_classes[i]
            confidence = predictions[i][pred_idx]
            pred_class = class_names[pred_idx]
            
            if show_ground_truth:
                true_class = class_names[batch_labels[i]]
                title = f"Prediction: {pred_class}\n"
                title += f"Confidence: {confidence:.2f}\n"
                title += f"Actual: {true_class}"
                color = "green" if pred_class == true_class else "red"
            else:
                title = f"Prediction: {pred_class}\n"
                title += f"Confidence: {confidence:.2f}"
                color = "black"
                
            plt.title(title, color=color, fontsize=10)
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()


def hsv_feature_extraction(image):
    # Convert to HSV colorspace
    hsv = tf.image.rgb_to_hsv(image)
    
    # Extract all channels
    h_channel = hsv[:, :, :, 0]  # Hue
    s_channel = hsv[:, :, :, 1]  # Saturation
    v_channel = hsv[:, :, :, 2]  # Value (brightness)
    
    # Get image dimensions
    batch_size = tf.shape(image)[0]
    height = tf.shape(image)[1]
    width = tf.shape(image)[2]
    
    # Divide image into three vertical regions (top, middle, bottom)
    h_third = height // 3
    
    # Extract regions for all channels
    # Brightness (Value)
    top_v = v_channel[:, :h_third, :] 
    middle_v = v_channel[:, h_third:2*h_third, :]
    bottom_v = v_channel[:, 2*h_third:, :]
    
    # Saturation
    top_s = s_channel[:, :h_third, :] 
    middle_s = s_channel[:, h_third:2*h_third, :]
    bottom_s = s_channel[:, 2*h_third:, :]
    
    # Hue
    top_h = h_channel[:, :h_third, :] 
    middle_h = h_channel[:, h_third:2*h_third, :]
    bottom_h = h_channel[:, 2*h_third:, :]
    
    # --- BRIGHTNESS FEATURES ---
    # Keep all existing brightness features
    top_brightness = tf.reduce_mean(top_v, axis=[1, 2])
    middle_brightness = tf.reduce_mean(middle_v, axis=[1, 2])
    bottom_brightness = tf.reduce_mean(bottom_v, axis=[1, 2])
    
    overall_brightness = tf.reduce_mean(v_channel, axis=[1, 2])
    
    epsilon = 1e-7
    rel_top_brightness = top_brightness / (overall_brightness + epsilon)
    rel_middle_brightness = middle_brightness / (overall_brightness + epsilon)
    rel_bottom_brightness = bottom_brightness / (overall_brightness + epsilon)
    
    max_top_brightness = tf.reduce_max(top_v, axis=[1, 2])
    max_middle_brightness = tf.reduce_max(middle_v, axis=[1, 2])
    max_bottom_brightness = tf.reduce_max(bottom_v, axis=[1, 2])
    
    var_top_brightness = tf.math.reduce_variance(tf.reshape(top_v, [batch_size, -1]), axis=1)
    var_middle_brightness = tf.math.reduce_variance(tf.reshape(middle_v, [batch_size, -1]), axis=1)
    var_bottom_brightness = tf.math.reduce_variance(tf.reshape(bottom_v, [batch_size, -1]), axis=1)
    
    # --- SATURATION FEATURES ---
    # Average saturation per region
    top_saturation = tf.reduce_mean(top_s, axis=[1, 2])
    middle_saturation = tf.reduce_mean(middle_s, axis=[1, 2])
    bottom_saturation = tf.reduce_mean(bottom_s, axis=[1, 2])
    
    # Maximum saturation (helps detect vivid colors)
    max_top_saturation = tf.reduce_max(top_s, axis=[1, 2])
    max_middle_saturation = tf.reduce_max(middle_s, axis=[1, 2])
    max_bottom_saturation = tf.reduce_max(bottom_s, axis=[1, 2])
    
    # --- HUE FEATURES ---
    # Calculate histograms of hue values in each region
    # Traffic light colors have specific hue ranges:
    # Red: ~0.0 or ~1.0
    # Yellow: ~0.1-0.2
    # Green: ~0.3-0.4
    
    red_mask1 = tf.logical_and(
        tf.greater_equal(h_channel, 0.0),
        tf.less_equal(h_channel, 10.0/180.0)
    )
    red_mask2 = tf.logical_and(
        tf.greater_equal(h_channel, 170.0/180.0),
        tf.less_equal(h_channel, 1.0)
    )
    
    # Also require minimum saturation and value
    red_mask1 = tf.logical_and(
        red_mask1, 
        tf.logical_and(
            tf.greater_equal(s_channel, 100.0/255.0),
            tf.greater_equal(v_channel, 100.0/255.0)
        )
    )
    red_mask2 = tf.logical_and(
        red_mask2, 
        tf.logical_and(
            tf.greater_equal(s_channel, 100.0/255.0),
            tf.greater_equal(v_channel, 100.0/255.0)
        )
    )
    
    red_mask = tf.logical_or(red_mask1, red_mask2)
    
    # Yellow mask (20-30 in OpenCV scale)
    yellow_mask = tf.logical_and(
        tf.logical_and(
            tf.greater_equal(h_channel, 20.0/180.0),
            tf.less_equal(h_channel, 30.0/180.0)
        ),
        tf.logical_and(
            tf.greater_equal(s_channel, 100.0/255.0),
            tf.greater_equal(v_channel, 100.0/255.0)
        )
    )
    
    # Green mask (40-80 in OpenCV scale)
    green_mask = tf.logical_and(
        tf.logical_and(
            tf.greater_equal(h_channel, 40.0/180.0),
            tf.less_equal(h_channel, 80.0/180.0)
        ),
        tf.logical_and(
            tf.greater_equal(s_channel, 100.0/255.0),
            tf.greater_equal(v_channel, 100.0/255.0)
        )
    )
    
    # Convert boolean masks to float
    red_mask = tf.cast(red_mask, tf.float32)
    yellow_mask = tf.cast(yellow_mask, tf.float32)
    green_mask = tf.cast(green_mask, tf.float32)
    
    # Count pixels with specific hues in each region
    top_red_ratio = tf.reduce_mean(red_mask[:, :h_third, :], axis=[1, 2])
    middle_red_ratio = tf.reduce_mean(red_mask[:, h_third:2*h_third, :], axis=[1, 2])
    bottom_red_ratio = tf.reduce_mean(red_mask[:, 2*h_third:, :], axis=[1, 2])
    
    top_yellow_ratio = tf.reduce_mean(yellow_mask[:, :h_third, :], axis=[1, 2])
    middle_yellow_ratio = tf.reduce_mean(yellow_mask[:, h_third:2*h_third, :], axis=[1, 2])
    bottom_yellow_ratio = tf.reduce_mean(yellow_mask[:, 2*h_third:, :], axis=[1, 2])
    
    top_green_ratio = tf.reduce_mean(green_mask[:, :h_third, :], axis=[1, 2])
    middle_green_ratio = tf.reduce_mean(green_mask[:, h_third:2*h_third, :], axis=[1, 2])
    bottom_green_ratio = tf.reduce_mean(green_mask[:, 2*h_third:, :], axis=[1, 2])
    
    # Stack all features
    hsv_features = tf.stack([
        # Original brightness features
        top_brightness, middle_brightness, bottom_brightness,
        rel_top_brightness, rel_middle_brightness, rel_bottom_brightness,
        max_top_brightness, max_middle_brightness, max_bottom_brightness,
        var_top_brightness, var_middle_brightness, var_bottom_brightness,
        overall_brightness,
        
        # Saturation features
        top_saturation, middle_saturation, bottom_saturation,
        max_top_saturation, max_middle_saturation, max_bottom_saturation,
        
        # Hue distribution features
        top_red_ratio, middle_red_ratio, bottom_red_ratio,
        top_yellow_ratio, middle_yellow_ratio, bottom_yellow_ratio,
        top_green_ratio, middle_green_ratio, bottom_green_ratio
    ], axis=1)
    
    return hsv_features

#Only for testing images/predictions
def process_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    #Color Jitter
    tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_contrast(x, lower=0.8, upper=1.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, lower=0.8, upper=1.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, max_delta=0.05))
])

if not os.path.exists("debug_visualizations"):
    print("Generating debug visualizations...")
    visualize_sample_annotations('dtld', num_samples=10)
    visualize_sample_annotations('lisa', num_samples=10)
    print("Debug visualizations complete.")
else:
    print("Debug visualizations already exist.")

if not os.path.exists(os.path.join(CROP_DS_LISA, "green")):
    print("LISA Dataset hasn't been processed yet. Processing now...")
    process_lisa_dataset(force_overwrite=True)
else:
    print("LISA Dataset is already processed. Skipping dataset processing.")

if not os.path.exists(os.path.join(CROP_DS_DTLD, "green")):
    print("DTLD Dataset hasn't been processed yet. Processing now...")
    process_dtld_dataset()
else:
    print("DTLD Dataset is already Processed. Skipping dataset processing.")

if (os.path.exists(os.path.join(CROP_DS_DTLD, "green")) and 
    not os.path.exists(os.path.join(MERGED_DS, "green"))):
    print("Creating merged dataset from processed data...")
    create_merged_dataset()

if not os.path.exists(MERGED_DS) or not os.path.exists(os.path.join(MERGED_DS, "green")):
    print("Error: Merged dataset not found. Please ensure DTLD dataset is processed.")
    sys.exit(1)

full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    MERGED_DS,
    batch_size=None,
    image_size=IMG_SIZE,
    shuffle=True,
    subset="training",
    validation_split=0.2,
    seed=SEED
)

print(full_dataset.class_names)

dataset_size = sum(1 for _ in full_dataset)
print(f"Total images: {dataset_size}")


train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

full_dataset = full_dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=False)

train_ds = full_dataset.take(train_size)
val_ds = full_dataset.skip(train_size).take(val_size)
test_ds = full_dataset.skip(train_size + val_size)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


print(f"Train dataset size: {len(train_ds)} batches")
print(f"Validation dataset size: {len(val_ds)} batches")
print(f"Test dataset size: {len(test_ds)} batches")

all_train_labels = []
for _, labels in train_ds.unbatch():
    all_train_labels.append(int(labels.numpy()))

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(STATES)),
    y=all_train_labels
)
class_weights = {i: w for i, w in enumerate(weights)}
print("Class weights:", class_weights)



def build_traffic_light_model():
    inputs = Input(shape=INPUT_SHAPE)
    augmented = data_augmentation(inputs)

    base_model = EfficientNetB0(include_top=False, input_tensor=augmented, weights='imagenet')
    base_model.trainable = False  # freeze during initial training

    x = GlobalAveragePooling2D()(base_model.output)
    cnn_features = Dense(128, activation='relu')(x)
    cnn_features = Dropout(0.5)(cnn_features)

    hsv_features = Lambda(hsv_feature_extraction)(augmented)
    hsv_branch = Dense(32, activation='relu')(hsv_features)

    combined = Concatenate()([cnn_features, hsv_branch])
    outputs = Dense(len(STATES), activation='softmax', dtype='float32')(combined)

    model = Model(inputs=inputs, outputs=outputs)
    return model


model = build_traffic_light_model()

print(model.summary())

if os.path.exists("traffic_light_classification.h5"):
    model.load_weights("traffic_light_classification_weights.h5")
    print("Model loaded successfully.")
    visualize_predictions(model, val_ds)


elif os.path.exists("traffic_light_classification_checkpoint.h5"):
    model.load_weights("traffic_light_classification_checkpoint.h5")
    print("Model Checkpoint loaded successfully.")
    visualize_predictions(model, val_ds)
else:

    early_stopping = EarlyStopping(
    'val_accuracy', 
    patience=4, 
    restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "traffic_light_classification_checkpoint.h5",
        save_weights_only=True,
        monitor="accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    model.save_weights("traffic_light_classification_weights.h5")
    print("Model weights saved successfully.")
    visualize_predictions(model, val_ds)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()