import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
import csv
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import json
import cv2 as cv
from PIL import Image
from skimage import exposure
from skimage import io
import tifffile  # For advanced TIFF handling


BATCH_SIZE = 64
IMG_SIZE = (64,64)
SEED = 123
EPOCHS = 20
PADDING = 10

ORIGINAL_DS_DIR = "dtld_dataset"
DS_DIR = "dataset"
CLASS_DS_DIR = "classified_dataset"
STATES = ["green", "red", "yellow", "off"]
STATE_DIRS = {state: os.path.join(CLASS_DS_DIR, state) for state in STATES}
ANNOATION = os.path.join(ORIGINAL_DS_DIR, "Berlin.json")
CROP_DS = os.path.join(ORIGINAL_DS_DIR, "cropped_dataset")



def crop_dataset():
    os.makedirs(CROP_DS, exist_ok=True)
    print("Loading annotations from original dataset...")
    
    # Create preview directories
    debug_dir = os.path.join(CROP_DS, "debug")
    preview_dir = os.path.join(CROP_DS, "previews")
    temp_dir = os.path.join(CROP_DS, "temp_converted")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Track statistics
    stats = {
        'processed': 0,
        'skipped_tiny': 0,
        'skipped_dark': 0,
        'skipped_unknown_state': 0,
        'saved': {'green': 0, 'red': 0, 'yellow': 0, 'off': 0}
    }
    
    with open(ANNOATION, "r") as f:
        data = json.load(f)

    for entry in tqdm(data["images"], desc="Processing images"):
        rel_path = entry["image_path"]
        full_path = os.path.normpath(os.path.join(ORIGINAL_DS_DIR, rel_path.lstrip('./')))
        
        # Skip if no traffic lights
        if not entry["labels"]:
            continue
            
        # Step 1: Convert TIFF to JPEG first (better color handling)
        img_name = os.path.basename(rel_path)
        jpg_path = os.path.join(temp_dir, os.path.splitext(img_name)[0] + ".png")
        
        try:
            # Use tifffile with imagecodecs plugin
            np_img = robust_tiff_reader(full_path)
            
            # Handle 16-bit conversion
            if np_img is None or np_img.size == 0:
                print(f"Empty image: {full_path}")
                stats['skipped_dark'] += 1
                continue
            
            # Apply gamma correction
            avg_luminance = np.mean(cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY))
            gamma = 0.5 if avg_luminance > 128 else 0.7
            np_img = exposure.adjust_gamma(np_img, gamma=gamma)
            
            h_img, w_img = np_img.shape[:2]
            
            # Rest of your processing code remains the same...
            debug_img = np_img.copy()
            
            for label in entry["labels"]:
                stats['processed'] += 1

        except Exception as e:
            print(f"TIFF processing failed for {full_path}: {str(e)}")
            stats['skipped_dark'] += 1
            continue
        
        h_img, w_img = np_img.shape[:2]
        
        # Create debug visualization
        debug_img = np_img.copy()
        
        for label in entry["labels"]:
            stats['processed'] += 1
            attr = label["attributes"]
            
            if attr["relevance"] != "relevant" or attr["direction"] != "front":
                continue

            x, y, w, h = label["x"], label["y"], label["w"], label["h"]
            state = attr["state"]
            
            # Skip states that aren't in our defined states list
            if state not in STATES:
                print(f"Skipping unknown state: {state}")
                stats['skipped_unknown_state'] += 1
                continue
            
            # Skip very tiny traffic lights
            if w < 4 or h < 8:
                print(f"Skipping tiny traffic light: {w}x{h}")
                stats['skipped_tiny'] += 1
                continue
            
            # Use larger padding for small traffic lights
            padding_factor = 1.5  # Default padding factor
            
            # Add more padding for small traffic lights
            if w < 15:
                padding_factor = 3.0
            
            x_padding = int(w * padding_factor)
            y_padding = int(h * padding_factor)
            
            x0 = max(x - x_padding, 0)
            y0 = max(y - y_padding, 0)
            x1 = min(x + w + x_padding, w_img)
            y1 = min(y + h + y_padding, h_img)
            
            # Draw on debug image
            cv.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Original box in red
            cv.rectangle(debug_img, (x0, y0), (x1, y1), (0, 255, 0), 1)  # Padded box in green
            cv.putText(debug_img, state, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Extract crop with original colors (no normalization)
            crop = np_img[y0:y1, x0:x1].copy()
            if crop.size == 0:
                print(f"Empty crop at {rel_path} box {x,y,w,h}")
                continue
            
            # Only basic contrast enhancement - no aggressive processing
            # Simple auto-contrast to maximize dynamic range
            crop_enhanced = crop.copy()
            hsv = cv.cvtColor(crop_enhanced, cv.COLOR_RGB2HSV)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            crop_enhanced = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
            
            # Create a directory for this state
            state_dir = os.path.join(CROP_DS, state)
            os.makedirs(state_dir, exist_ok=True)
            
            # Save original crop with original colors (PNG for best quality)
            base = os.path.splitext(os.path.basename(rel_path))[0]
            out_name = f"{base}_{x}_{y}.png"
            out_path = os.path.join(state_dir, out_name)
            
            # Use simple resize for training - no aggressive manipulation
            resized_crop = cv.resize(crop_enhanced, IMG_SIZE, interpolation=cv.INTER_AREA)
            
            # Save crop for training (BGR for OpenCV)
            cv.imwrite(out_path, cv.cvtColor(resized_crop, cv.COLOR_RGB2BGR))
            stats['saved'][state] += 1
            
            # Save preview with both original and enhanced versions
            preview_state_dir = os.path.join(preview_dir, state)
            os.makedirs(preview_state_dir, exist_ok=True)
            
            # Side-by-side preview
            preview_img = np.ones((80, 160, 3), dtype=np.uint8) * 255
            crop_h, crop_w = crop.shape[:2]
            
            # Calculate display size
            display_w = 40
            display_h = int(crop_h * (display_w / crop_w))
            if display_h > 60:
                display_h = 60
                display_w = int(crop_w * (display_h / crop_h))
            
            # Original and enhanced crops
            display_crop = cv.resize(crop, (display_w, display_h), interpolation=cv.INTER_AREA)
            display_enhanced = cv.resize(crop_enhanced, (display_w, display_h), interpolation=cv.INTER_AREA)
            
            # Paste both
            y_offset = (80 - display_h) // 2
            x_offset = (80 - display_w) // 2
            preview_img[y_offset:y_offset+display_h, x_offset:x_offset+display_w] = display_crop
            preview_img[y_offset:y_offset+display_h, 80+x_offset:80+x_offset+display_w] = display_enhanced
            
            # Add labels
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(preview_img, f"{w}x{h}", (5, 15), font, 0.4, (0, 0, 0), 1)
            cv.putText(preview_img, state, (5, 75), font, 0.4, (0, 0, 0), 1)
            
            # Save preview
            preview_path = os.path.join(preview_state_dir, out_name)
            cv.imwrite(preview_path, cv.cvtColor(preview_img, cv.COLOR_RGB2BGR))
        
        # Save debug image without aggressive contrast enhancement
        # Just a mild auto-contrast to preserve details
        debug_rgb_norm = debug_img.copy()
        for c in range(3):
            channel = debug_rgb_norm[:,:,c]
            p2 = np.percentile(channel, 2)
            p98 = np.percentile(channel, 98)
            if p98 > p2:
                channel = np.clip((channel - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
                debug_rgb_norm[:,:,c] = channel
        
        # Resize debug image to reasonable size
        if max(h_img, w_img) > 1200:
            scale = 1200 / max(h_img, w_img)
            debug_rgb_norm = cv.resize(debug_rgb_norm, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        
        cv.imwrite(os.path.join(debug_dir, img_name), cv.cvtColor(debug_rgb_norm, cv.COLOR_RGB2BGR))
    
    # Print statistics
    print("\nCropping statistics:")
    print(f"Processed: {stats['processed']} crops")
    print(f"Skipped tiny: {stats['skipped_tiny']} crops")
    print(f"Skipped dark/low contrast: {stats['skipped_dark']} crops")
    print(f"Skipped unknown states: {stats['skipped_unknown_state']} crops")
    print("Saved crops by state:")
    for state, count in stats['saved'].items():
        print(f"  {state}: {count}")


import warnings
from PIL import Image, ImageSequence
import numpy as np
import cv2

def robust_tiff_reader(filepath):
    """Robust TIFF reader that handles various formats and edge cases"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(filepath) as img:
                # Handle multi-page TIFFs
                if 'n_frames' in dir(img) and img.n_frames > 1:
                    img.seek(0)  # Get first frame
                
                # Handle 16-bit images
                if img.mode == 'I;16':
                    img = img.point(lambda i: i * (1./256)).convert('L')
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array and then to BGR for OpenCV
                img_array = np.array(img)
                return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    except Exception as e:
        # Fallback to tifffile if PIL fails
        try:
            import tifffile
            img = tifffile.imread(filepath)
            
            # Handle 16-bit images
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            
            # Handle grayscale
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)
            
            # Handle RGBA
            elif img.shape[2] == 4:
                img = img[..., :3]
            
            return img
        
        except Exception:
            # Final fallback to OpenCV
            img = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR)
            if img is not None:
                return img
            raise RuntimeError(f"All TIFF reading methods failed for {filepath}")

crop_dataset()