import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

# Configuration matching DTLD structure
DTLD_DIR = "dtld_dataset"
ANNOTATION_FILE = os.path.join(DTLD_DIR, "Berlin.json")
CROP_SIZE = (64, 64)
STATES = ["red", "yellow", "green", "off"]
VALID_DIRECTIONS = ["front"]
VALID_RELEVANCE = ["relevant"]
CROP_DS = os.path.join(DTLD_DIR, "cropped_dataset")
PADDING = 5


def process_dtld_dataset():
    """Process DTLD dataset by cropping and organizing traffic light images"""
    print(f"Processing DTLD dataset from {DTLD_DIR}")
    
    # Create output directories
    for state in STATES:
        os.makedirs(os.path.join(CROP_DS, state), exist_ok=True)
    
    # Check if annotation file exists
    if not os.path.exists(ANNOTATION_FILE):
        print(f"Error: Annotation file not found: {ANNOTATION_FILE}")
        return
    
    print(f"Loading annotation file: {ANNOTATION_FILE}")
    
    # Track statistics
    processed_images = 0
    cropped_lights = {state: 0 for state in STATES}
    
    try:
        # Load the annotation file
        with open(ANNOTATION_FILE, 'r') as f:
            data = json.load(f)
        
        # Print structure information for debugging
        print(f"Annotation file loaded. Structure:")
        for key in data.keys():
            if isinstance(data[key], list):
                print(f"  - {key}: list with {len(data[key])} items")
            else:
                print(f"  - {key}: {type(data[key])}")
        
        # Process each image entry
        if "images" in data and isinstance(data["images"], list):
            images = data["images"]
            print(f"Found {len(images)} image entries")
            
            # Sample one image entry to understand structure
            if images:
                print("\nSample image entry structure:")
                for key, value in images[0].items():
                    print(f"  - {key}: {type(value)}")
                
                # If labels exist, print a sample label
                if "labels" in images[0] and images[0]["labels"]:
                    print("\nSample label structure:")
                    for key, value in images[0]["labels"][0].items():
                        print(f"  - {key}: {type(value)}")
                    
                    if "attributes" in images[0]["labels"][0]:
                        print("\nSample attributes structure:")
                        for key, value in images[0]["labels"][0]["attributes"].items():
                            print(f"  - {key}: {value}")
            
            # Process each image in the dataset
            for image_entry in tqdm(images, desc="Processing images"):
                rel_path = image_entry.get("image_path", "")
                if not rel_path:
                    continue
                
                img_path = os.path.join(DTLD_DIR, rel_path)
                
                # Check if image exists
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found: {img_path}")
                    continue
                
                try:
                    # Load image with PIL
                    pil_image = Image.open(img_path)
                    
                    # Log details for every 50th image
                    if processed_images % 50 == 0:
                        print(f"\nProcessing image {processed_images}: {img_path}")
                        print(f"  Format: {pil_image.format}, Mode: {pil_image.mode}, Size: {pil_image.size}")
                    
                    # Convert to numpy array
                    np_img = np.array(pil_image)
                    
                    # Handle 16-bit images by normalizing to 8-bit
                    if np_img.dtype == np.uint16:
                        print(f"Converting 16-bit image: min={np_img.min()}, max={np_img.max()}")
                        np_img = ((np_img - np_img.min()) * 255.0 / (np_img.max() - np_img.min())).astype(np.uint8)
                    
                    # Ensure image has 3 channels
                    if len(np_img.shape) == 2:
                        np_img = np.stack([np_img] * 3, axis=-1)
                    
                    h_img, w_img = np_img.shape[:2]
                    processed_images += 1
                    
                    # Process each traffic light annotation
                    for label in image_entry.get("labels", []):
                        attr = label.get("attributes", {})
                        
                        # Filter by relevance and direction
                        if (attr.get("relevance") not in VALID_RELEVANCE or 
                            attr.get("direction") not in VALID_DIRECTIONS):
                            continue
                        
                        # Get traffic light state and bounding box
                        state = attr.get("state", "")
                        if state not in STATES:
                            continue
                            
                        x, y, w, h = label.get("x", 0), label.get("y", 0), label.get("w", 0), label.get("h", 0)
                        
                        # Add padding and ensure within image bounds
                        x0 = max(int(x - PADDING), 0)
                        y0 = max(int(y - PADDING), 0)
                        x1 = min(int(x + w + PADDING), w_img)
                        y1 = min(int(y + h + PADDING), h_img)
                        
                        # Skip if box is invalid
                        if x1 <= x0 or y1 <= y0:
                            continue
                        
                        # Crop the traffic light
                        crop = np_img[y0:y1, x0:x1]
                        if crop.size == 0:
                            continue
                        
                        # Resize to standard size
                        crop_resized = cv2.resize(
                            crop,
                            CROP_SIZE,
                            interpolation=cv2.INTER_CUBIC
                        )
                        
                        # Convert RGB to BGR for OpenCV
                        if pil_image.mode == 'RGB':
                            crop_resized = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR)
                        
                        # Create output filename
                        base = os.path.splitext(os.path.basename(rel_path))[0]
                        out_name = f"{base}_{int(x)}_{int(y)}.jpg"
                        out_path = os.path.join(CROP_DS, state, out_name)
                        
                        # Save the cropped image
                        cv2.imwrite(out_path, crop_resized)
                        cropped_lights[state] += 1
                        
                        # Print debug info for every 100th light
                        if sum(cropped_lights.values()) % 100 == 0:
                            print(f"Saved {state} crop: {out_path}")
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    
        else:
            print("No 'images' list found in the annotation file")
            
    except Exception as e:
        print(f"Error processing annotation file: {e}")
    
    print("\n--- Processing Complete ---")
    print(f"Processed {processed_images} images")
    print("Cropped traffic lights by state:")
    for state in STATES:
        print(f"  - {state}: {cropped_lights[state]} images")
    print(f"Total: {sum(cropped_lights.values())} cropped traffic lights")
    print(f"Results saved to {CROP_DS}")


if __name__ == "__main__":
    process_dtld_dataset()