import os
import cv2
import json
import numpy as np
import tifffile
from tqdm import tqdm
from PIL import Image, ImageFile
import imagecodecs
from skimage import exposure

# Enable truncated image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configuration matching DTLD structure
DTLD_DIR = "dtld_dataset"
CROP_SIZE = (64, 64)
STATES = ["red", "yellow", "green", "off"]
VALID_DIRECTIONS = ["front"]
VALID_RELEVANCE = ["relevant"]

def load_dtld_image(path):
    """Load DTLD TIFF with specific handling for sparse data"""
    try:
        # Method 1: Use PIL - works best for DTLD
        try:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert('RGB')  # Force conversion to RGB
                np_img = np.array(pil_img)
                
                # Check if the image is too sparse (mostly black)
                non_zero = np.count_nonzero(np_img) / np_img.size
                
                # For extremely sparse images, apply special scaling
                if non_zero < 0.1:  # Less than 10% non-zero values
                    tqdm.write(f"Sparse image detected ({non_zero*100:.2f}% non-zero), enhancing...")
                    # Apply channel-wise enhancement for sparse images
                    for c in range(3):
                        channel = np_img[:,:,c]
                        if np.max(channel) > 0:
                            # Only consider non-zero values for percentile calculation
                            mask = channel > 0
                            if np.any(mask):
                                p99 = np.percentile(channel[mask], 99)
                                if p99 > 0:
                                    scale = 255.0 / p99
                                    channel = np.clip(channel * scale, 0, 255).astype(np.uint8)
                                    np_img[:,:,c] = channel
                
                return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        except Exception as e1:
            tqdm.write(f"PIL failed for {path}: {e1}")
        
        # Method 2: Direct OpenCV as fallback
        try:
            cv_img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
            if cv_img is not None and cv_img.size > 0:
                tqdm.write(f"Loaded {path} with OpenCV: {cv_img.shape}")
                return cv_img
        except Exception as e3:
            tqdm.write(f"OpenCV failed for {path}: {e3}")
            
        tqdm.write(f"All image loading methods failed for {path}")
        return None
    except Exception as e:
        tqdm.write(f"Error in load_dtld_image for {path}: {str(e)}")
        return None

def dtld_enhance(img):
    """Better enhancement for DTLD images with special handling for sparse data"""
    # Calculate percentage of non-zero pixels
    non_zero = np.count_nonzero(img) / img.size * 100
    
    # For sparse images (common in DTLD dataset)
    if non_zero < 10.0:  # Less than 10% non-zero pixels
        # Apply stronger contrast enhancement
        for c in range(3):
            channel = img[:,:,c]
            if np.std(channel) > 0:  # Only process if channel has variation
                # Use non-zero values to determine scaling
                mask = channel > 0
                if np.any(mask):
                    p99 = np.percentile(channel[mask], 99)
                    if p99 > 0:
                        scale = 255.0 / p99
                        channel = np.clip(channel * scale, 0, 255).astype(np.uint8)
                        img[:,:,c] = channel
    
    # Then apply CLAHE with adaptive settings based on image properties
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # More aggressive settings for sparse images
    clip_limit = 3.0 if non_zero < 5.0 else 1.5
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l_channel)
    
    # Merge enhanced L channel with original color
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def check_brightness(img_path):
    """Quick diagnostic to check image brightness issues with progress display"""
    output_dir = os.path.join(DTLD_DIR, "brightness_test")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing {img_path}...")
    
    try:
        # Try PIL first (most compatible)
        try:
            print("Attempting to load with PIL...")
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert('RGB')
                np_img = np.array(pil_img)
                print(f"PIL loaded image: {np_img.shape}, {np_img.dtype}")
                
                # Check if the image is too sparse (mostly black)
                non_zero = np.count_nonzero(np_img) / np_img.size
                print(f"Non-zero pixel percentage: {non_zero*100:.2f}%")
                
                if non_zero < 0.1:  # If less than 10% of pixels have values
                    print("WARNING: Image is extremely sparse, applying special scaling...")
                    # Boost contrast dramatically for sparse images
                    for c in range(3):
                        channel = np_img[:,:,c]
                        if np.max(channel) > 0:
                            p99 = np.percentile(channel[channel > 0], 99)
                            channel = np.clip(channel * 255.0 / p99, 0, 255).astype(np.uint8)
                            np_img[:,:,c] = channel
                
                # Save PIL version for comparison
                pil_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, "pil_version.jpg"), pil_bgr)
                print(f"Saved PIL version to {output_dir}/pil_version.jpg")
                
                # Save enhanced versions
                enhanced = pil_bgr.copy()
                # Apply autocontrast to each channel separately
                for c in range(3):
                    channel = enhanced[:,:,c]
                    if np.std(channel) > 0:
                        p1, p99 = np.percentile(channel, (1, 99))
                        if p99 > p1:
                            channel = np.clip((channel - p1) * 255.0 / (p99 - p1), 0, 255).astype(np.uint8)
                            enhanced[:,:,c] = channel
                
                cv2.imwrite(os.path.join(output_dir, "enhanced_pil.jpg"), enhanced)
                
                # Apply CLAHE for even more contrast enhancement
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                clahe_img = cv2.merge((cl, a, b))
                clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)
                cv2.imwrite(os.path.join(output_dir, "clahe_pil.jpg"), clahe_img)
                
                return True
                
        except Exception as e:
            print(f"PIL loading failed: {e}")
            
        # Try Direct OpenCV approach if PIL fails
        try:
            cv_img = cv2.imread(img_path)
            if cv_img is not None and cv_img.size > 0:
                print(f"OpenCV loaded image: {cv_img.shape}")
                non_zero = np.count_nonzero(cv_img) / cv_img.size
                print(f"Non-zero pixel percentage: {non_zero*100:.2f}%")
                
                # Save OpenCV version
                cv2.imwrite(os.path.join(output_dir, "opencv_version.jpg"), cv_img)
                return True
        except Exception as e:
            print(f"OpenCV loading failed: {e}")
            
        return False
            
    except Exception as e:
        print(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_dtld_crop(crop):
    """Special handling for DTLD's small traffic lights"""
    # Enhanced upscaling for small crops without using SR model
    if max(crop.shape[:2]) < 32:
        # Use Lanczos4 for high-quality upscaling
        h, w = crop.shape[:2]
        crop = cv2.resize(crop, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    
    # Maintain original aspect ratio
    h, w = crop.shape[:2]
    if h / w > 2.5:  # DTLD's vertical traffic lights
        crop = cv2.resize(crop, (CROP_SIZE[0], int(CROP_SIZE[0]*h/w)))
    elif w / h > 2:  # Horizontal configurations
        crop = cv2.resize(crop, (int(CROP_SIZE[1]*w/h), CROP_SIZE[1]))
    else:
        crop = cv2.resize(crop, CROP_SIZE)
    
    return crop

def parse_dtld_attributes(label):
    """Parse DTLD v2.0 JSON attributes"""
    attrs = label.get("attributes", {})
    return {
        "state": attrs.get("state", "unknown"),
        "direction": attrs.get("direction", "unknown"),
        "relevance": attrs.get("relevance", "unknown"),
        "pictogram": attrs.get("pictogram", "unknown")
    }

def crop_dataset():
    # Initialize directory structure
    output_dir = os.path.join(DTLD_DIR, "processed_crops")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create debug directory
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load DTLD annotations
    print("Loading annotations...")
    with open(os.path.join(DTLD_DIR, "Berlin.json")) as f:
        dataset = json.load(f)
    
    stats = {
        "total": 0,
        "valid": 0,
        "invalid_state": 0,
        "small_crop": 0,
        "processing_errors": 0,
        "by_state": {state: 0 for state in STATES}
    }

    # Process each image with DTLD-specific parameters
    for entry in tqdm(dataset["images"], desc="Processing images"):
        try:
            # Get the image path
            rel_path = entry["image_path"]
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
                
            img_path = os.path.join(DTLD_DIR, rel_path)
            
            # Skip if file doesn't exist
            if not os.path.exists(img_path):
                tqdm.write(f"File not found: {img_path}")
                stats["processing_errors"] += 1
                continue
                
            # Skip if no labels
            if not entry.get("labels"):
                continue
                
            # Load the image - progress will show on tqdm
            img = None
            
            # Try PIL first
            try:
                with Image.open(img_path) as pil_img:
                    pil_img = pil_img.convert('RGB')
                    np_img = np.array(pil_img)
                    img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    tqdm.write(f"Loaded {img_path} with PIL: {img.shape}")
            except Exception as e1:
                tqdm.write(f"PIL failed for {img_path}: {e1}")
                
                # Try tifffile page access
                try:
                    with tifffile.TiffFile(img_path) as tif:
                        page = tif.pages[0]
                        raw_img = page.asarray()
                        
                        # Handle 16-bit images with better scaling
                        if raw_img.dtype == np.uint16:
                            # Use percentile scaling for better visibility
                            p1, p99 = np.percentile(raw_img, (1, 99))
                            if p99 > p1:
                                img = np.clip((raw_img - p1) * 255.0 / (p99 - p1), 0, 255).astype(np.uint8)
                            else:
                                img = np.zeros_like(raw_img, dtype=np.uint8)
                        else:
                            img = raw_img
                            
                        # Ensure 3 channels
                        if img.ndim == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            
                        tqdm.write(f"Loaded {img_path} with tifffile: {img.shape}")
                except Exception as e2:
                    tqdm.write(f"tifffile failed for {img_path}: {e2}")
                    
                    # Try OpenCV as last resort
                    img = cv2.imread(img_path)
                    if img is not None:
                        tqdm.write(f"Loaded {img_path} with OpenCV: {img.shape}")
            
            # If we couldn't load the image, skip it
            if img is None or img.size == 0:
                tqdm.write(f"Failed to load image: {img_path}")
                stats["processing_errors"] += 1
                continue
            
            # Save a debug image with crops marked
            debug_img = img.copy()
            
            # Apply DTLD-specific enhancement
            enhanced = dtld_enhance(img)
            
            # Process each traffic light
            valid_crops_in_image = 0
            
            for label in tqdm(entry.get("labels", []), desc="Processing crops", leave=False):
                stats["total"] += 1
                attrs = parse_dtld_attributes(label)
                
                # Filter based on DTLD attributes
                if (attrs["state"] not in STATES or
                    attrs["direction"] not in VALID_DIRECTIONS or
                    attrs["relevance"] not in VALID_RELEVANCE):
                    stats["invalid_state"] += 1
                    continue
                
                # Extract traffic light coordinates
                x, y, w, h = label["x"], label["y"], label["w"], label["h"]
                
                # Mark on debug image
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(debug_img, attrs["state"], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 255), 2)
                
                # Calculate dynamic padding for tiny lights
                padding = int(max(w, h) * 2.0)  # Larger padding factor for better context
                
                # Extract crop with context
                x0 = max(x - padding, 0)
                y0 = max(y - padding, 0)
                x1 = min(x + w + padding, img.shape[1])
                y1 = min(y + h + padding, img.shape[0])
                crop = enhanced[y0:y1, x0:x1].copy()
                
                # Skip tiny or empty crops
                if crop.size == 0 or min(crop.shape[:2]) < 8:
                    stats["small_crop"] += 1
                    continue
                
                # Process crop with special handling
                processed = process_dtld_crop(crop)
                
                # Save in folder structure
                state_dir = os.path.join(output_dir, attrs["state"])
                os.makedirs(state_dir, exist_ok=True)
                filename = f"{os.path.basename(img_path)}_{x}_{y}.png"
                cv2.imwrite(os.path.join(state_dir, filename), processed)
                
                # Update stats
                stats["valid"] += 1
                stats["by_state"][attrs["state"]] += 1
                valid_crops_in_image += 1
            
            # Save debug image if we found any valid crops
            if valid_crops_in_image > 0:
                # Resize debug image if too large
                if max(debug_img.shape[:2]) > 1200:
                    scale = 1200 / max(debug_img.shape[:2])
                    debug_img = cv2.resize(debug_img, None, fx=scale, fy=scale, 
                                         interpolation=cv2.INTER_AREA)
                
                debug_filename = os.path.basename(img_path)
                cv2.imwrite(os.path.join(debug_dir, debug_filename), debug_img)
                
        except Exception as e:
            tqdm.write(f"Error processing {img_path}: {e}")
            stats["processing_errors"] += 1
            continue

    # Print DTLD-specific statistics with nice formatting
    print("\n" + "="*50)
    print("DTLD Processing Report:")
    print("="*50)
    print(f"Total annotations processed: {stats['total']}")
    print(f"Valid traffic light crops: {stats['valid']}")
    print(f"Skipped - Invalid state/direction: {stats['invalid_state']}")
    print(f"Skipped - Small crops: {stats['small_crop']}")
    print(f"Processing errors: {stats['processing_errors']}")
    print("\nValid crops by state:")
    for state in STATES:
        print(f"  - {state}: {stats['by_state'][state]}")
    print("="*50)

def create_diagnostic_report():
    """Create a diagnostic report about the DTLD dataset"""
    stats = {
        "total_images": 0,
        "accessible_images": 0,
        "failed_images": 0,
        "loaders": {
            "tifffile": 0,
            "pil": 0,
            "opencv": 0,
            "failed": 0
        }
    }
    
    # Load annotations
    with open(os.path.join(DTLD_DIR, "Berlin.json")) as f:
        dataset = json.load(f)
    
    # Create diagnostics directory
    diag_dir = os.path.join(DTLD_DIR, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    
    # Open file with UTF-8 encoding
    with open(os.path.join(diag_dir, "loading_report.txt"), "w", encoding="utf-8") as report:
        report.write("DTLD Dataset Loading Diagnostics\n")
        report.write("===============================\n\n")
        
        for i, entry in enumerate(tqdm(dataset["images"][:20], desc="Diagnosing")):  # Reduce to 20 instead of 100
            stats["total_images"] += 1
            
            rel_path = entry["image_path"]
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
                
            img_path = os.path.join(DTLD_DIR, rel_path)
            
            if not os.path.exists(img_path):
                report.write(f"[MISSING] {img_path} - FILE MISSING\n")
                stats["failed_images"] += 1
                continue
                
            # Try different loading methods and record which one works
            result = "[FAILED] Failed with all methods"
            
            # Method 1: PIL first (better for DTLD)
            try:
                with Image.open(img_path) as pil_img:
                    pil_img = pil_img.convert('RGB')
                    np_img = np.array(pil_img)
                    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    
                    result = "[SUCCESS] Successfully loaded with PIL"
                    stats["loaders"]["pil"] += 1
                    stats["accessible_images"] += 1
                    
                    # Save diagnostic JPEG
                    cv2.imwrite(os.path.join(diag_dir, f"sample_{i}_pil.jpg"), cv_img)
            except Exception as e1:
                # Method 2: Try tifffile with page access
                try:
                    with tifffile.TiffFile(img_path) as tif:
                        # Get first page only
                        page = tif.pages[0]
                        img = page.asarray()
                        
                        # Handle 16-bit images
                        if img.dtype == np.uint16:
                            img = np.clip((img - np.min(img)) * 255.0 / (np.max(img) - np.min(img)), 0, 255).astype(np.uint8)
                        
                        # Ensure 3 channels
                        if img.ndim == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        
                        result = "[SUCCESS] Successfully loaded with tifffile (page)"
                        stats["loaders"]["tifffile"] += 1
                        stats["accessible_images"] += 1
                        
                        cv2.imwrite(os.path.join(diag_dir, f"sample_{i}_tifffile.jpg"), img)
                except Exception as e2:
                    # Method 3: OpenCV
                    try:
                        cv_img = cv2.imread(img_path)
                        if cv_img is not None and cv_img.size > 0:
                            result = "[SUCCESS] Successfully loaded with OpenCV"
                            stats["loaders"]["opencv"] += 1
                            stats["accessible_images"] += 1
                            
                            # Save diagnostic JPEG
                            cv2.imwrite(os.path.join(diag_dir, f"sample_{i}_opencv.jpg"), cv_img)
                        else:
                            stats["loaders"]["failed"] += 1
                            stats["failed_images"] += 1
                    except Exception as e3:
                        stats["loaders"]["failed"] += 1
                        stats["failed_images"] += 1
            
            report.write(f"{result}: {img_path}\n")
        
        # Write summary statistics
        report.write("\n\nSummary Statistics\n")
        report.write("=================\n")
        report.write(f"Total images examined: {stats['total_images']}\n")
        if stats['total_images'] > 0:
            report.write(f"Successfully loaded: {stats['accessible_images']} ({stats['accessible_images']/stats['total_images']*100:.1f}%)\n")
            report.write(f"Failed to load: {stats['failed_images']} ({stats['failed_images']/stats['total_images']*100:.1f}%)\n\n")
        report.write("Loading method stats:\n")
        report.write(f"  tifffile: {stats['loaders']['tifffile']}\n")
        report.write(f"  PIL: {stats['loaders']['pil']}\n") 
        report.write(f"  OpenCV: {stats['loaders']['opencv']}\n")
        report.write(f"  Failed: {stats['loaders']['failed']}\n")
    
    print(f"Diagnostic report created in {diag_dir}")
    return stats

if __name__ == "__main__":
    # First test a sample TIFF file for brightness issues
    test_img = os.path.join(DTLD_DIR, "Berlin/Berlin1/2015-04-17_10-50-41/DE_BBBR667_2015-04-17_10-50-46-968138_k0.tiff")
    if os.path.exists(test_img):
        print("Testing brightness handling on a single image first...")
        check_brightness(test_img)
        print("\nBrightness test completed. Continuing with dataset processing...\n")

    # Version checks
    try:
        print(f"Using imagecodecs {imagecodecs.__version__}")
        print(f"Using OpenCV {cv2.__version__}")
        print(f"Using tifffile {tifffile.__version__}")
    except Exception as e:
        print(f"Warning during version check: {e}")
    
    # Jump straight to crop_dataset (skip diagnostic report for now)
    print("\nStarting crop process...")
    crop_dataset()