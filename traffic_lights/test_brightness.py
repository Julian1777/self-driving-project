import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random

# Constants for cropping
CROP_LEFT_RIGHT = 25
CROP_TOP_BOTTOM = 10
IMG_SIZE = (64, 64)

def get_brightness_vector(image):
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    
    height, width, _ = image.shape
    cropped_v = hsv[CROP_TOP_BOTTOM:height-CROP_TOP_BOTTOM, 
                    CROP_LEFT_RIGHT:width-CROP_LEFT_RIGHT, 2]
    
    row_brightness = np.mean(cropped_v, axis=1)
    
    return row_brightness

def create_brightness_debug(image, filename):
    hsv = cv.cvtColor(image.astype(np.uint8), cv.COLOR_RGB2HSV)
    
    height, width, _ = image.shape
    
    cropped_img = image[CROP_TOP_BOTTOM:height-CROP_TOP_BOTTOM, CROP_LEFT_RIGHT:width-CROP_LEFT_RIGHT]
    cropped_v = hsv[CROP_TOP_BOTTOM:height-CROP_TOP_BOTTOM, CROP_LEFT_RIGHT:width-CROP_LEFT_RIGHT, 2]
    
    row_brightness = np.mean(cropped_v, axis=1)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(cropped_img)
    plt.title("Cropped Image")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(cropped_v, cmap='gray')
    plt.title("Brightness (V)")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    y = np.arange(len(row_brightness))
    plt.barh(y, row_brightness)
    plt.gca().invert_yaxis()
    
    section_size = len(row_brightness) // 3
    plt.axhline(y=section_size, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=section_size*2, color='r', linestyle='--', alpha=0.5)
    
    plt.text(-5, section_size//2, "RED", ha='right', va='center', color='red', fontsize=12)
    plt.text(-5, section_size + section_size//2, "YELLOW", ha='right', va='center', color='orange', fontsize=12)
    plt.text(-5, 2*section_size + section_size//2, "GREEN", ha='right', va='center', color='green', fontsize=12)
    
    plt.title("Row Brightness")
    
    plt.tight_layout()
    
    debug_dir = os.path.join("debug_visualizations", "brightness_test")
    os.makedirs(debug_dir, exist_ok=True)
    plt.savefig(os.path.join(debug_dir, filename))
    plt.close()

def analyze_brightness_pattern(image, true_label=None):
    brightness = get_brightness_vector(image)
    
    section_size = len(brightness) // 3
    
    top_section = np.sum(brightness[:section_size])
    middle_section = np.sum(brightness[section_size:2*section_size])
    bottom_section = np.sum(brightness[2*section_size:])
    
    sections = [top_section, middle_section, bottom_section]
    brightest_section = np.argmax(sections)
    
    states = ["red", "yellow", "green"]
    predicted_state = states[brightest_section]
    
    total_brightness = sum(sections)
    if total_brightness > 0:
        confidences = [s/total_brightness for s in sections]
    else:
        confidences = [0.33, 0.33, 0.33]
    
    result = {
        "predicted_state": predicted_state,
        "confidence": confidences[brightest_section],
        "section_brightness": {
            "red": sections[0],
            "yellow": sections[1],
            "green": sections[2]
        },
        "normalized_brightness": {
            "red": confidences[0],
            "yellow": confidences[1],
            "green": confidences[2]
        }
    }
    
    if true_label is not None:
        result["true_state"] = true_label
        result["is_correct"] = predicted_state == true_label
    
    return result

def extract_state_from_filename(filename):
    """Extract the traffic light state from filename"""
    states = ["red", "yellow", "green"]
    
    for state in states:
        if state in filename.lower():
            return state
    
    return None

def test_with_cropped_images(test_images_dir):
    images_data = []
    
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found: {test_images_dir}")
        return
    
    files = os.listdir(test_images_dir)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} image files in {test_images_dir}")
    
    for file in image_files:
        file_path = os.path.join(test_images_dir, file)
        try:
            true_label = extract_state_from_filename(file)
            
            if true_label is None:
                print(f"Warning: Could not determine state for {file}, skipping")
                continue
            
            img = cv.imread(file_path)
            if img is None:
                print(f"Warning: Could not read {file_path}")
                continue
                
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            if img.shape[0] != IMG_SIZE[0] or img.shape[1] != IMG_SIZE[1]:
                img = cv.resize(img, IMG_SIZE)
            
            images_data.append({
                "image": img,
                "filename": file,
                "true_label": true_label
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Loaded {len(images_data)} images for testing")
    
    correct_count = 0
    
    for i, data in enumerate(images_data):
        img = data["image"]
        true_label = data["true_label"]
        filename = data["filename"]
        
        result = analyze_brightness_pattern(img, true_label)
        
        debug_filename = f"{i+1}_{true_label}_pred_{result['predicted_state']}.jpg"
        create_brightness_debug(img, debug_filename)
        
        if result.get('is_correct', False):
            correct_count += 1
        
        print(f"Image {i+1}: {filename}")
        print(f"  True state: {true_label}")
        print(f"  Predicted state: {result['predicted_state']} (confidence: {result['confidence']:.2f})")
        print(f"  Section brightness: Red={result['section_brightness']['red']:.1f}, "
                f"Yellow={result['section_brightness']['yellow']:.1f}, "
                f"Green={result['section_brightness']['green']:.1f}")
        print(f"  Normalized: Red={result['normalized_brightness']['red']:.2f}, "
                f"Yellow={result['normalized_brightness']['yellow']:.2f}, "
                f"Green={result['normalized_brightness']['green']:.2f}")
        print(f"  Result: {'✓ Correct' if result.get('is_correct', False) else '✗ Incorrect'}")
        print()
    
    if images_data:
        accuracy = correct_count / len(images_data) * 100
        print(f"Overall accuracy: {correct_count}/{len(images_data)} ({accuracy:.1f}%)")

if __name__ == "__main__":
    merged_dataset_path = "merged_dataset"
    test_with_random_images(merged_dataset_path, num_images=20)
    
    print("Brightness testing complete.")
    print("Visualizations saved to debug_visualizations/brightness_test/")