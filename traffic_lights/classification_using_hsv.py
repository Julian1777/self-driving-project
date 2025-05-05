import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random

# Constants for cropping
CROP_LEFT_RIGHT = 12
CROP_TOP_BOTTOM = 3
IMG_SIZE = (64, 64)

def get_brightness_vector(image):
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    
    height, width, _ = image.shape
    cropped_v = hsv[CROP_TOP_BOTTOM:height-CROP_TOP_BOTTOM, 
                    CROP_LEFT_RIGHT:width-CROP_LEFT_RIGHT, 2]
    
    row_brightness = np.mean(cropped_v, axis=1)
    
    return row_brightness


def analyze_brightness_pattern(image, true_label=None):
    """Analyze brightness pattern and make a simple classification"""
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

def test_with_random_images(merged_dataset_path, num_images=5):
    if not os.path.exists(merged_dataset_path):
        print(f"Error: Dataset path {merged_dataset_path} not found.")
        return
    
    state_dirs = [d for d in os.listdir(merged_dataset_path) 
                  if os.path.isdir(os.path.join(merged_dataset_path, d))]
    
    images_data = []
    
    num_per_state = max(1, num_images // len(state_dirs))
    
    for state in state_dirs:
        state_path = os.path.join(merged_dataset_path, state)
        image_files = [f for f in os.listdir(state_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            continue
            
        selected_files = random.sample(image_files, min(num_per_state, len(image_files)))
        
        for file in selected_files:
            file_path = os.path.join(state_path, file)
            try:
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
                    "true_label": state
                })
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Loaded {len(images_data)} images for testing")
    
    for i, data in enumerate(images_data):
        img = data["image"]
        true_label = data["true_label"]
        filename = data["filename"]
        
        result = analyze_brightness_pattern(img, true_label)
        
        # Print analysis result
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

if __name__ == "__main__":
    merged_dataset_path = "merged_dataset"
    test_with_random_images(merged_dataset_path, num_images=5)
    
    print("Brightness testing complete.")
    print("Visualizations saved to debug_visualizations/brightness_test/")