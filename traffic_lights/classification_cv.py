import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random
from tqdm import tqdm
import tensorflow as tf

CROP_LEFT_RIGHT = 20
CROP_TOP_BOTTOM = 5
IMG_SIZE = (64, 64)
STATES = ["red", "yellow", "green"]
MERGED_DS = "merged_dataset"

def extract_brightness_features(image):
    hsv = tf.image.rgb_to_hsv(image)
    v_channel = hsv[:, :, :, 2]
    cropped_v = v_channel[:, CROP_TOP_BOTTOM:tf.shape(image)[1]-CROP_TOP_BOTTOM, 
                        CROP_LEFT_RIGHT:tf.shape(image)[2]-CROP_LEFT_RIGHT]
    row_brightness = tf.reduce_mean(cropped_v, axis=2)
    return row_brightness

def get_brightness_vector(image):
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    
    height, width, _ = image.shape
    cropped_v = hsv[CROP_TOP_BOTTOM:height-CROP_TOP_BOTTOM, 
                    CROP_LEFT_RIGHT:width-CROP_LEFT_RIGHT, 2]
    
    row_brightness = np.mean(cropped_v, axis=1)
    
    return row_brightness

def analyze_brightness_pattern(image, true_label=None):
    brightness = get_brightness_vector(image)
    
    section_size = len(brightness) // 3
    
    top_section = np.sum(brightness[:section_size])
    middle_section = np.sum(brightness[section_size:2*section_size])
    bottom_section = np.sum(brightness[2*section_size:])
    
    sections = [top_section, middle_section, bottom_section]
    brightest_section = np.argmax(sections)
    
    predicted_state = STATES[brightest_section]
    
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

def predict_light(image_path):
    img = cv.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}
        
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    if img.shape[0] != IMG_SIZE[0] or img.shape[1] != IMG_SIZE[1]:
        img = cv.resize(img, IMG_SIZE)
    
    result = analyze_brightness_pattern(img)
    
    return {
        "class": result["predicted_state"],
        "confidence": result["confidence"],
        "probabilities": {
            "red": result["normalized_brightness"]["red"],
            "yellow": result["normalized_brightness"]["yellow"],
            "green": result["normalized_brightness"]["green"]
        }
    }


test_images_dir = "test_images/cropped"
image_files = [f for f in os.listdir(test_images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in image_files:
    img_path = os.path.join(test_images_dir, img_file)
    prediction = predict_light(img_path)
    print(f"Predicted: {prediction['class']} with {prediction['confidence']:.2f} confidence")


plt.figure(figsize=(15, 5 * len(image_files)))

for i, img_file in enumerate(image_files):
    img_path = os.path.join(test_images_dir, img_file)
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    actual_class = None
    for state in STATES:
        if state in img_file.lower():
            actual_class = state
            break
    
    if actual_class is None:
        actual_class = "unknown"
    
    prediction = predict_light(img_path)
    
    print(f"File: {img_file}")
    print(f"  Actual: {actual_class}")
    print(f"  Predicted: {prediction['class']} with {prediction['confidence']:.2f} confidence")
    
    plt.subplot(len(image_files), 2, 2*i+1)
    plt.imshow(img)
    
    correct = actual_class == prediction['class']
    title_color = 'green' if correct else 'red'
    title = f"Actual: {actual_class}\nPredicted: {prediction['class']}"
    plt.title(title, color=title_color)
    plt.axis('off')
    
    plt.subplot(len(image_files), 2, 2*i+2)
    
    brightness = get_brightness_vector(img)
    x = range(len(brightness))
    plt.plot(x, brightness)
    
    section_size = len(brightness) // 3
    for j, color in enumerate(['red', 'yellow', 'green']):
        start = j * section_size
        end = (j + 1) * section_size if j < 2 else len(brightness)
        plt.fill_between(range(start, end), brightness[start:end], alpha=0.3, color=color)
    
    plt.title("Brightness Profile")
    plt.xlabel("Vertical Position")
    plt.ylabel("Brightness")
    
    plt.text(1.05, 0.8, f"Red: {prediction['probabilities']['red']:.2f}", 
            transform=plt.gca().transAxes, color='red')
    plt.text(1.05, 0.6, f"Yellow: {prediction['probabilities']['yellow']:.2f}", 
            transform=plt.gca().transAxes, color='orange')
    plt.text(1.05, 0.4, f"Green: {prediction['probabilities']['green']:.2f}", 
            transform=plt.gca().transAxes, color='green')

plt.tight_layout()
plt.show()