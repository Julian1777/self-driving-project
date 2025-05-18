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

def classify_traffic_light_crop(image_crop):
    if image_crop.shape[0] != IMG_SIZE[0] or image_crop.shape[1] != IMG_SIZE[1]:
        image_crop = cv.resize(image_crop, IMG_SIZE)
    
    # Analyze brighness pattern
    result = analyze_brightness_pattern(image_crop)
    
    return {
        "class": result["predicted_state"],
        "confidence": result["confidence"],
        "probabilities": {
            "red": result["normalized_brightness"]["red"],
            "yellow": result["normalized_brightness"]["yellow"],
            "green": result["normalized_brightness"]["green"]
        }
    }