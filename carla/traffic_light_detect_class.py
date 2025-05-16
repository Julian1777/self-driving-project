import os
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import cv2 as cv

import sys
import importlib.util

def get_models_dict():
    try:
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'MODELS'):
            return main_module.MODELS
        return None
    except:
        return None

IMG_SIZE = (224, 224)
LIGHT_MODEL_PATH = os.path.join("model", "traffic_light_detect_class.pt")

def img_preprocessing(image_input):
    if isinstance(image_input, str):
        # If it's a file path
        image = cv.imread(image_input)
        if image is None:
            raise ValueError(f"Could not read image from {image_input}")
    else:
        image = image_input.copy()
    
    if len(image.shape) == 2:  # Grayscale
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
    elif image.shape[2] == 3 and image_input is not str:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, IMG_SIZE)
    image = image / 255.0  # Normalize to [0, 1]
    
    image = np.expand_dims(image, axis=0)
    return image

def detect_classify_traffic_light(frame):
    models_dict = get_models_dict()
    
    if models_dict is not None and 'traffic_light' in models_dict:
        model = models_dict['traffic_light']
    else:
        model = YOLO(LIGHT_MODEL_PATH)
        print(f"Warning: Loading traffic light model from scratch - slower!")

    
    results = model(frame, conf=0.25)

    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if "traffic" in class_name.lower() and "light" in class_name.lower():
                # Determine light state from class name
                light_state = "unknown"
                if "red" in class_name.lower():
                    light_state = "red"
                elif "yellow" in class_name.lower() or "amber" in class_name.lower():
                    light_state = "yellow"
                elif "green" in class_name.lower():
                    light_state = "green"
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': light_state,
                    'confidence': confidence
                })
    return detections
