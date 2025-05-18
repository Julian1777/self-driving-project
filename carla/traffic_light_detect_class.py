import os
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import cv2 as cv

import sys

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

def detect_classify_traffic_light(frame):
    models_dict = get_models_dict()
    
    if models_dict is not None and 'traffic_light' in models_dict:
        model = models_dict['traffic_light']
    else:
        model = YOLO(LIGHT_MODEL_PATH)
        print(f"Warning: Loading traffic light model from scratch - slower!")

    
    results = model(frame, conf=0.15)

    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            light_state = class_name.lower()

            if "red" in light_state or "stop" in light_state:
                light_state = "red"
            elif "yellow" in light_state or "amber" in light_state:
                light_state = "yellow"
            elif "green" in light_state or "go" in light_state:
                light_state = "green"
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class': light_state,
                'confidence': confidence
            })
    print(f"Returning {len(detections)} traffic lights: {detections}")
    return detections
