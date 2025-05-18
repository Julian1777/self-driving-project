import cv2 as cv
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
import tensorflow as tf
import sys

IMG_SIZE = (224, 224)
DETECTION_MODEL_PATH = os.path.join("model", "vehicle_pedestrian_detection.pt")

def get_models_dict():
    try:
        # Try to get the models from the main module
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'MODELS'):
            return main_module.MODELS
        return None
    except:
        return None

def detect_vehicles_pedestrians(frame, model=None, include_traffic_lights=True, include_traffic_signs=True):
    if model is None:
        models_dict = get_models_dict()
        if models_dict is not None and 'vehicle' in models_dict:
            model = models_dict['vehicle']
        else:
            model = YOLO(DETECTION_MODEL_PATH)
            print(f"Warning: Loading vehicle detection model from scratch - slower!")

    target_classes = ["car", "truck", "bus", "motor", "bike", "person"]

    if include_traffic_lights:
        target_classes.extend(["traffic light"])

    if include_traffic_signs:
        target_classes.extend(["traffic signs"])

    results = model(frame, conf=0.30)

    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            if class_name.lower() in target_classes:
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': class_name,
                    'confidence': confidence,
                    'source': 'vehicle_model'
                })
    
    return detections

