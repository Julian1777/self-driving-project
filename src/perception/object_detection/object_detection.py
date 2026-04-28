import cv2 as cv
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
import tensorflow as tf
import sys
from config.config import OBJECT_DETECTION_MODEL

IMG_SIZE = (224, 224)
DETECTION_MODEL_PATH = str(OBJECT_DETECTION_MODEL)

def get_models_dict():
    try:
        # Try to get the models from the main module
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'MODELS'):
            return main_module.MODELS
        return None
    except:
        return None

def detect_objects(frame, model=None, confidence_threshold=0.40):
    if model is None:
        models_dict = get_models_dict()
        if models_dict is not None and 'vehicle' in models_dict:
            model = models_dict['vehicle']
        else:
            try:
                model = YOLO(DETECTION_MODEL_PATH)
                print(f"Warning: Loading vehicle detection model from scratch - slower!")
            except Exception as e:
                print(f"Error loading vehicle detection model: {e}")
                return []

    results = model(frame, conf=confidence_threshold)

    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class': class_name,
                'confidence': confidence,
                'source': 'vehicle_model'
                })
    
    return detections

