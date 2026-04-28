import cv2
import numpy as np
import sys
from ultralytics import YOLO
from config.config import LIGHT_DETECTION_CLASSIFICATION_MODEL

LIGHT_MODEL_PATH = str(LIGHT_DETECTION_CLASSIFICATION_MODEL)

def get_models_dict():
    try:
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'MODELS'):
            return main_module.MODELS
        return None
    except:
        return None

def detect_traffic_lights(frame, model=None):
    """
    Detect traffic lights in a single frame.
    Args:
        frame: numpy array image (BGR or RGB) - YOLO handles it, but usually expects BGR if loaded with cv2, or RGB if PIL. 
               BeamNG provides RGB usually (check beamng.py).
        model: Pre-loaded traffic light detection model
    Returns:
        list of detections
    """
    if model is None:
        models_dict = get_models_dict()
        
        if models_dict is not None and 'traffic_light' in models_dict:
            model = models_dict['traffic_light']
        else:
            try:
                model = YOLO(LIGHT_MODEL_PATH)
                print(f"Warning: Loading traffic light model from scratch - slower!")
            except Exception as e:
                print(f"Error loading traffic light model: {e}")
                return []
    
    # TUNE CONFIDENCE THRESHOLD HERE
    results = model.predict(
        source=frame, 
        verbose=False, 
        conf=0.2, 
        iou=0.45,
    )
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class': class_name,
                'confidence': confidence,
                'class_id': class_id
            })
            
    return detections
