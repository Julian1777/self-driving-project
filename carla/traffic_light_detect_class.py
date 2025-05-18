import os
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import cv2 as cv

import sys

from traffic_light_class_cv import classify_traffic_light_crop
from vehicle_pedestrian_detection import detect_vehicles_pedestrians

IMG_SIZE = (224, 224)
LIGHT_MODEL_PATH = os.path.join("model", "traffic_light_detect_class.pt")

def get_models_dict():
    try:
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'MODELS'):
            return main_module.MODELS
        return None
    except:
        return None
    
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area

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

            traffic_light_crop = frame[y1:y2, x1:x2]

            cv_classification = classify_traffic_light_crop(traffic_light_crop)

            cv_light_state = cv_classification["class"]
            cv_confidence = cv_classification["confidence"]
                
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class': cv_light_state,
                'yolo_class': light_state,
                'confidence': cv_confidence,
                'yolo_confidence': confidence,
                'probabilities': cv_classification["probabilities"]
            })
    print(f"Returning {len(detections)} traffic lights: {detections}")
    return detections

def combined_traffic_light_detection(frame):
    tl_model_detections = detect_classify_traffic_light(frame)
    vehicle_model_detections = detect_vehicles_pedestrians(frame, include_traffic_lights=True, include_traffic_signs=False)

    vehicle_model_tl_detections = []
    for detection in vehicle_model_detections:
        if 'traffic light' in detection['class']:
            detection['class'] = 'unknown'
            vehicle_model_tl_detections.append(detection)

    final_detections = []

    for tl_det in tl_model_detections:
        tl_det['source'] = 'traffic_light_model'
        final_detections.append(tl_det)

    for veh_det in vehicle_model_tl_detections:
        is_new = True
        
        for tl_det in tl_model_detections:
            iou = calculate_iou(veh_det['bbox'], tl_det['bbox'])
            
            if iou > 0.3:
                is_new = False
                matching_idx = final_detections.index(tl_det)
                final_detections[matching_idx]['confidence'] *= 1.2
                final_detections[matching_idx]['confidence'] = min(0.99, final_detections[matching_idx]['confidence'])  # Cap at 0.99
                final_detections[matching_idx]['verified'] = True
                break
        
        if is_new and veh_det['confidence'] > 0.4:
            try:
                x1, y1, x2, y2 = veh_det['bbox']
                light_crop = frame[y1:y2, x1:x2]
                cv_result = classify_traffic_light_crop(light_crop)
                
                veh_det['class'] = cv_result['class']
                veh_det['confidence'] = cv_result['confidence'] * 0.8
                veh_det['source'] = 'vehicle_model'
                final_detections.append(veh_det)
            except Exception as e:
                print(f"Error classifying vehicle model traffic light: {e}")
    
    print(f"Combined traffic light detection found {len(final_detections)} lights")
    return final_detections
