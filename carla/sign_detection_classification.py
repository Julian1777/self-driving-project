import cv2 as cv
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
import tensorflow as tf
import sys
from vehicle_pedestrian_detection import detect_vehicles_pedestrians

IMG_SIZE = (224, 224)
SIGN_MODEL_PATH = os.path.join("model", "sign_detection")
SIGN_CLASSIFY_MODEL_PATH = os.path.join("model", "sign_classification.h5")

def get_models_dict():
    try:
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'MODELS'):
            return main_module.MODELS
        return None
    except:
        return None

def load_class_names(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Loading class names from {csv_path}")
        print("CSV columns found:", df.columns.tolist())
        
        class_names = {}
        id_column = 'id'
        desc_column = 'description'
        
        for _, row in df.iterrows():
            class_id = row[id_column]
            name = row[desc_column]
            class_names[str(class_id)] = name
            
        print(f"Loaded {len(class_names)} class names")
        return class_names
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return {}

class_names_dict = load_class_names("sign_dic.csv")
print(class_names_dict)

num_classes = max(map(int, class_names_dict.keys())) + 1
class_descriptions = ["Unknown Class"] * num_classes
for class_id, description in class_names_dict.items():
    class_id = int(class_id)
    if 0 <= class_id < num_classes:
        class_descriptions[class_id] = description

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

def img_preprocessing(frame):
    img = frame.copy()
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_classify_sign(frame):
    models_dict = get_models_dict()
    
    if models_dict is not None and 'sign_detect' in models_dict:
        detection_model = models_dict['sign_detect']
    else:
        detection_model = YOLO(SIGN_MODEL_PATH)
        print(f"Warning: Loading sign detection model from scratch - slower!")
    
    if models_dict is not None and 'sign_classify' in models_dict:
        classification_model = models_dict['sign_classify']
    else:
        classification_model = tf.keras.models.load_model(SIGN_CLASSIFY_MODEL_PATH)
        print(f"Warning: Loading sign classification model from scratch - slower!")

    results = detection_model(frame, conf=0.2)

    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            class_id = int(box.cls[0])
            class_name = detection_model.names[class_id]
            confidence = float(box.conf[0])
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:  # Ensure valid crop dimensions
                sign_crop = frame[y1:y2, x1:x2]
                
                img = cv.resize(sign_crop, IMG_SIZE)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                
                try:
                    pred = classification_model.predict(img, verbose=0)
                    class_idx = np.argmax(pred[0])
                    class_confidence = float(pred[0][class_idx])
                    
                    if 0 <= class_idx < len(class_descriptions):
                        classification = class_descriptions[class_idx]
                    else:
                        classification = f"Class {class_idx}"
                        
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'detection_class': class_name,
                        'detection_confidence': confidence,
                        'classification': classification,
                        'classification_confidence': class_confidence
                    })
                    
                except Exception as e:
                    print(f"Error during classification: {e}")
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'detection_class': class_name,
                        'detection_confidence': confidence,
                        'classification': "Classification failed",
                        'classification_confidence': 0.0
                    })
            
    return detections

def combined_sign_detection_classification(frame):
    sign_model_detections = detect_classify_sign(frame)

    vehicle_model_detections = detect_vehicles_pedestrians(frame, include_traffic_lights=False, include_traffic_signs=True)

    classes_found = set()
    for detection in vehicle_model_detections:
        classes_found.add(detection['class'])
    print(f"Vehicle model found classes: {classes_found}")

    vehicle_model_sign_detections = []

    for detection in vehicle_model_detections:
        if 'traffic sign' in detection['class']:
            detection['source'] = 'vehicle_model'
            detection['class'] = 'unknown'
            vehicle_model_sign_detections.append(detection)

    final_detections = []

    for sign_det in sign_model_detections:
        sign_det['source'] = 'sign_model'
        sign_det['verified'] = False
        final_detections.append(sign_det)

    for veh_det in vehicle_model_sign_detections:
        is_new = True
        
        for sign_det in sign_model_detections:
            iou = calculate_iou(veh_det['bbox'], sign_det['bbox'])
            
            if iou > 0.3:
                is_new = False
                for i, det in enumerate(final_detections):
                    if det is sign_det:
                        final_detections[i]['detection_confidence'] *= 1.1
                        final_detections[i]['detection_confidence'] = min(0.99, final_detections[i]['detection_confidence'])
                        final_detections[i]['verified'] = True
                        break
                break
        
        if is_new and veh_det['confidence'] > 0.5:
            try:
                x1, y1, x2, y2 = veh_det['bbox']
                sign_crop = frame[y1:y2, x1:x2]
                
                models_dict = get_models_dict()
                if models_dict is not None and 'sign_classify' in models_dict:
                    classification_model = models_dict['sign_classify']
                else:
                    classification_model = tf.keras.models.load_model(SIGN_CLASSIFY_MODEL_PATH)
                
                img = cv.resize(sign_crop, IMG_SIZE)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                
                pred = classification_model.predict(img, verbose=0)
                class_idx = np.argmax(pred[0])
                class_confidence = float(pred[0][class_idx])
                
                if 0 <= class_idx < len(class_descriptions):
                    classification = class_descriptions[class_idx]
                else:
                    classification = f"Class {class_idx}"
                
                enhanced_det = {
                    'bbox': veh_det['bbox'],
                    'detection_class': veh_det['class'],
                    'detection_confidence': veh_det['confidence'],
                    'classification': classification,
                    'classification_confidence': class_confidence,
                    'source': 'vehicle_model',
                    'verified': False
                }
                
                final_detections.append(enhanced_det)
                
            except Exception as e:
                print(f"Error classifying vehicle model sign: {e}")
    
    print(f"Combined sign detection returning {len(final_detections)} signs")
    return final_detections
