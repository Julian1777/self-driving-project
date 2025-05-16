import cv2 as cv
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
import tensorflow as tf
import sys

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

    results = detection_model(frame, conf=0.25)

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