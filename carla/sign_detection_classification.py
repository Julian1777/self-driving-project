import cv2 as cv
import numpy as np
import os
import pandas as pd
import uuid
from ultralytics import YOLO
import tensorflow as tf
import sys
from PIL import Image
from vehicle_pedestrian_detection import detect_vehicles_pedestrians

IMG_SIZE = (30, 30)
SIGN_MODEL_PATH = os.path.join("model", "sign_detection.pt")
SIGN_CLASSIFY_MODEL_PATH = os.path.join("model", "traffic_sign_classification.h5")

def get_models_dict():
    try:
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'MODELS'):
            return main_module.MODELS
        return None
    except:
        return None
    
def random_brightness(x):
    return tf.image.random_brightness(x, max_delta=0.2)

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

def classify_sign_crop(sign_crop):
    try:
        if sign_crop.shape[2] == 3:
            sign_crop_rgb = cv.cvtColor(sign_crop, cv.COLOR_BGR2RGB)
        else:
            sign_crop_rgb = sign_crop

        save_dir = "debug_sign_crops"
        os.makedirs(save_dir, exist_ok=True)
        unique_name = f"crop_{uuid.uuid4().hex[:8]}.png"
        Image.fromarray(sign_crop_rgb).save(os.path.join(save_dir, unique_name))
        
        img_pil = Image.fromarray(sign_crop_rgb)
        img_pil = img_pil.resize(IMG_SIZE)
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img = np.expand_dims(img_np, axis=0)
        
        models_dict = get_models_dict()
        if models_dict is not None and 'sign_classify' in models_dict:
            classification_model = models_dict['sign_classify']
        else:
            classification_model = tf.keras.models.load_model(SIGN_CLASSIFY_MODEL_PATH, custom_objects={"random_brightness": random_brightness})
        
        pred = classification_model.predict(img, verbose=0)
        class_idx = np.argmax(pred[0])
        class_confidence = float(pred[0][class_idx])
        
        if 0 <= class_idx < len(class_descriptions):
            classification = class_descriptions[class_idx]
        else:
            classification = f"Class {class_idx}"

        print("class_idx:", class_idx)
        print("class_descriptions:", class_descriptions)
        print("classification:", classification)
        print("pred[0]:", pred[0])
        
        top_indices = np.argsort(pred[0])[-3:][::-1]
        print(f"Top 3 predictions for sign:")
        for i in top_indices:
            print(f"  {class_descriptions[i]}: {pred[0][i]:.4f}")
        
        return {
            'class': classification,
            'confidence': class_confidence,
            'class_index': int(class_idx)
        }
        
    except Exception as e:
        print(f"Error in classify_sign_crop: {e}")
        import traceback
        traceback.print_exc()
        return {
            'class': 'Classification Error',
            'confidence': 0.0,
            'class_index': -1
        }

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
        classification_model = tf.keras.models.load_model(SIGN_CLASSIFY_MODEL_PATH, custom_objects={"random_brightness": random_brightness})
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
            
            if x2 > x1 and y2 > y1:
                sign_crop = frame[y1:y2, x1:x2]
                
                try:
                    classification_result = classify_sign_crop(sign_crop)
                    classification = classification_result['class']
                    class_confidence = classification_result['confidence']
                    class_idx = classification_result['class_index']
                    
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
    print(f"Sign model detected {len(sign_model_detections)} signs")

    vehicle_model_detections = detect_vehicles_pedestrians(frame, include_traffic_lights=False, include_traffic_signs=True)

    classes_found = set()
    for detection in vehicle_model_detections:
        classes_found.add(detection['class'])
    print(f"Vehicle model found classes: {classes_found}")

    vehicle_model_sign_detections = []
    for detection in vehicle_model_detections:
        if ('traffic sign' in detection['class'].lower() or 
            'traffic signs' in detection['class'].lower() or 
            'sign' in detection['class'].lower()):
            detection['source'] = 'vehicle_model'
            detection['class'] = 'unknown'
            vehicle_model_sign_detections.append(detection)
    
    print(f"Vehicle model detected {len(vehicle_model_sign_detections)} signs")

    final_detections = []

    for sign_det in sign_model_detections:
        if sign_det['detection_confidence'] > 0.4:
            sign_det['source'] = 'sign_model'
            sign_det['verified'] = False
            final_detections.append(sign_det)

    for veh_det in vehicle_model_sign_detections:
        if veh_det['confidence'] > 0.6:
            try:
                x1, y1, x2, y2 = veh_det['bbox']
                sign_crop = frame[y1:y2, x1:x2]
                
                classification_result = classify_sign_crop(sign_crop)
                classification = classification_result['class']
                class_confidence = classification_result['confidence']
                class_idx = classification_result['class_index']
                
                if class_confidence > 0.4:
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
    
    filtered_detections = []
    for i, det1 in enumerate(final_detections):
        x1_1, y1_1, x2_1, y2_1 = det1['bbox']
        center1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
        
        if 'skip' in det1 and det1['skip']:
            continue
        
        filtered_detections.append(det1)
        
        for j in range(i+1, len(final_detections)):
            det2 = final_detections[j]
            x1_2, y1_2, x2_2, y2_2 = det2['bbox']
            center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
            
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            if distance < 50:
                if det1['detection_confidence'] < det2['detection_confidence']:
                    filtered_detections.pop()
                    filtered_detections.append(det2)
                    det1['skip'] = True
                    break
                else:
                    det2['skip'] = True
    
    print(f"Combined sign detection returning {len(filtered_detections)} signs")
    return filtered_detections
