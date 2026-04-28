import cv2 as cv
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import sys
from tensorflow.keras.models import load_model
from config.config import SIGN_DETECTION_MODEL, SIGN_CLASSIFICATION_MODEL

IMG_SIZE = (48, 48)
SIGN_MODEL_PATH = str(SIGN_DETECTION_MODEL)
SIGN_CLASSIFY_MODEL_PATH = str(SIGN_CLASSIFICATION_MODEL)

# GTSRB class names
SIGN_CLASSES = { 
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow',
    31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons' 
}

def get_models_dict():
    try:
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'MODELS'):
            return main_module.MODELS
        return None
    except:
        return None

def preprocess_img(img):
    """
    Training code does not normalize to [0,1]!
    Model was trained on 0-255 uint8 images.
    
    Args:
        img: numpy array image (RGB, 0-255)
    Returns:
        Preprocessed image (48x48, 0-255 uint8)
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty")
        
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    hsv[:,:,2] = cv.equalizeHist(hsv[:,:,2])
    img = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    img = cv.resize(img, IMG_SIZE)
    return img.astype(np.uint8)

class_descriptions = ["Unknown Class"] * 43
for class_id, description in SIGN_CLASSES.items():
    if 0 <= class_id < 43:
        class_descriptions[class_id] = description

def classify_sign_crop(sign_crop):
    try:
        img = preprocess_img(sign_crop)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        models_dict = get_models_dict()
        if models_dict is not None and 'sign_classify' in models_dict:
            classification_model = models_dict['sign_classify']
        else:
            classification_model = load_model(SIGN_CLASSIFY_MODEL_PATH)

        pred = classification_model.predict(img, verbose=0)
        class_idx = np.argmax(pred[0])
        class_confidence = float(pred[0][class_idx])

        if 0 <= class_idx < len(class_descriptions):
            classification = class_descriptions[class_idx]
        else:
            classification = f"Class {class_idx}"

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

def detect_classify_sign(frame, detection_model=None, classification_model=None):
    if detection_model is None or classification_model is None:
        models_dict = get_models_dict()
    
    if detection_model is None:
        if models_dict is not None and 'sign_detect' in models_dict:
            detection_model = models_dict['sign_detect']
        else:
            try:
                detection_model = YOLO(SIGN_MODEL_PATH)
                print(f"Warning: Loading sign detection model from scratch - slower!")
            except Exception as e:
                print(f"Error loading sign detection model: {e}")
                return []
    
    if classification_model is None:
        if models_dict is not None and 'sign_classify' in models_dict:
            classification_model = models_dict['sign_classify']
        else:
            try:
                classification_model = tf.keras.models.load_model(SIGN_CLASSIFY_MODEL_PATH)
                print(f"Warning: Loading sign classification model from scratch - slower!")
            except Exception as e:
                print(f"Error loading sign classification model: {e}")
                return []

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

def sign_detection_only(frame, confidence_threshold=0.2):
    """
    Detect traffic signs only, without classification.
    Returns bounding boxes and detection confidence.
    
    Args:
        frame: Input image
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        List of detections with bbox and detection_confidence
    """
    models_dict = get_models_dict()
    
    if models_dict is not None and 'sign_detect' in models_dict:
        detection_model = models_dict['sign_detect']
    else:
        detection_model = YOLO(SIGN_MODEL_PATH)
    
    results = detection_model(frame, conf=confidence_threshold)
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
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'detection_class': class_name,
                    'detection_confidence': confidence
                })
    
    return detections

def sign_classification_only(frame, bboxes=None):
    """
    Classify traffic signs in given bounding boxes or full frame.
    If bboxes not provided, performs detection then classification.
    
    Args:
        frame: Input image
        bboxes: Optional list of (x1, y1, x2, y2) to classify
        
    Returns:
        List of classifications with bbox and classification_confidence
    """
    models_dict = get_models_dict()
    
    if models_dict is not None and 'sign_classify' in models_dict:
        classification_model = models_dict['sign_classify']
    else:
        classification_model = load_model(SIGN_CLASSIFY_MODEL_PATH)
    
    # If no bboxes provided, detect first then classify
    if bboxes is None:
        detected = sign_detection_only(frame, confidence_threshold=0.2)
        bboxes = [det['bbox'] for det in detected]
    
    classifications = []
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            sign_crop = frame[y1:y2, x1:x2]
            
            try:
                classification_result = classify_sign_crop(sign_crop)
                classifications.append({
                    'bbox': bbox,
                    'classification': classification_result['class'],
                    'classification_confidence': classification_result['confidence'],
                    'class_index': classification_result['class_index']
                })
            except Exception as e:
                print(f"Error during classification: {e}")
                classifications.append({
                    'bbox': bbox,
                    'classification': 'Classification failed',
                    'classification_confidence': 0.0,
                    'class_index': -1
                })
    
    return classifications

def sign_detection_classification(frame, detection_model=None, classification_model=None):
    """
    Pure sign detection and classification.
    """
    return detect_classify_sign(frame, detection_model=detection_model, classification_model=classification_model)
