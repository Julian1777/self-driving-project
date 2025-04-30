from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

IMG_SIZE = (224, 224)
CLASSIFICATION_MODEL_PATH = 'traffic_light.h5'
DETECTION_MODEL_PATH = 'traffic_light_detection.h5'
CONFIDENCE_THRESHOLD = 0.5

STATES = {
    0: {'name': 'go', 'color': (0, 255, 0)},        # Green
    1: {'name': 'goLeft', 'color': (0, 255, 128)},  # Green-Yellow
    2: {'name': 'stop', 'color': (0, 0, 255)},      # Red
    3: {'name': 'stopLeft', 'color': (128, 0, 255)},# Purple-Red
    4: {'name': 'warning', 'color': (0, 255, 255)}  # Yellow
}

def img_preprocessing(image):
    img = image.copy()
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_traffic_lights(image, detection_model, confidence_threshold=0.5):
    """Detect traffic lights in an image using the detection model"""
    h, w = image.shape[:2]
    preprocessed = img_preprocessing(image)
    detections = detection_model.predict(preprocessed, verbose=0)[0]
    
    if detections[4] < confidence_threshold:
        return []
    
    x_center, y_center, width, height = detections[:4]
    confidence = detections[4]
    
    x1 = int((x_center - width/2) * w)
    y1 = int((y_center - height/2) * h) 
    x2 = int((x_center + width/2) * w)
    y2 = int((y_center + height/2) * h)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 > x1 and y2 > y1:
        return [{
            'bbox': (x1, y1, x2, y2),
            'confidence': confidence
        }]
    
    return []

def classify_traffic_light(image, bbox, classification_model):
    """Classify a detected traffic light"""
    x1, y1, x2, y2 = bbox
    
    tl_crop = image[y1:y2, x1:x2]
    
    if tl_crop.size == 0 or tl_crop.shape[0] < 4 or tl_crop.shape[1] < 4:
        return None, 0
    
    tl_input = img_preprocessing(tl_crop)
    classification_result = classification_model.predict(tl_input, verbose=0)[0]
    class_id = np.argmax(classification_result)
    confidence = classification_result[class_id]
    
    return class_id, confidence

def process_image(image_path):
    """Process a single image for traffic light detection and classification"""
    detection_model = load_model(DETECTION_MODEL_PATH, compile=False)
    classification_model = load_model(CLASSIFICATION_MODEL_PATH, compile=False)
    
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    result_image = image_rgb.copy()
    
    detections = detect_traffic_lights(image, detection_model, CONFIDENCE_THRESHOLD)
    
    if not detections:
        print("No traffic lights detected with high confidence, processing entire image")
        detection_input = img_preprocessing(image)
        classification_result = classification_model.predict(detection_input, verbose=0)[0]
        class_id = np.argmax(classification_result)
        confidence = classification_result[class_id]
        
        if class_id in STATES:
            state = STATES[class_id]
            h, w = image.shape[:2]
            cv.rectangle(result_image, (10, 10), (w-10, h-10), state['color'], 3)
            label = f"{state['name']} ({confidence:.2f})"
            cv.putText(result_image, label, (15, 35), cv.FONT_HERSHEY_SIMPLEX, 
                      1.0, state['color'], 2)
    else:
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            detection_confidence = detection['confidence']
            
            class_id, class_confidence = classify_traffic_light(image, detection['bbox'], classification_model)
            
            if class_id in STATES:
                state = STATES[class_id]
                cv.rectangle(result_image, (x1, y1), (x2, y2), state['color'], 2)
                label = f"{state['name']} ({class_confidence:.2f})"
                cv.putText(result_image, label, (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX,
                          0.7, state['color'], 2)
                
                det_label = f"Detection: {detection_confidence:.2f}"
                cv.putText(result_image, det_label, (x1, y2+20), cv.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 255, 255), 1)
    
    return image_rgb, result_image

def test_model(image_path=None):
    if image_path is None or not os.path.exists(image_path):
        test_images = [
            'traffic_light1_stop.jpg',
            'traffic_light2_go.jpg'
        ]
    elif os.path.isdir(image_path):
        test_images = []
        for ext in ['jpg', 'jpeg', 'png']:
            test_images.extend(glob.glob(os.path.join(image_path, f'*.{ext}')))
    else:
        test_images = [image_path]
    
    for img_path in test_images:
        print(f"Processing {img_path}...")
        results = process_image(img_path)
        
        if results is None:
            continue
            
        original, detection = results
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(original)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Traffic Light Detection & Classification")
        plt.imshow(detection)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_model()