import cv2 as cv
import numpy as np
import os
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
SIGN_MODEL_PATH = os.path.join("model", "sign.h5")

#ADD OTHERS LATER
SIGN_CLASSES = [
    'speed_limit', 'stop', 'yield', 'no_entry', 'no_parking',
    'pedestrian_crossing', 'traffic_light_ahead', 'railroad_crossing'
]

def img_preprocessing(frame):
    img = frame.copy()
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_sign(frame):
    model = load_model(SIGN_MODEL_PATH)
    input_tensor = img_preprocessing(frame)
    
    prediction = model.predict(input_tensor, verbose=0)[0]  # Get first item in batch
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction values: {prediction}")
    
    class_id = np.argmax(prediction)
    print(f"Predicted class ID: {class_id}")
    
    confidence = float(prediction[class_id])
    
    if confidence < 0.5:
        return []
    
    class_name = SIGN_CLASSES[class_id] if class_id < len(SIGN_CLASSES) else f"Unknown_{class_id}"
    
    h, w = frame.shape[:2]
    x, y, width, height = w - 150, 50, 100, 100
    
    return [{'bbox': (x, y, width, height), 'label': class_name, 'confidence': confidence}]