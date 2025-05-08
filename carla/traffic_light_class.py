import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2 as cv
from traffic_light_class_cv import analyze_brightness_pattern



IMG_SIZE = (224, 224)
LIGHT_MODEL_PATH = os.path.join("model", "traffic_light_classification.h5")
STATES = ["red", "yellow", "green"]

def img_preprocessing(image_input):
    if isinstance(image_input, str):
        image = cv.imread(image_input)
        if image is None:
            raise ValueError(f"Could not read image from {image_input}")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    else:
        image = image_input.copy()
        if image.shape[2] == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    image = cv.resize(image, IMG_SIZE)
    
    image = image / 255.0
    
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_traffic_light(frame):
    model = load_model(LIGHT_MODEL_PATH)
    input_tensor = img_preprocessing(frame)
    prediction = model.predict(input_tensor, verbose=0)[0]
    
    state_id = np.argmax(prediction)
    confidence = float(prediction[state_id])

    if confidence < 0.6:
        print("CNN confidence too low, falling back to OpenCV classifier")
        if isinstance(frame, str):
            img = cv.imread(frame)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:
            img = frame.copy()

        img = cv.resize(img, (64, 64))

        cv_result = analyze_brightness_pattern(img)
        cv_state = cv_result["predicted_state"]
        cv_confidence = cv_result["confidence"]

        return [{'state': cv_state, 'confidence': cv_confidence, 'method': 'opencv'}]
    
    state_name = STATES[state_id] if state_id < len(STATES) else f"Unknown_{state_id}"
    
    return [{'state': state_name, 'confidence': confidence}]
    