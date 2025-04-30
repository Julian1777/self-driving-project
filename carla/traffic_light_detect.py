import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2 as cv


IMG_SIZE = (224, 224)
LIGHT_MODEL_PATH = os.path.join("model", "traffic_light_detection.h5")

def yolo_loss(y_true, y_pred):
    coord_loss = tf.reduce_mean(tf.square(y_true[:, :4] - y_pred[:, :4]))
    conf_loss = tf.reduce_mean(tf.square(y_true[:, 4] - y_pred[:, 4]))
    return coord_loss + conf_loss

def img_preprocessing(image_input):
    if isinstance(image_input, str):
        # If it's a file path
        image = cv.imread(image_input)
        if image is None:
            raise ValueError(f"Could not read image from {image_input}")
    else:
        image = image_input.copy()
    
    if len(image.shape) == 2:  # Grayscale
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
    elif image.shape[2] == 3 and image_input is not str:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, IMG_SIZE)
    image = image / 255.0  # Normalize to [0, 1]
    
    image = np.expand_dims(image, axis=0)
    return image

def detect_traffic_light(frame):
    model = load_model(LIGHT_MODEL_PATH, custom_objects={'yolo_loss': yolo_loss})
    input_tensor = img_preprocessing(frame)
    prediction = model.predict(input_tensor, verbose=0)[0]
    
    x_center, y_center, width, height, confidence = prediction
    
    if confidence < 0.4:
        return []
    
    h, w = frame.shape[:2]
    x1 = int((x_center - width/2) * w)
    y1 = int((y_center - height/2) * h)
    x2 = int((x_center + width/2) * w)
    y2 = int((y_center + height/2) * h)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    return [{'bbox': (x1, y1, x2, y2), 'confidence': float(confidence)}]