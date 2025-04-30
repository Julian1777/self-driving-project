import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

IMG_SIZE = (224, 224)
LANE_MODEL_PATH = os.path.join("model",'lane_detection_model.h5')

def iou_metric(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    iou = intersection / (union + tf.keras.backend.epsilon())
    
    return iou

def img_preprocessing(frame):
    img = frame.copy()
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_lane(frame):
    model = load_model(LANE_MODEL_PATH, custom_objects={'iou_metric': iou_metric})
    input_tensor = img_preprocessing(frame)
    mask_pred = model.predict(input_tensor, verbose=0)[0]
    
    binary_mask = (mask_pred > 0.5).astype(np.uint8)
    
    h, w = frame.shape[:2]

    debug_mask = cv.resize(binary_mask * 255, (w, h))
        
    left_line = ((int(w * 0.25), int(h)), (int(w * 0.45), int(h * 0.6)))
    right_line = ((int(w * 0.75), int(h)), (int(w * 0.55), int(h * 0.6)))
        
    return [left_line, right_line]