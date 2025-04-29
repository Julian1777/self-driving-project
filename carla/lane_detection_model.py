import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
LANE_MODEL_PATH = 'lane_model.h5'

def img_preprocessing(frame):
    img = frame.copy()
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_lane(frame):
    model = load_model(LANE_MODEL_PATH)
    input_tensor = img_preprocessing(frame)
    prediction = model.predict(input_tensor)
    return prediction