import cv2 as cv
import numpy as np

IMG_SIZE = (224, 224)
LANE_MODEL_PATH = 'lane_model.h5'

def img_preprocessing(image_path):
    img = cv.imread(image_path)
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_lane(image):
    """Takes an input image and predicts lane segmentation"""
    input_tensor = img_preprocessing(image)
    prediction = LANE_MODEL_PATH.predict(np.expand_dims(input_tensor, axis=0))
    return prediction