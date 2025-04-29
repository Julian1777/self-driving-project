import cv2 as cv
import numpy as np
import os

IMG_SIZE = (224, 224)
SIGN_MODEL_PATH = os.path.join("model", "sign_model.h5")


def img_preprocessing(frame):
    img = frame.copy()
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_sign(frame):
    """Takes an input image and predicts lane segmentation"""
    input_tensor = img_preprocessing(frame)
    prediction = SIGN_MODEL_PATH.predict(np.expand_dims(input_tensor, axis=0))
    return prediction

print("Prediction: ", prediction)