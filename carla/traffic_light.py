import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model


IMG_SIZE = (224, 224)
LIGHT_MODEL_PATH = os.path.join("model", "traffic_light.h5")

def img_preprocessing(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def predict_traffic_light(frame):
    model = load_model(LIGHT_MODEL_PATH)
    input_tensor = img_preprocessing(frame)
    prediction = model.predict(input_tensor)
    return prediction