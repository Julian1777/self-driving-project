from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
LANE_MODEL_PATH = 'lane_detection_model.h5'
TEST_IMAGE_PATH = 'lane.jpeg'

def test_model():
    image = cv.imread(TEST_IMAGE_PATH)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    prediction = predict_lane(image)


    if len(prediction.shape) == 4:
        prediction_display = prediction[0, :, :, 0]
    else:
        prediction_display = prediction

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Prediction")

    plt.imshow(prediction_display, cmap='gray')

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return prediction


def img_preprocessing(image):
    img = image.copy()
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_lane(frame):
    model = load_model(LANE_MODEL_PATH)
    input_tensor = img_preprocessing(frame)
    prediction = model.predict(input_tensor)
    return prediction


test_model()
