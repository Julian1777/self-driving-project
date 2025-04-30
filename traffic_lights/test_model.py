from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
LANE_MODEL_PATH = 'traffic_light.h5'
TEST_IMAGE_PATH = 'traffic_light1_stop.jpg'
TEST_IMAGE_PATH2 = 'traffic_light2_go.jpg'

STATES = {
    0: {'name': 'Stop', 'color': (0, 0, 255)},      # Red
    1: {'name': 'Warning', 'color': (0, 255, 255)}, # Yellow
    2: {'name': 'Go', 'color': (0, 255, 0)}         # Green
}

def test_model():
    image = cv.imread(TEST_IMAGE_PATH2)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    prediction = predict_traffic_light(image)

    result_image = image_rgb.copy()


    if len(prediction.shape) == 2 and prediction.shape[1] <= 5:
        class_id = np.argmax(prediction[0])
        confidence = prediction[0][class_id]
        
        h, w = image_rgb.shape[:2]
        state = STATES[class_id]
        
        cv.rectangle(result_image, (10, 10), (w-10, h-10), state['color'], 3)
        
        label = f"{state['name']} ({confidence:.2f})"
        cv.putText(result_image, label, (15, 35), cv.FONT_HERSHEY_SIMPLEX, 
                   1.0, state['color'], 2)
                   
    elif len(prediction.shape) > 2:
        
        for detection in prediction[0]:
            if detection[4] < 0.5:
                continue
                
            x, y, w, h = detection[0:4]
            x1, y1 = int(x - w/2), int(y - h/2)  # Convert center to top-left
            x2, y2 = int(x + w/2), int(y + h/2)  # Convert to bottom-right
            
            class_id = np.argmax(detection[5:])
            confidence = detection[5 + class_id]
            state = STATES[class_id]
            
            cv.rectangle(result_image, (x1, y1), (x2, y2), state['color'], 2)
            
            label = f"{state['name']} ({confidence:.2f})"
            cv.putText(result_image, label, (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX,
                      0.5, state['color'], 2)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Prediction")

    plt.imshow(result_image)

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

def predict_traffic_light(frame):
    model = load_model(LANE_MODEL_PATH, compile=False)
    input_tensor = img_preprocessing(frame)
    prediction = model.predict(input_tensor)
    return prediction


test_model()
