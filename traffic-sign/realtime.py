import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import os

model = load_model("traffic_sign_model.h5")
csv_path = os.path.jon("dataset", "merged_data", "global_labels.csv")


def load_ordered_descriptions(csv_path):
    df = pd.read_csv(csv_path)
    df['id'] = df['id'].astype(int)
    df = df.sort_values('id')
    return df['description'].tolist()

ordered_descriptions = load_ordered_descriptions(csv_path)

def detect_sign(frame, model, confidence_threshold=0.7):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed = cv2.resize(rgb, (224, 224)) / 255.0
    processed = np.expand_dims(processed, axis=0)

    predictions = model.predict(processed)
    confidence = np.max(predictions)
    class_id = np.argmax(predictions)

    if confidence > confidence_threshold:
        return ordered_descriptions[class_id], confidence
    return None, None

test_img = cv2.imread("test_image_30kmh.jpg")
label, conf = detect_sign(test_img, model)
print(f"Test prediction: {label} ({conf:.2f})")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        

    label, conf = detect_sign(frame, model)
    
    if label:
        cv2.putText(frame, f"{label} ({conf:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Traffic Sign Detector', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()