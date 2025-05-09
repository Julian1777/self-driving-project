import cv2 as cv
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
SIGN_MODEL_PATH = os.path.join("model", "sign_inference.h5")


def load_class_names(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Loading class names from {csv_path}")
        print("CSV columns found:", df.columns.tolist())
        
        class_names = {}
        id_column = 'id'
        desc_column = 'description'
        
        for _, row in df.iterrows():
            class_id = row[id_column]
            name = row[desc_column]
            class_names[str(class_id)] = name
            
        print(f"Loaded {len(class_names)} class names")
        return class_names
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return {}

class_names_dict = load_class_names("sign_dic.csv")
print(class_names_dict)

num_classes = max(map(int, class_names_dict.keys())) + 1
class_descriptions = ["Unknown Class"] * num_classes
for class_id, description in class_names_dict.items():
    class_id = int(class_id)
    if 0 <= class_id < num_classes:
        class_descriptions[class_id] = description

def img_preprocessing(frame):
    img = frame.copy()
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_sign(frame):
    model = load_model(SIGN_MODEL_PATH)
    input_tensor = img_preprocessing(frame)
    
    prediction = model.predict(input_tensor, verbose=0)[0]  # Get first item in batch
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction values: {prediction}")
    
    class_id = np.argmax(prediction)
    print(f"Predicted class ID: {class_id}")
    
    confidence = float(prediction[class_id])
    
    if confidence < 0.5:
        return []
    
    class_name = class_descriptions[class_id] if class_id < len(class_descriptions) else f"Unknown_{class_id}"
    
    h, w = frame.shape[:2]
    x, y, width, height = w - 150, 50, 100, 100
    
    return [{'bbox': (x, y, width, height), 'label': class_name, 'confidence': confidence}]