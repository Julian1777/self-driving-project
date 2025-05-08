import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import pandas as pd

IMG_SIZE = (224, 224)
test_dir = "test_images"
model_path = "sign_inference.h5"
csv_path = "labels.csv"

def preprocess_image(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, IMG_SIZE)
    
    display_img = img.copy().astype(np.uint8)
    
    img = img.astype(np.float32)  # Remove the /255.0
    img = np.expand_dims(img, axis=0)
    
    return img, display_img

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

print(f"Loading model from {model_path}")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully")

class_names_dict = load_class_names("sign_dic.csv")
print(class_names_dict)

num_classes = max(map(int, class_names_dict.keys())) + 1
class_descriptions = ["Unknown Class"] * num_classes
for class_id, description in class_names_dict.items():
    class_id = int(class_id)
    if 0 <= class_id < num_classes:
        class_descriptions[class_id] = description

images_path = os.path.join(os.path.dirname(__file__), test_dir)
image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

plt.figure(figsize=(15, len(image_files) * 3))

for i, img_file in enumerate(image_files):
    img_path = os.path.join(images_path, img_file)
    img_batch, display_img = preprocess_image(img_path)
    
    prediction = model.predict(img_batch, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]


    top_indices = np.argsort(prediction)[-3:][::-1]
    top_classes = [class_descriptions[idx] if idx < len(class_descriptions) else f"Unknown ({idx})" 
                   for idx in top_indices]
    top_confidences = [prediction[idx] for idx in top_indices]
    
    print(f"\nImage: {img_file}")
    print(f"Top predictions:")
    for j, (cls, conf) in enumerate(zip(top_classes, top_confidences)):
        print(f"  {j+1}. {cls}: {conf:.4f}")
    
    plt.subplot(len(image_files), 1, i+1)
    plt.imshow(display_img)
    title = f"{img_file}\nPredicted: {top_classes[0]} ({top_confidences[0]:.2f})"
    title += f"\n2nd: {top_classes[1]} ({top_confidences[1]:.2f})"
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()