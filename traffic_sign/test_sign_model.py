import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score


IMG_SIZE = (30, 30)
MODEL_PATH = "traffic_sign_classification.h5"
TEST_CSV = "dataset/Test.csv"
TEST_ROOT = "dataset"

# Load test CSV
test_df = pd.read_csv(TEST_CSV)
imgs = test_df["Path"].values
labels = test_df["ClassId"].values

# Pick 16 random indices
indices = random.sample(range(len(imgs)), 16)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

plt.figure(figsize=(16, 8))
for i, idx in enumerate(indices):
    img_rel_path = imgs[idx]
    img_path = os.path.join(TEST_ROOT, img_rel_path)
    label = labels[idx]

    # Load and preprocess image
    img_pil = Image.open(img_path)
    img_pil = img_pil.resize(IMG_SIZE)
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_np, axis=0)

    # Predict
    pred = np.argmax(model.predict(img_input, verbose=0), axis=-1)[0]
    print(f"Image: {img_rel_path}, True: {label}, Pred: {pred}")

    # Plot
    plt.subplot(4, 4, i+1)
    plt.imshow(img_np)
    plt.title(f"True: {label}\nPred: {pred}")
    plt.axis('off')

plt.tight_layout()
plt.show()

X_test = []
for img in imgs:
    img_path = os.path.join(TEST_ROOT, img)
    img_pil = Image.open(img_path)
    img_pil = img_pil.resize(IMG_SIZE)
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    X_test.append(img_np)
X_test = np.array(X_test)
preds = np.argmax(model.predict(X_test, verbose=1), axis=-1)
print("Test Data accuracy: ", accuracy_score(labels, preds) * 100)