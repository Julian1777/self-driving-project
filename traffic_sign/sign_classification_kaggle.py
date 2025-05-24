import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score
import cv2 as cv
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable



IMG_SIZE = (30, 30)
NUM_CLASSES = 43
TRAIN_DIR = "/kaggle/input/gtsrb-dataset/dataset/dataset/Train"
MODEL_PATH = "traffic_sign_classification.h5"
TEST_CSV = "/kaggle/input/gtsrb-dataset/Test.csv"
TEST_ROOT = "/kaggle/input/gtsrb-dataset/dataset/dataset"
EPOCHS = 10
BATCH_SIZE = 64


def preprocess_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

@register_keras_serializable()
def random_brightness(x):
    return tf.image.random_brightness(x, max_delta=0.2)

image_paths = []
labels = []
for class_id in range(NUM_CLASSES):
    class_dir = os.path.join(TRAIN_DIR, str(class_id))
    for img_name in os.listdir(class_dir):
        image_paths.append(os.path.join(class_dir, img_name))
        labels.append(class_id)

image_paths = np.array(image_paths)
labels = np.array(labels)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

train_labels = to_categorical(train_labels, NUM_CLASSES)
val_labels = to_categorical(val_labels, NUM_CLASSES)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(buffer_size=10000)
train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
    layers.Lambda(random_brightness)
])

model = Sequential([
    layers.Input(shape=(30, 30, 3)),
    data_augmentation,
    Conv2D(32, (5,5), activation='relu'),
    Conv2D(64, (5,5), activation='relu'),
    MaxPool2D((2,2)),
    Dropout(0.15),
    Conv2D(128, (3,3), activation='relu'),
    Conv2D(256, (3,3), activation='relu'),
    MaxPool2D((2,2)),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
   train_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_ds
)

model.save("traffic_sign_classification.h5")

test_df = pd.read_csv(TEST_CSV)
labels = test_df["ClassId"].values
imgs = test_df["Path"].values

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"random_brightness": random_brightness})

print("First 5 test image paths from CSV:", imgs[:5])
data = []
for img in imgs:
    img_path = os.path.join(TEST_ROOT, img)
    try:
        img_pil = Image.open(img_path)
        img_pil = img_pil.resize(IMG_SIZE)
        img_np = np.array(img_pil)
        img_np = img_np.astype(np.float32) / 255.0
        data.append(img_np)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
    if len(data) == 1:
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.title(f"Label: {labels[0]}")
        plt.show()

X_test = np.array(data)

pred = np.argmax(model.predict(X_test, verbose=1), axis=-1)

print("Test Data accuracy: ", accuracy_score(labels, pred) * 100)