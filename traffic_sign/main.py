import tensorflow as tf
import tensorflow
import matplotlib.pyplot as plt
import hashlib
import json
import os
from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import pandas as pd
from tensorflow.keras import layers


BATCH_SIZE = 64
IMG_SIZE = (224,224)
SEED = 123
EPOCHS = 20


print("GPU Available:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])  # 8GB * 0.9
    except RuntimeError as e:
        print(e)

def preprocess_image(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path, confidence_threshold=0.7):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]
    
    if confidence < confidence_threshold:
        return "Unknown sign (low confidence)"
    return ordered_descriptions[predicted_index]


def load_class_names(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print("CSV columns found:", df.columns.tolist())
        
        unique_classes = df['ClassId'].unique()
        
        class_names = {}
        for class_id in unique_classes:
            class_names[str(class_id)] = f"Traffic Sign Class {class_id}"
            
        return class_names
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return {}

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomRotation(0.2),
])

ds_path = os.path.join("dataset", "Train")


print("Path to dataset:", ds_path)

train_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    ds_path,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=SEED
)

val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    ds_path,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=SEED
)

class_names = load_class_names(os.path.join("dataset", "Train.csv"))
print(class_names)

original_class_names = train_val_ds.class_names
num_classes = len(original_class_names)
print(num_classes)

# Now apply cache and prefetch
train_val_ds = train_val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

class_indices = {i: name for i, name in enumerate(original_class_names)}
ordered_descriptions = []
for i in range(num_classes):
    class_name = class_indices[i]
    description = class_names.get(class_name, f"Class {class_name}")
    ordered_descriptions.append(description)

class_counts = np.bincount([y.numpy() for x, y in train_val_ds.unbatch()])
class_weights = {i: 1/(count/len(class_counts)) for i, count in enumerate(class_counts)}

val_batches = int(0.5 * len(val_test_ds))
val_ds = val_test_ds.take(val_batches)
test_ds = val_test_ds.skip(val_batches)



print(f"Train dataset size: {len(train_val_ds)} batches")
print(f"Validation dataset size: {len(val_ds)} batches")
print(f"Test dataset size: {len(test_ds)} batches")

normalization_layer = tf.keras.layers.Rescaling(1./255)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    data_augmentation,

    normalization_layer,        

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', dtype='float32'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
])

print(model.summary())

if not os.path.exists("sign.h5"):

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_sign_model.h5",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history = model.fit(
        train_val_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        class_weight=class_weights,
        callbacks=[checkpoint]
    )

    model.save("sign.h5")

    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

else:
    model = tf.keras.models.load_model("sign.h5")
    print("Model loaded successfully.")

plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):  
    predictions = model.predict(images)
    predicted_classes = tf.argmax(predictions, axis=1)
    for i in range(16):
        ax = plt.subplot(4, 4, i+1)
        plt.imshow((images[i].numpy()).astype("uint8")) 
        plt.title(f"True: {ordered_descriptions[labels[i]]}\nPred: {ordered_descriptions[predicted_classes[i]]}")
        plt.axis("off")
plt.tight_layout()
plt.show()