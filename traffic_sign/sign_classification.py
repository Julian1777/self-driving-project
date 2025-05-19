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
EPOCHS = 50


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

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.GaussianNoise(0.05) 
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

class_names = load_class_names(os.path.join("sign_dic.csv"))
print(class_names)

original_class_names = train_val_ds.class_names
num_classes = len(original_class_names)
print(num_classes)

print("Dataset class names (directories):", train_val_ds.class_names)

dir_to_index = {name: i for i, name in enumerate(train_val_ds.class_names)}

ordered_descriptions = []
for dir_name in train_val_ds.class_names:
    description = class_names.get(dir_name, f"Class {dir_name}")
    ordered_descriptions.append(description)
    print(f"Directory: {dir_name} → Description: {description}")

train_val_ds = train_val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

class_counts = np.bincount([y.numpy() for x, y in train_val_ds.unbatch()])
class_weights = {i: 1/(count/len(class_counts)) for i, count in enumerate(class_counts)}

val_batches = int(0.5 * len(val_test_ds))
val_ds = val_test_ds.take(val_batches)
test_ds = val_test_ds.skip(val_batches)



print(f"Train dataset size: {len(train_val_ds)} batches")
print(f"Validation dataset size: {len(val_ds)} batches")
print(f"Test dataset size: {len(test_ds)} batches")

normalization_layer = tf.keras.layers.Rescaling(1./255)

base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights='imagenet'
    )

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    data_augmentation,

    normalization_layer,

    base_model,        

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(256, activation='relu', dtype='float32'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
])

print(model.summary())

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_sign_model.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

if not os.path.exists("sign_model.h5"):
    base_model.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    history_phase1 = model.fit(
        train_val_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=1,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping]
    )
    
    print("Phase 1 complete. Beginning fine-tuning phase...")
    base_model.trainable = True
    
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 10x smaller
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001
    )
    
    history_phase2 = model.fit(
        train_val_ds,
        validation_data=val_ds,
        epochs=15,
        verbose=1,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    combined_history = {}
    for key in history_phase1.history:
        combined_history[key] = history_phase1.history[key] + history_phase2.history[key]
    
    model.save("sign_model.h5")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(combined_history['loss'], label='Train Loss')
    plt.plot(combined_history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(combined_history['accuracy'], label='Train Accuracy')
    plt.plot(combined_history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

else:
    model = tf.keras.models.load_model("sign_model.h5")
    print("Model loaded successfully.")

if not os.path.exists("sign_inference.h5"):
    print("Model layers:")
    for i, layer in enumerate(model.layers):
        print(f"  {i}: {layer.name} ({type(layer).__name__})")

    inference_model = tf.keras.Sequential()
    inference_model.add(tf.keras.layers.Input(shape=(224, 224, 3)))

    inference_model.add(tf.keras.layers.Rescaling(1./255, name="inference_rescaling"))

    base_model = model.get_layer("mobilenetv2_1.00_224")
    inference_model.add(base_model)

    inference_model.add(model.get_layer("global_average_pooling2d"))
    inference_model.add(model.get_layer("dense"))
    inference_model.add(model.get_layer("batch_normalization"))
    inference_model.add(model.get_layer("dense_1"))

    inference_model.save("sign_inference.h5")
    print("Saved inference-only model to sign_inference.h5")

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