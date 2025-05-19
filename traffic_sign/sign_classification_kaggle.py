import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras import layers
import gc
import collections


BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 123
EPOCHS = 50

KAGGLE_PATH = "/kaggle/input/gtsrb-dataset/dataset/dataset"
TRAIN_PATH = os.path.join(KAGGLE_PATH, "Train")
TEST_PATH = os.path.join(KAGGLE_PATH, "Test")
SAVE_PATH = "/kaggle/working"

os.makedirs(SAVE_PATH, exist_ok=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) # Notice here
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()
    print("Memory cleared")
        
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path, model, ordered_descriptions, confidence_threshold=0.7):
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
        
        print("First 5 rows of CSV:")
        print(df.head())
        
        class_names = {}
        
        id_column = None
        desc_column = None
        
        for col in ['id']:
            if col in df.columns:
                id_column = col
                break
                
        for col in ['description']:
            if col in df.columns:
                desc_column = col
                break
        
        if id_column is None or desc_column is None:
            print(f"Could not identify id/description columns. Using first two columns.")
            id_column = df.columns[0]
            desc_column = df.columns[1]
            
        print(f"Using columns: ID='{id_column}', Description='{desc_column}'")
        
        for _, row in df.iterrows():
            class_id = row[id_column]
            name = row[desc_column]
            class_names[str(class_id)] = name
            class_names[int(class_id)] = name
            
        print(f"Loaded {len(class_names)} class names")
        
        print("Sample mappings:")
        sample_keys = list(class_names.keys())[:5]
        for key in sample_keys:
            print(f"  {key} → {class_names[key]}")
            
        return class_names
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        import traceback
        traceback.print_exc()
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

print("Path to dataset:", TRAIN_PATH)

train_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.1,
    subset="training",
    seed=SEED
)

val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.1,
    subset="validation",
    seed=SEED
)

try:
    class_names = load_class_names("/kaggle/input/gtsrb-dataset/sign_dic.csv")
except:
    class_names = load_class_names("sign_dic.csv")
    if not class_names:
        print("Warning: Could not load class descriptions. Using directory names instead.")
        class_names = {name: name for name in train_val_ds.class_names}

original_class_names = train_val_ds.class_names
num_classes = len(original_class_names)
print(f"Number of classes: {num_classes}")

print("Dataset class names (directories):", train_val_ds.class_names)

dir_to_index = {name: i for i, name in enumerate(train_val_ds.class_names)}

ordered_descriptions = []
for dir_name in train_val_ds.class_names:
    description = None
    
    if dir_name in class_names:
        description = class_names[dir_name]
    elif dir_name.isdigit() and int(dir_name) in class_names:
        description = class_names[int(dir_name)]
    elif dir_name.isdigit():
        for padding in range(1, 5):
            padded = dir_name.zfill(padding)
            if padded in class_names:
                description = class_names[padded]
                break
    
    if description is None:
        description = f"Class {dir_name}"
        
    ordered_descriptions.append(description)
    print(f"Directory: {dir_name} → Description: {description}")

train_val_ds = train_val_ds.prefetch(tf.data.AUTOTUNE)
val_test_ds = val_test_ds.prefetch(tf.data.AUTOTUNE)

clear_memory()

label_counts = collections.Counter()
for _, labels in train_val_ds:
    flat_labels = labels.numpy().flatten()
    for label in flat_labels:
        label_counts[int(label)] += 1

total = sum(label_counts.values())
class_weights = {i: total / (len(label_counts) * count) for i, count in label_counts.items()}

val_batches = int(0.5 * len(val_test_ds))
val_ds = val_test_ds.take(val_batches)
test_ds = val_test_ds.skip(val_batches)

print(f"Train dataset size: {len(train_val_ds)} batches")
print(f"Validation dataset size: {len(val_ds)} batches")
print(f"Test dataset size: {len(test_ds)} batches")

normalization_layer = tf.keras.layers.Rescaling(1./255)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    normalization_layer,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu', dtype='float32'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
])

print(model.summary())

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(SAVE_PATH, "best_sign_model.h5"),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=7,
    restore_best_weights=True
)

# Phase 1: Train only the top layers
base_model.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Phase 1: Training top layers...")
history_phase1 = model.fit(
    train_val_ds,
    validation_data=val_ds,
    epochs=15,
    verbose=1,
    callbacks=[checkpoint, early_stopping]
)

print("Phase 2: Fine-tuning the model...")
base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=0.00001
)

history_phase2 = model.fit(
    train_val_ds,
    validation_data=val_ds,
    epochs=35,
    verbose=1,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

combined_history = {}
for key in history_phase1.history:
    combined_history[key] = history_phase1.history[key] + history_phase2.history[key]

final_model_path = os.path.join(SAVE_PATH, "sign_model.h5")
model.save(final_model_path)
print(f"Model saved to {final_model_path}")

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
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "training_history.png"))
plt.show()

print("Creating inference model...")
inference_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./255, name="inference_rescaling"),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    model.get_layer("dense"),
    model.get_layer("batch_normalization"),
    model.get_layer("dense_1")
])

inference_model_path = os.path.join(SAVE_PATH, "sign_inference.h5") 
inference_model.save(inference_model_path)
print(f"Inference model saved to {inference_model_path}")

plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    predicted_classes = tf.argmax(predictions, axis=1)
    
    num_images = min(16, len(images))
    for i in range(num_images):
        ax = plt.subplot(4, 4, i+1)
        plt.imshow((images[i].numpy()).astype("uint8"))
        plt.title(f"True: {ordered_descriptions[labels[i]]}\nPred: {ordered_descriptions[predicted_classes[i]]}")
        plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "predictions.png"))
plt.show()

test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.4f}")