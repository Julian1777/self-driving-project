import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
import csv
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import json
import cv2 as cv
from PIL import Image

BATCH_SIZE = 64
IMG_SIZE = (64,64)
SEED = 123
EPOCHS = 20
PADDING = 10

ORIGINAL_DS_DIR = "dtld_dataset"
DS_DIR = "dataset"
CLASS_DS_DIR = "classified_dataset"
STATES = ["green", "red", "yellow", "off"]
STATE_DIRS = {state: os.path.join(CLASS_DS_DIR, state) for state in STATES}
ANNOATION = os.path.join(ORIGINAL_DS_DIR, "Berlin.json")
CROP_DS = os.path.join(ORIGINAL_DS_DIR, "cropped_dataset")


def crop_dataset():
    
    os.makedirs(CROP_DS, exist_ok=True)
    print("Loading annotations from original dataset...")
    
    with open(ANNOATION, "r") as f:
        data = json.load(f)

    for entry in data["images"]:
        rel_path = entry["image_path"]
        full_path = os.path.join(ORIGINAL_DS_DIR, rel_path)
        try:
            pil_image = Image.open(full_path)
        except Exception as e:
            print(f"Failed to open {full_path}: {e}")
            continue
        
        np_img = np.array(pil_image)
        if np_img.ndim == 2:
            np_img = np.stack([np_img]*3, axis=-1)
        h_img, w_img = np_img.shape[:2]
        print(f"Image size: {h_img}x{w_img}")
        print(f"Image shape: {np_img.shape}, dtype: {np_img.dtype}, min: {np.min(np_img)}, max: {np.max(np_img)}")


        for label in entry["labels"]:
            attr = label["attributes"]
            
            if attr["relevance"] != "relevant" or attr["direction"] != "front":
                continue

            x, y, w, h = label["x"], label["y"], label["w"], label["h"]
            state = attr["state"]
            
            x0 = max(x - PADDING, 0)
            y0 = max(y - PADDING, 0)
            x1 = min(x + w + PADDING, w_img)
            y1 = min(y + h + PADDING, h_img)

            crop = np_img[y0:y1, x0:x1]
            if crop.size == 0:
                print(f"Empty crop at {rel_path} box {x,y,w,h}")
                continue

            crop_f = crop.astype(np.float32)
            minv, maxv = crop_f.min(), crop_f.max()
            if maxv > minv:
                crop_norm = ((crop_f - minv) / (maxv - minv) * 255.0).astype(np.uint8)
            else:
                crop_norm = np.zeros_like(crop, dtype=np.uint8)

            crop_resized = cv.resize(
                crop_norm,
                IMG_SIZE,
                interpolation=cv.INTER_CUBIC
            )

            crop_bgr = cv.cvtColor(crop_resized, cv.COLOR_RGB2BGR)

            state_dir = os.path.join(CROP_DS, state)
            os.makedirs(state_dir, exist_ok=True)

            base = os.path.splitext(os.path.basename(rel_path))[0]
            out_name = f"{base}_{x}_{y}.jpg"
            out_path = os.path.join(state_dir, out_name)

            cv.imwrite(out_path, crop_bgr)
            print(f"Saved {state} crop: {out_path}")


def visualize_predictions(model, dataset, num_batches=1, show_ground_truth=True):
    class_names = ['green', 'red', 'yellow', 'off']
    
    for batch_images, batch_labels in dataset.take(num_batches):
        predictions = model.predict(batch_images, verbose=0)
        pred_classes = tf.argmax(predictions, axis=1)
        
        num_images = min(len(batch_images), 12)
        columns = 3
        rows = (num_images + columns - 1) // columns
        
        plt.figure(figsize=(12, 4 * rows))
        
        for i in range(num_images):
            plt.subplot(rows, columns, i+1)
            plt.imshow(batch_images[i].numpy().astype("uint8"))
            
            pred_idx = pred_classes[i]
            confidence = predictions[i][pred_idx]
            pred_class = class_names[pred_idx]
            
            if show_ground_truth:
                true_class = class_names[batch_labels[i]]
                title = f"Prediction: {pred_class}\n"
                title += f"Confidence: {confidence:.2f}\n"
                title += f"Actual: {true_class}"
                color = "green" if pred_class == true_class else "red"
            else:
                title = f"Prediction: {pred_class}\n"
                title += f"Confidence: {confidence:.2f}"
                color = "black"
                
            plt.title(title, color=color, fontsize=10)
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()


def create_hsv_features(image):
    hsv = tf.image.rgb_to_hsv(image)
    
    h_channel = hsv[:, :, :, 0]  # Hue
    s_channel = hsv[:, :, :, 1]  # Saturation  
    v_channel = hsv[:, :, :, 2]  # Value (brightness)
    
    brightness = tf.reduce_mean(v_channel, axis=[1, 2])
    
    # Red: H in [0, 0.05] or [0.95, 1.0]
    red_hue_mask1 = tf.logical_and(h_channel >= 0.0, h_channel <= 0.05)
    red_hue_mask2 = tf.logical_and(h_channel >= 0.95, h_channel <= 1.0)
    red_hue_mask = tf.logical_or(red_hue_mask1, red_hue_mask2)
    red_mask = tf.logical_and(red_hue_mask, 
                 tf.logical_and(s_channel > 0.4, v_channel > 0.2))
    red_count = tf.reduce_sum(tf.cast(red_mask, tf.float32), axis=[1, 2])
    
    # Yellow: H in [0.10, 0.17]
    yellow_hue_mask = tf.logical_and(h_channel >= 0.10, h_channel <= 0.17)
    yellow_mask = tf.logical_and(yellow_hue_mask, 
                   tf.logical_and(s_channel > 0.4, v_channel > 0.2))
    yellow_count = tf.reduce_sum(tf.cast(yellow_mask, tf.float32), axis=[1, 2])
    
    # Green: H in [0.25, 0.45]
    green_hue_mask = tf.logical_and(h_channel >= 0.25, h_channel <= 0.45)
    green_mask = tf.logical_and(green_hue_mask, 
                  tf.logical_and(s_channel > 0.4, v_channel > 0.2))
    green_count = tf.reduce_sum(tf.cast(green_mask, tf.float32), axis=[1, 2])
    
    total_pixels = tf.cast(tf.shape(h_channel)[1] * tf.shape(h_channel)[2], tf.float32)
    red_ratio = red_count / total_pixels
    yellow_ratio = yellow_count / total_pixels
    green_ratio = green_count / total_pixels
    
    hsv_features = tf.stack([brightness, red_ratio, yellow_ratio, green_ratio], axis=1)
    return hsv_features

#Only for testing images/predictions
def process_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    #Color Jitter
    tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_contrast(x, lower=0.8, upper=1.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, lower=0.8, upper=1.2)),
    tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, max_delta=0.05))
])



# if not os.path.exists(os.path.join(DS_DIR, "daySequence1")):
#     print("Dataset hasn't been processed yet. Processing now...")
#     process_dataset(ORIGINAL_DS_DIR)
# else:
#     print("Dataset is already processed. Skipping dataset processing.")

# if not os.path.exists(os.path.join(CLASS_DS_DIR, "go")):
#     print("Dataset hasn't been classified yet. Processing now...")
#     classify_dataset(DS_DIR)
# else:
#     print("Dataset is already classified. Skipping dataset processing.")

if not os.path.exists(os.path.join(CROP_DS, "go")):
    print("Dataset hasn't been cropped yet. Processing now...")
    crop_dataset()
else:
    print("Dataset is already cropped. Skipping dataset processing.")

full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    CROP_DS,
    batch_size=None,
    image_size=IMG_SIZE,
    shuffle=True,
    subset="training",
    validation_split=0.2,
    seed=SEED,
    class_names=STATES
)

print(full_dataset.class_names)

dataset_size = sum(1 for _ in full_dataset)
print(f"Total images: {dataset_size}")


train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

full_dataset = full_dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=False)

train_ds = full_dataset.take(train_size)
val_ds = full_dataset.skip(train_size).take(val_size)
test_ds = full_dataset.skip(train_size + val_size)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


print(f"Train dataset size: {len(train_ds)} batches")
print(f"Validation dataset size: {len(val_ds)} batches")
print(f"Test dataset size: {len(test_ds)} batches")

all_train_labels = []
for _, labels in train_ds.unbatch():
    all_train_labels.append(int(labels.numpy()))

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(STATES)),
    y=all_train_labels
)
class_weights = {i: w for i, w in enumerate(weights)}
print("Class weights:", class_weights)

early_stopping = EarlyStopping(
    'val_accuracy', 
    patience=3, 
    restore_best_weights=True
)

def build_traffic_light_model():
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    
    augmented = data_augmentation(inputs)
    
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(augmented)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2,2)(x)
    
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2,2)(x)
    
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2,2)(x)
    
    x = tf.keras.layers.Conv2D(256, (3,3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2,2)(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    cnn_features = tf.keras.layers.Dense(128, activation='relu')(x)
    cnn_features = tf.keras.layers.Dropout(0.5)(cnn_features)
    
    hsv_features = tf.keras.layers.Lambda(create_hsv_features)(augmented)
    hsv_branch = tf.keras.layers.Dense(16, activation='relu')(hsv_features)
    
    combined = tf.keras.layers.Concatenate()([cnn_features, hsv_branch])
    
    outputs = tf.keras.layers.Dense(3, activation='softmax', dtype='float32')(combined)
    
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, hsv_features])
    
    return model


model = build_traffic_light_model()

print(model.summary())

if os.path.exists("traffic_light_classification.h5"):
    print("Model already exists. Loading model...")
    model = tf.keras.models.load_model("traffic_light_classification.h5")
    print("Model loaded successfully.")
    visualize_predictions(model, val_ds)
elif os.path.exists("light_classification_checkpoint.h5"):
    print("Using Checkpoint model...")
    model = tf.keras.models.load_model("light_classification_checkpoint.h5") 
    print("Model loaded successfully.")
    visualize_predictions(model, val_ds)
else:

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "light_classification_checkpoint.h5",
        save_weights_only=True,
        monitor="accuracy",
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
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    model.save("traffic_light_classification.h5")
    visualize_predictions(model, val_ds)

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

