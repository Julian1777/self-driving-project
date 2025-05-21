import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import cv2
import collections


BATCH_SIZE = 64
IMG_SIZE = (224, 224)
SEED = 123
EPOCHS = 50

KAGGLE_PATH = "/kaggle/input/gtsrb-dataset/dataset/dataset"
TRAIN_PATH = os.path.join(KAGGLE_PATH, "Train")
TEST_PATH = os.path.join(KAGGLE_PATH, "Test")
SAVE_PATH = "/kaggle/working"

os.makedirs(SAVE_PATH, exist_ok=True)
        
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
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

def augment_image(image, label):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    if tf.random.uniform(()) > 0.7:
        image = tf.image.resize(image, [IMG_SIZE[0] // 2, IMG_SIZE[1] // 2])
        image = tf.image.resize(image, IMG_SIZE)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
    image = tf.clip_by_value(image + noise, 0.0, 255.0)
    return image, label


normalization_layer = tf.keras.layers.Rescaling(1./255)

print("Path to dataset:", TRAIN_PATH)

train_val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.1,
    subset="training",
    seed=SEED
)

val_test_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.1,
    subset="validation",
    seed=SEED
)

original_class_names = train_val_ds.class_names
num_classes = len(original_class_names)
print(f"Number of classes: {num_classes}")

try:
    class_names = load_class_names("/kaggle/input/gtsrb-dataset/sign_dic.csv")
except:
    class_names = load_class_names("sign_dic.csv")
    if not class_names:
        print("Warning: Could not load class descriptions. Using directory names instead.")
        class_names = {name: name for name in original_class_names}

val_batches = int(0.5 * len(val_test_ds))
val_ds = val_test_ds.take(val_batches)
test_ds = val_test_ds.skip(val_batches)

print(f"Train dataset size: {len(train_val_ds)} batches")
print(f"Validation dataset size: {len(val_ds)} batches")
print(f"Test dataset size: {len(test_ds)} batches")

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

directory_to_index = {class_name: idx for idx, class_name in enumerate(original_class_names)}
print("First few directory_to_index mappings:", list(directory_to_index.items())[:5])

csv_classid_to_index = {}

for i, class_name in enumerate(original_class_names):
    if class_name.isdigit():
        class_id = int(class_name)
        csv_classid_to_index[class_id] = i

for class_id in range(num_classes):
    if class_id not in csv_classid_to_index and str(class_id) in directory_to_index:
        csv_classid_to_index[class_id] = directory_to_index[str(class_id)]

print(f"Created CSV ClassId to model index mapping with {len(csv_classid_to_index)} entries")
print(f"Sample mappings: {list(csv_classid_to_index.items())[:10]}")


train_val_ds = train_val_ds.map(
    augment_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_val_ds = train_val_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_val_ds = train_val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

label_counts = collections.Counter()
for _, labels in train_val_ds:
    flat_labels = labels.numpy().flatten()
    for label in flat_labels:
        label_counts[int(label)] += 1

total = sum(label_counts.values())
class_weights = {i: total / (len(label_counts) * count) for i, count in label_counts.items()}

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu', dtype='float32'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
])

print(model.summary())

if not os.path.exists(os.path.join(SAVE_PATH, "sign_classification")):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(SAVE_PATH, "best_sign_classification"),
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
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping]
    )
    
    print("Phase 2: Fine-tuning the model...")
    base_model.trainable = True
    
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
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    combined_history = {}
    for key in history_phase1.history:
        combined_history[key] = history_phase1.history[key] + history_phase2.history[key]
    
    final_model_path = os.path.join(SAVE_PATH, "sign_classification")
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
else:
    print("Evaluating model on validation data...")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    validation_loss, validation_accuracy = model.evaluate(val_ds)
    print(f"Validation accuracy: {validation_accuracy*100:.2f}%")
    
    plt.figure(figsize=(16, 16))
    
    for images, labels in val_ds.take(1):
        predictions = model.predict(images, verbose=0)
        predicted_classes = tf.argmax(predictions, axis=1)
        
        num_images = min(16, len(images))
        for i in range(num_images):
            ax = plt.subplot(4, 4, i+1)
            
            display_image = images[i].numpy() * 255.0
            plt.imshow(display_image.astype("uint8"))
            
            confidence = predictions[i][predicted_classes[i]].numpy() * 100
            
            true_label = labels[i].numpy()
            pred_label = predicted_classes[i].numpy()
            is_correct = true_label == pred_label
            
            title_color = "green" if is_correct else "red"
            plt.title(
                f"True: {ordered_descriptions[true_label]}\n"
                f"Pred: {ordered_descriptions[pred_label]}\n"
                f"Conf: {confidence:.1f}%",
                color=title_color, fontsize=8
            )
            plt.axis("off")
            
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "validation_predictions.png"))
    plt.show()