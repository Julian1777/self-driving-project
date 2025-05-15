import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
import cv2 as cv
import numpy as np
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CULANE_DIR = os.path.join(SCRIPT_DIR, "dataset", "culane")
IMAGES_DIR = os.path.join(CULANE_DIR, "images", "lane")
ANNOTATIONS_DIR = os.path.join(CULANE_DIR, "annotations")
MASKS_DIR = os.path.join(CULANE_DIR, "masks")
MODEL_PATH = "lane_detection_model.h5"

os.makedirs(CULANE_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
 

IMG_SIZE = (224, 224)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
BATCH_SIZE = 16
SEED = 123
EPOCHS = 30

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

def iou_metric(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    iou = intersection / (union + tf.keras.backend.epsilon())
    
    return iou

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

def visualize_predictions(model, dataset, num_images=3):
    if isinstance(dataset, tf.data.Dataset):
        for i, (images, masks) in enumerate(dataset.take(1)):
            if i >= num_images:
                break
                
            display_count = min(num_images, images.shape[0])
            pred_masks = model.predict(images[:display_count])
            pred_masks = (pred_masks > 0.5).astype("float32")
            
            plt.figure(figsize=(15, 5*display_count))
            for j in range(display_count):
                plt.subplot(display_count, 3, j*3+1)
                plt.imshow(images[j])
                plt.title("Image")
                plt.axis('off')
                
                plt.subplot(display_count, 3, j*3+2)
                plt.imshow(masks[j], cmap='gray')
                plt.title("True Mask")
                plt.axis('off')
                
                plt.subplot(display_count, 3, j*3+3)
                plt.imshow(pred_masks[j], cmap='gray')
                plt.title("Predicted Mask")
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    else:
        if isinstance(dataset, tuple) and len(dataset) == 2:
            images, masks = dataset
        else:
            print("Invalid input to visualize_predictions")
            return
            
        if len(images) > num_images:
            images = images[:num_images]
            masks = masks[:num_images]
            
        pred_masks = model.predict(images)
        pred_masks = (pred_masks > 0.5).astype("float32")
        
        plt.figure(figsize=(15, 5*num_images))
        for j in range(num_images):
            plt.subplot(num_images, 3, j*3+1)
            plt.imshow(images[j])
            plt.title("Image")
            plt.axis('off')
            
            plt.subplot(num_images, 3, j*3+2)
            plt.imshow(masks[j], cmap='gray')
            plt.title("True Mask")
            plt.axis('off')
            
            plt.subplot(num_images, 3, j*3+3)
            plt.imshow(pred_masks[j], cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def load_image_mask_pair(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE)
    mask = tf.cast(mask > 127, tf.float32)

    return image, mask

def get_dataset(images_dir, masks_dir):
    image_paths = sorted([
        os.path.join(images_dir, f) 
        for f in os.listdir(images_dir) 
        if f.endswith(('.jpg', '.png'))
    ])
    
    mask_paths = sorted([
        os.path.join(masks_dir, f) 
        for f in os.listdir(masks_dir) 
        if f.endswith('.png')
    ])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image_mask_pair, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def process_dataset():

    def draw_lane_mask(anno_path, image_shape):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)  # H x W
        with open(anno_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                coords = list(map(float, line.strip().split()))
                points = [(int(coords[i]), int(coords[i+1])) for i in range(0, len(coords), 2)]
                for i in range(1, len(points)):
                    cv.line(mask, points[i-1], points[i], color=255, thickness=2)
        return mask

    dataset_folders = [
        "driver_161_90frame",
        "driver_23_30frame",
        "driver_182_30frame"
    ]
    
    output_dir = CULANE_DIR
    images_dir = IMAGES_DIR
    annotations_dir = ANNOTATIONS_DIR
    
    lane_class_dir = os.path.join(images_dir)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(lane_class_dir, exist_ok=True)

    img_count = 0
    total_subfolder_count = 0
    
    for folder_name in dataset_folders:
        base_dir = os.path.join(SCRIPT_DIR, "dataset", folder_name)
        
        if not os.path.exists(base_dir):
            print(f"Warning: Dataset directory not found at {base_dir}, skipping...")
            continue
            
        print(f"\nProcessing dataset folder: {folder_name}")
        
        subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
        total_subfolder_count += len(subfolders)
        print(f"Found {len(subfolders)} subfolders in {folder_name}")
        
        for subfolder in tqdm(subfolders, desc=f"Processing {folder_name} subfolders"):
            files = os.listdir(subfolder)
            
            img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            img_files = [f for f in files if any(f.endswith(ext) for ext in img_extensions)]
            
            for img_file in img_files:
                img_path = os.path.join(subfolder, img_file)
                img_basename, img_ext = os.path.splitext(img_file)
                
                anno_file = f"{img_basename}.lines.txt"
                anno_path = os.path.join(subfolder, anno_file)
                
                if not os.path.exists(anno_path):
                    continue
                
                img_count += 1
                new_img_name = f"img_{img_count}{img_ext}"
                new_anno_name = f"img_{img_count}_anno.txt"
                
                shutil.copy(img_path, os.path.join(lane_class_dir, new_img_name))
                shutil.copy(anno_path, os.path.join(annotations_dir, new_anno_name))
                
                img = cv.imread(img_path)
                mask = draw_lane_mask(anno_path, img.shape)
                mask = cv.resize(mask, IMG_SIZE)
                mask_output_path = os.path.join(MASKS_DIR, f"img_{img_count}.png")
                cv.imwrite(mask_output_path, mask)
                
                if img_count % 100 == 0:
                    print(f"Processed {img_count} images so far")
    
    print(f"\nProcessing complete. Organized {img_count} image-annotation pairs from {total_subfolder_count} subfolders across {len(dataset_folders)} datasets.")
    
    copied_images = len(os.listdir(images_dir))
    copied_annos = len(os.listdir(annotations_dir))
    masks_count = len(os.listdir(MASKS_DIR))
    print(f"Files in output directories: {copied_images} images, {copied_annos} annotations, {masks_count} masks")
    
    return {
        "base_dir": output_dir,
        "images_dir": images_dir,
        "annotations_dir": annotations_dir,
        "masks_dir": MASKS_DIR,
        "count": img_count
    }


def create_lane_segmenation_model(input_shape=INPUT_SHAPE):
    # More reliable model without shape issues
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Use MobileNetV2 as backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Encoder path
    # Get the output of the backbone
    encoder = base_model.output
    
    # Decoder path - Simple upsampling without skip connections
    # This is guaranteed to work without shape issues
    
    # Start with a 10×10 bottleneck (size may vary based on input)
    x = encoder
    
    # Series of upsampling blocks to reach original resolution
    # First upsampling: 10×10 -> 20×20
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    # Second upsampling: 20×20 -> 40×40
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    # Third upsampling: 40×40 -> 80×80
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    # Fourth upsampling: 80×80 -> 160×160
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    # Final upsampling: 160×160 -> 320×320
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    
    # Output layer
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

lane_images_path = os.path.join(IMAGES_DIR)

image_files = [f for f in os.listdir(lane_images_path) if f.endswith(('.jpg', '.png'))]
mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.png')]
annotaion_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.txt')]

print(f"Lane images folder: {lane_images_path} contains {len(image_files)} images.")
print(f"Annotations folder: {ANNOTATIONS_DIR} contains {len(annotaion_files)} annotaions.")
print(f"Masks folder: {MASKS_DIR} contains {len(mask_files)} annotaions.")

if os.path.exists(lane_images_path) and os.path.exists(ANNOTATIONS_DIR) and len(image_files) == len(mask_files) and len(mask_files) > 0:
    print(f"Dataset already processed. Found {len(image_files)} images and {len(mask_files)} masks.")
    processed_data = {
        "base_dir": CULANE_DIR,
        "images_dir": IMAGES_DIR,
        "annotations_dir": ANNOTATIONS_DIR,
        "count": len(image_files)
    }
else:
    print("Running dataset processing...")
    processed_data = process_dataset()
    

DATASET_PATH = processed_data["images_dir"]
print(f"Updated DATASET_PATH to: {DATASET_PATH}")

full_dataset = get_dataset(IMAGES_DIR, MASKS_DIR)

# 80-10-10 split
dataset_size = processed_data["count"]
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



if os.path.exists(MODEL_PATH):
    print(f"Loading existing model from {MODEL_PATH}")
    model = load_model(
        MODEL_PATH,
        custom_objects={'iou_metric': iou_metric}
    )
else:
    print("No existing model found. Training a new model.")

    model = create_lane_segmenation_model(input_shape=INPUT_SHAPE)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor="val_iou_metric",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_iou_metric',
        patience=6,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_iou_metric',
        mode='max',
        factor=0.5,
        patience=6,
        verbose=1,
        min_lr=1e-6
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
        loss=combined_loss, 
        metrics=[iou_metric]
    )

    print("Phase 1: Training with frozen backbone")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, reduce_lr, early_stopping]
    )

    print("Phase 2: Fine-tuning backbone layers")
    backbone = model.layers[1]
    backbone.trainable = True

    for layer in backbone.layers[:100]:  # Freeze first 100 layers
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 10x smaller
        loss=combined_loss,
        metrics=[iou_metric]
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,  # Additional epochs with fine-tuning
        callbacks=[checkpoint, reduce_lr]
    )

    combined_history = {}
    for key in history_phase1.history:
        combined_history[key] = history_phase1.history[key] + history_phase2.history[key]

    model.save(MODEL_PATH)
    visualize_predictions(model, val_ds)

    # Plotting loss, accuracy, and IoU for both train and validation sets
    plt.figure(figsize=(16, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(combined_history['loss'], label='Train Loss')
    plt.plot(combined_history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # IoU
    plt.subplot(1, 2, 2)
    plt.plot(combined_history['iou_metric'], label='Train IoU')
    plt.plot(combined_history['val_iou_metric'], label='Val IoU')
    plt.title('IoU over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    plt.show()