import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
import csv
import random

BATCH_SIZE = 32
IMG_SIZE = (224,224)
SEED = 123
EPOCHS = 20

ORIGINAL_DS_DIR = "lisa_dataset"
DS_DIR = "dataset"
CLASS_DS_DIR = "classified_dataset"
STATES = ["go", "stop", "warning", "stopLeft", "goLeft"]
STATE_DIRS = {state: os.path.join(CLASS_DS_DIR, state) for state in STATES}
ANNOTATIONS_DIR = "lisa_dataset/Annotations/Annotations"
sequences = ["daySequence1", "daySequence2", "dayTrain", "nightSequence1", "nightSequence2", "nightTrain"]
CROP_DS = "cropped_dataset"

os.makedirs(DS_DIR, exist_ok=True)
os.makedirs(CLASS_DS_DIR, exist_ok=True)
os.makedirs(CROP_DS, exist_ok=True)


def crop_dataset():
    os.makedirs(CROP_DS, exist_ok=True)
    
    all_annotations = {}
    print("Loading annotations from original dataset...")
    
    for seq in sequences:
        seq_dir = os.path.join(ANNOTATIONS_DIR, seq)
        seq_ann_file = os.path.join(seq_dir, "frameAnnotationsBOX.csv")

        if os.path.exists(seq_ann_file):
            ann_file = seq_ann_file
            print(f"Reading annotations from {ann_file}")
            
            try:
                with open(ann_file, 'r') as f:
                    print(f"First lines of {ann_file}")
                    for i, line in enumerate(f):
                        if i < 3:
                            print(f"Line {i}: {line.strip()}")
                        else:
                            break
                    f.seek(0)
                    next(f, None)
                    
                    for line in f:
                        parts = line.strip().split(';')
                        if len(parts) < 6:
                            continue
                        
                        full_path = parts[0]
                        filename = os.path.basename(full_path)
                        state = parts[1]
                        
                        x1 = int(float(parts[2]))
                        y1 = int(float(parts[3]))
                        x2 = int(float(parts[4]))
                        y2 = int(float(parts[5]))
                        
                        all_annotations[filename] = {
                            'state': state,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        }

                        if random.random() < 0.01:
                            print(f"DEBUG: Annotation for {filename} from {seq}: state={state}, box=({x1},{y1},{x2},{y2})")
            except Exception as e:
                print(f"Error reading {ann_file}: {e}")
                
        else:
            clip_dirs = []
            
            if os.path.exists(seq_dir):
                clip_dirs = [d for d in os.listdir(seq_dir) 
                            if d.startswith("Clip") and os.path.isdir(os.path.join(seq_dir, d))]
            
            if clip_dirs:
                print(f"Found {len(clip_dirs)} Clip subfolders for {seq}")
                
                for clip in clip_dirs:
                    clip_dir = os.path.join(seq_dir, clip)
                    clip_ann_file = os.path.join(clip_dir, "frameAnnotationsBOX.csv")
                    
                    if os.path.exists(clip_ann_file):
                        print(f"Reading annotations from {clip_ann_file}")
                        
                        try:
                            with open(clip_ann_file, 'r') as f:
                                next(f, None)  # Skip header
                                
                                for line in f:
                                    parts = line.strip().split(';')
                                    if len(parts) < 6:
                                        continue
                                    
                                    full_path = parts[0]
                                    filename = os.path.basename(full_path)
                                    state = parts[1]
                                    
                                    x1 = int(float(parts[2]))
                                    y1 = int(float(parts[3]))
                                    x2 = int(float(parts[4]))
                                    y2 = int(float(parts[5]))
                                    
                                    all_annotations[filename] = {
                                        'state': state,
                                        'x1': x1,
                                        'y1': y1,
                                        'x2': x2,
                                        'y2': y2
                                    }
                        except Exception as e:
                            print(f"Error reading {clip_ann_file}: {e}")
    
    print(f"Loaded {len(all_annotations)} total annotations")
    
    for state in STATES:

        invalid_box_count = 0
        load_error_count = 0
        empty_crop_count = 0
        success_count = 0

        source_dir = os.path.join(CLASS_DS_DIR, state)
        target_dir = os.path.join(CROP_DS, state)
        
        if not os.path.exists(source_dir):
            print(f"Directory not found: {source_dir}")
            continue
            
        os.makedirs(target_dir, exist_ok=True)
        
        images = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(images)} images in {state} directory")
        
        if images:
            sample_imgs = random.sample(images, min(3, len(images)))
            print(f"Sample images: {sample_imgs}")
            
            for img in sample_imgs:
                if img in all_annotations:
                    print(f"✓ {img} has annotations: {all_annotations[img]}")
                else:
                    print(f"✗ {img} does not have annotations")
        
        matched_count = sum(1 for img in images if img in all_annotations)
        print(f"Images with matching annotations: {matched_count}/{len(images)} ({matched_count/len(images)*100:.1f}%)")

        cropped_count = 0
        print_image = 0
        print_image_limit = 4
        print(f"\nDiagnostic information for {state}:")
        sample_boxes = []
        for img_filename in tqdm(images, desc=f"Cropping {state} images"):
            print_image += 1
            if img_filename in all_annotations and print_image < print_image_limit:
                annotation = all_annotations[img_filename]
                print(f"Cropping {img_filename} with box ({annotation['x1']}, {annotation['y1']}) to ({annotation['x2']}, {annotation['y2']})")
            if img_filename not in all_annotations:
                continue
                
            annotation = all_annotations[img_filename]
            
            x1 = annotation['x1']
            y1 = annotation['y1']
            x2 = annotation['x2']
            y2 = annotation['y2']
            
            if len(sample_boxes) < 5:
                sample_boxes.append((img_filename, x1, y1, x2, y2, x2-x1, y2-y1))
            
            if x1 >= x2 or y1 >= y2:
                invalid_box_count += 1
                continue
                
            try:
                img_path = os.path.join(source_dir, img_filename)
                crop_path = os.path.join(target_dir, img_filename)
                
                image = tf.keras.preprocessing.image.load_img(img_path)
                image_array = tf.keras.preprocessing.image.img_to_array(image)
                print(f"Image size: {image_array.shape}, Crop box: ({x1},{y1}) to ({x2},{y2})")
                
                crop = image_array[y1:y2, x1:x2]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    resized = tf.image.resize(crop, IMG_SIZE)
                    tf.keras.preprocessing.image.save_img(crop_path, resized.numpy())
                    cropped_count += 1
                    success_count += 1
                else:
                    empty_crop_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"Successfully cropped {cropped_count} images for state: {state}")

        print(f"\nCropping results for {state}:")
        print(f"- Total images with annotations: {len(images)}")
        print(f"- Invalid bounding boxes: {invalid_box_count}")
        print(f"- Image loading errors: {load_error_count}")
        print(f"- Empty crops: {empty_crop_count}")
        print(f"- Successfully cropped: {success_count}")

        print("\nSample bounding boxes (filename, x1, y1, x2, y2, width, height):")
        for box in sample_boxes:
            print(f"- {box[0]}: ({box[1]}, {box[2]}) to ({box[3]}, {box[4]}) - Size: {box[5]}×{box[6]} pixels")

        print(f"Cropping complete. Results saved to {CROP_DS}")


def copy_images_based_on_annotations(seq_dir, annotation_file):
    with open(annotation_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for row in reader:
            if len(row) >= 2:
                full_image_path = row[0]  # Filename
                image_name = os.path.basename(full_image_path)

                light_state = row[1]  # Annotation tag (go, stop, warning)

                if light_state in STATE_DIRS:
                    img_src_path = os.path.join(seq_dir, image_name)

                    if os.path.exists(img_src_path):
                        img_dest_path = os.path.join(STATE_DIRS[light_state], image_name)

                        if not os.path.exists(img_dest_path):
                            print(f"Copying {img_src_path} to {img_dest_path}")
                            shutil.copy2(img_src_path, img_dest_path)
                        else:
                            print(f"File already exists: {img_dest_path} — Skipping.")
                    else:
                        print(f"File not found: {img_src_path}")
                else:
                    print(f"Skipping invalid light state: {light_state} in row: {row}")
            else:
                print(f"Skipping malformed row: {row}")

def classify_dataset(dataset):
    for state_dir in STATE_DIRS.values():
        os.makedirs(state_dir, exist_ok=True)

    for seq in sequences:
        seq_dir = os.path.join(DS_DIR, seq)
        annotation_dir = os.path.join(ANNOTATIONS_DIR, seq)

        if not os.path.exists(seq_dir):
            print(f"Skipping {seq}, folder not found.")
            continue
        
        for ann_file in ["frameAnnotationsBOX.csv", "frameAnnotationsBULB.csv"]:
            ann_file_path = os.path.join(annotation_dir, ann_file)
            if os.path.exists(ann_file_path):
                print(f"Processing {seq} with annotation file {ann_file}")
                copy_images_based_on_annotations(seq_dir, ann_file_path)
                break

def process_dataset(dataset):
    if not os.path.exists(DS_DIR):
        os.makedirs(DS_DIR)

    print(f"Sequences: {sequences}")
    print(f"Original Dataset Directory: {dataset}")
    print(f"Dataset Directory: {DS_DIR}")

    for seq in tqdm(sequences, desc="Processing Sequences"):
        possible_seq_dirs = [
            os.path.join(dataset, seq, seq),
            os.path.join(dataset, seq)
        ]

        seq_dir = None
        for d in possible_seq_dirs:
            if os.path.exists(d):
                seq_dir = d
                break

        annotation_dir = os.path.join(dataset, "Annotations", "Annotations", seq)

        if not seq_dir or not os.path.exists(annotation_dir):
            print(f"Skipping {seq} — couldn't find data or annotations.")
            continue

        frames_path = os.path.join(seq_dir, "frames")
        has_frames_dir = os.path.exists(frames_path)
        has_subfolders = any(
            os.path.isdir(os.path.join(seq_dir, f)) and f != "frames" for f in os.listdir(seq_dir)
        )

        sequence_folder = os.path.join(DS_DIR, seq)
        os.makedirs(sequence_folder, exist_ok=True)

        merged_annotation_lines = []
        header_written = False

        if has_subfolders:
            subfolders = [f for f in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, f))]
            for subfolder in tqdm(subfolders, desc=(f"{seq}: Processing Subfolders"), leave=False):
                subfolder_path = os.path.join(seq_dir, subfolder)
                frames_dir = os.path.join(subfolder_path, "frames")
                if not os.path.exists(frames_dir):
                    frames_dir = subfolder_path

                image_files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
                for img_file in tqdm(image_files, desc=f"{seq}/{subfolder}: Saving Images", leave=False):
                    img_path = os.path.join(frames_dir, img_file)
                    image = process_image(img_path)
                    save_path = os.path.join(sequence_folder, img_file)
                    tf.keras.preprocessing.image.save_img(save_path, image * 255.0)

                for ann_file in ["frameAnnotationsBOX.csv", "frameAnnotationsBULB.csv"]:
                    ann_path = os.path.join(annotation_dir, subfolder, ann_file)
                    if os.path.exists(ann_path):
                        with open(ann_path, "r") as f:
                            lines = f.readlines()
                            if not header_written:
                                merged_annotation_lines.append(lines[0])  # header
                                header_written = True
                            merged_annotation_lines.extend(lines[1:])  # data rows
                        break
        elif has_frames_dir:
            frames_dir = frames_path
            if not os.path.exists(frames_dir):
                print(f"[{seq}] Expected frames/ folder, but not found. Trying to use base seq_dir.")
                frames_dir = seq_dir

            image_files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
            for img_file in tqdm(image_files, desc=f"{seq}: Saving Images", leave=False):
                img_path = os.path.join(frames_dir, img_file)
                image = process_image(img_path)
                save_path = os.path.join(sequence_folder, img_file)
                tf.keras.preprocessing.image.save_img(save_path, image * 255.0)

            for ann_file in ["frameAnnotationsBOX.csv", "frameAnnotationsBULB.csv"]:
                ann_path = os.path.join(annotation_dir, ann_file)
                if os.path.exists(ann_path):
                    with open(ann_path, "r") as f:
                        lines = f.readlines()
                        merged_annotation_lines.extend(lines)
                    break
        else:
            print(f"[{seq}] No frames/ folder or subfolders found — skipping sequence.")

        if merged_annotation_lines:
            merged_path = os.path.join(sequence_folder, "frameAnnotations.csv")
            with open(merged_path, "w") as out_file:
                out_file.writelines(merged_annotation_lines)


def visualize_predictions(model, dataset, num_batches=1, show_ground_truth=True):
    class_names = ['go', 'goLeft', 'stop', 'stopLeft', 'warning']
    
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
    tf.keras.layers.RandomContrast(0.2),
])

if not os.path.exists(os.path.join(DS_DIR, "daySequence1")):
    print("Dataset hasn't been processed yet. Processing now...")
    process_dataset(ORIGINAL_DS_DIR)
else:
    print("Dataset is already processed. Skipping dataset processing.")




if not os.path.exists(os.path.join(CLASS_DS_DIR, "go")):
    print("Dataset hasn't been classified yet. Processing now...")
    classify_dataset(DS_DIR)
else:
    print("Dataset is already classified. Skipping dataset processing.")

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
    seed=SEED
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

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    data_augmentation,

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(5, activation='softmax', dtype='float32')
])


print(model.summary())

if not os.path.exists("traffic_light.h5"):

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_light_model.h5",
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
        callbacks=[checkpoint]
    )

    model.save("traffic_light.h5")
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


else:
    model = tf.keras.models.load_model("traffic_light.h5")
    print("Model loaded successfully.")
    visualize_predictions(model, val_ds)

