import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
import csv

BATCH_SIZE = 32
IMG_SIZE = (224,224)
SEED = 123
EPOCHS = 10

ORIGINAL_DS_DIR = "lisa_dataset"
DS_DIR = "dataset"
CLASS_DS_DIR = "classified_dataset"
STATES = ["go", "stop", "warning", "stopLeft", "goLeft"]
STATE_DIRS = {state: os.path.join(CLASS_DS_DIR, state) for state in STATES}
ANNOTATIONS_DIR = "lisa_dataset/Annotations/Annotations"
sequences = ["daySequence1", "daySequence2", "dayTrain", "nightSequence1", "nightSequence2", "nightTrain"]

os.makedirs(DS_DIR, exist_ok=True)
os.makedirs(CLASS_DS_DIR, exist_ok=True)


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


def visualize_predictions(model, dataset, num_images=4):
    """
    Visualizes predictions against true masks.
    """
    for i, (images, masks) in enumerate(dataset.take(num_images)):
        pred_masks = model.predict(images)

        pred_masks = (pred_masks > 0.5).astype("float32")

        plt.figure(figsize=(12, 4))
        for j in range(num_images):
            # Image
            plt.subplot(1, 3, 1)
            plt.imshow(images[j])
            plt.title("Image")
            plt.axis('off')

            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(masks[j])
            plt.title("True Mask")
            plt.axis('off')

            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(pred_masks[j])
            plt.title("Predicted Mask")
            plt.axis('off')

        plt.show()

#Only for testing images/predictions
def process_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image


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



full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    CLASS_DS_DIR,
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

model = tf.keras.Sequental([
    tf.keras.layers.Input(shape=(224, 224, 3)),

    #Data augmentation later on

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(5, activation='softmax', dtype='float32')
])

print(model.summary())

if not os.path.exists("traffic_light.h5"):

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
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
        full_dataset,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[checkpoint]
    )

    model.save("lane_detection_model.h5")
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

