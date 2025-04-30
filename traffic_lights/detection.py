import os
import pandas as pd
import shutil
import random
import glob
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models

# Config
SOURCE_DIR = "dataset"
TARGET_DIR = "yolo_dataset"
TRAIN_RATIO = 0.9
CLASS_ID = 0

os.makedirs(f"{TARGET_DIR}/images/train", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/images/val", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/labels/val", exist_ok=True)

def process_subfolder(subfolder_path):
    annotation_files = glob.glob(os.path.join(subfolder_path, "*.csv"))
    
    if not annotation_files:
        print(f"No annotation file found in {subfolder_path}")
        return []
    
    annotation_file = annotation_files[0]
    print(f"Found annotation file: {annotation_file}")
    
    # Get all image files in this subfolder
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(glob.glob(os.path.join(subfolder_path, f"*{ext}")))
    
    if not image_files:
        print(f"No images found in {subfolder_path}")
        return []
    
    print(f"Found {len(image_files)} images in {subfolder_path}")
    
    try:
        # Parse the CSV file
        try:
            df = pd.read_csv(annotation_file, sep=';')
        except:
            try:
                df = pd.read_csv(annotation_file, sep=',')
            except:
                print(f"Could not parse annotation file: {annotation_file}")
                return []
        
        # Handle column mappings
        column_mappings = {
            'filename': ['Filename', 'filename', 'file', 'File', 'Filename:', 'Image'],
            'Upper left corner X': ['Upper left corner X', 'Roi.X1', 'X1', 'x1', 'xmin'],
            'Upper left corner Y': ['Upper left corner Y', 'Roi.Y1', 'Y1', 'y1', 'ymin'],
            'Lower right corner X': ['Lower right corner X', 'Roi.X2', 'X2', 'x2', 'xmax'],
            'Lower right corner Y': ['Lower right corner Y', 'Roi.Y2', 'Y2', 'y2', 'ymax']
        }
        
        column_map = {}
        for std_name, alternatives in column_mappings.items():
            for alt in alternatives:
                if alt in df.columns:
                    column_map[alt] = std_name
                    break
        
        if len(column_map) < 5:
            missing = set(column_mappings.keys()) - set(column_map.values())
            print(f"Missing columns in {annotation_file}: {missing}")
            print(f"Available columns: {df.columns.tolist()}")
            return []
        
        df = df.rename(columns=column_map)
        
        # Create image data from annotations
        image_data = []
        
        for _, row in df.iterrows():
            try:
                # Get filename and strip any path components
                full_filename = row['filename']
                # Extract just the base filename without any directories
                base_filename = os.path.basename(full_filename)
                
                # Look for matching image in the subfolder
                matching_file = None
                for img_file in image_files:
                    if os.path.basename(img_file) == base_filename:
                        matching_file = img_file
                        break
                
                # Try with different extensions if not found
                if not matching_file:
                    name_without_ext = os.path.splitext(base_filename)[0]
                    for img_file in image_files:
                        if os.path.basename(img_file).startswith(name_without_ext + '.'):
                            matching_file = img_file
                            break
                
                # Still not found - try if any file contains this name
                if not matching_file:
                    name_without_ext = os.path.splitext(base_filename)[0]
                    for img_file in image_files:
                        if name_without_ext in os.path.basename(img_file):
                            matching_file = img_file
                            break
                
                if not matching_file:
                    print(f"Could not find matching image for {base_filename}")
                    continue
                
                # Process the coordinates
                with Image.open(matching_file) as img:
                    w, h = img.size
                
                x1 = float(row['Upper left corner X'])
                y1 = float(row['Upper left corner Y'])
                x2 = float(row['Lower right corner X'])
                y2 = float(row['Lower right corner Y'])
                
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Constrain to valid range
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_annotation = f"{CLASS_ID} {x_center} {y_center} {width} {height}"
                
                image_data.append({
                    'image_path': matching_file,
                    'annotation': yolo_annotation
                })
                
            except Exception as e:
                print(f"Error processing row {_}: {e}")
                continue
        
        return image_data
        
    except Exception as e:
        print(f"Error processing annotation file {annotation_file}: {e}")
        return []

def main():
    subfolders = [os.path.join(SOURCE_DIR, d) for d in os.listdir(SOURCE_DIR) 
                 if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    print(f"Found {len(subfolders)} subfolders to process")
    
    all_image_data = []
    for subfolder in subfolders:
        image_data = process_subfolder(subfolder)
        all_image_data.extend(image_data)
        print(f"Added {len(image_data)} images from {subfolder}")
    
    print(f"Total images with annotations: {len(all_image_data)}")
    
    random.shuffle(all_image_data)
    split_idx = int(len(all_image_data) * TRAIN_RATIO)
    
    train_data = all_image_data[:split_idx]
    val_data = all_image_data[split_idx:]
    
    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    
    def process_dataset(dataset, name):
        print(f"Processing {name} dataset...")
        for i, item in enumerate(tqdm(dataset)):
            original_filename = os.path.basename(item['image_path'])
            base_name, ext = os.path.splitext(original_filename)
            
            new_filename = f"{base_name}_{i}{ext}"
            
            img_dest = os.path.join(TARGET_DIR, "images", name, new_filename)
            label_dest = os.path.join(TARGET_DIR, "labels", name, f"{base_name}_{i}.txt")
            
            shutil.copy2(item['image_path'], img_dest)
            
            with open(label_dest, 'w') as f:
                f.write(item['annotation'])
    
    process_dataset(train_data, "train")
    process_dataset(val_data, "val")
    
    yaml_content = f"""
# Traffic Lights Dataset
path: {os.path.abspath(TARGET_DIR)}
train: images/train
val: images/val

# Classes
nc: 1  # number of classes
names: ['traffic_light']
"""

    with open(os.path.join(TARGET_DIR, "dataset.yaml"), 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset preparation complete! YAML file created at {os.path.join(TARGET_DIR, 'dataset.yaml')}")


def yolo_model(input_shape=(224,224,3), num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Output: [x_center, y_center, width, height, object_confidence]
    outputs = layers.Dense(5, activation='sigmoid')(x)

    return models.Model(inputs, outputs)


model = yolo_model()
if not os.path.exists('traffic_light_detection.h5'):

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
    )
    model.save('traffic_light_detection.h5')
else:
    model = tf.keras.models.load_model('traffic_light_detection.h5')


main()