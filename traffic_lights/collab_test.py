def create_full_colab_dataset():
    """Creates a zip with the entire dataset for Google Colab"""
    import shutil
    import os
    from tqdm import tqdm
    
    print("Creating full dataset for Google Colab - this may be large!")
    
    COLAB_DIR = "colab_dataset"
    TARGET_DIR = "yolo_dataset"
    
    os.makedirs(os.path.join(COLAB_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(COLAB_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(COLAB_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(COLAB_DIR, "labels", "val"), exist_ok=True)
    
    train_dir = os.path.join(TARGET_DIR, "images", "train")
    labels_dir = os.path.join(TARGET_DIR, "labels", "train")
    
    print("Copying training images and labels...")
    for img_file in tqdm(os.listdir(train_dir)):
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        
        # Copy image
        shutil.copy2(
            os.path.join(train_dir, img_file),
            os.path.join(COLAB_DIR, "images", "train", img_file)
        )
        
        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy2(
                label_path,
                os.path.join(COLAB_DIR, "labels", "train", label_file)
            )
    
    val_dir = os.path.join(TARGET_DIR, "images", "val")
    val_labels = os.path.join(TARGET_DIR, "labels", "val")
    
    print("Copying validation images and labels...")
    for img_file in tqdm(os.listdir(val_dir)):
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        
        shutil.copy2(
            os.path.join(val_dir, img_file),
            os.path.join(COLAB_DIR, "images", "val", img_file)
        )
        
        if os.path.exists(os.path.join(val_labels, label_file)):
            shutil.copy2(
                os.path.join(val_labels, label_file),
                os.path.join(COLAB_DIR, "labels", "val", label_file)
            )
    
    train_count = len(os.listdir(os.path.join(COLAB_DIR, "images", "train")))
    val_count = len(os.listdir(os.path.join(COLAB_DIR, "images", "val")))
    total_count = train_count + val_count
    
    print(f"Dataset contains {train_count} training images and {val_count} validation images")
    print(f"Total: {total_count} images")
    
    with open(os.path.join(COLAB_DIR, "dataset.yaml"), 'w') as f:
        f.write(f"train: ./images/train\n")
        f.write(f"val: ./images/val\n")
        f.write(f"nc: 3\n")
        f.write(f"names: ['red', 'yellow', 'green']\n")
    
    if total_count > 20000:
        print(f"WARNING: Your dataset contains {total_count} images which may create a large zip file")
        print("This could be slow to upload to Colab. If you encounter issues, consider using the balanced version.")
    
    print("Creating zip file (this may take a while for large datasets)...")
    shutil.make_archive("colab_traffic_lights_full", 'zip', COLAB_DIR)
    print(f"Created full dataset zip at: {os.path.abspath('colab_traffic_lights_full.zip')}")

create_full_colab_dataset()