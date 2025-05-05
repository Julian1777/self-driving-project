import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
from tensorflow.keras.models import Model

# Constants
IMG_SIZE = (64, 64)
STATES = ["red", "yellow", "green"]
CHECKPOINT_PATH = "traffic_light_classification_checkpoint.h5"
SAVED_MODEL_PATH = "traffic_light_classification_savedmodel"

def process_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_resized = cv.resize(img, IMG_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, 0)
    
    return img, img_batch

def load_checkpoint_model():
    """Load model from checkpoint"""
    try:
        # Import from your classification module
        from classification import build_traffic_light_model, hsv_feature_extraction
        
        print("Building model from checkpoint...")
        model = build_traffic_light_model()
        model.load_weights(CHECKPOINT_PATH)
        print(f"Successfully loaded weights from {CHECKPOINT_PATH}")
        
        # Verify model loaded correctly
        model_size_mb = sum(np.prod(w.shape) * w.dtype.size for w in model.weights) / (1024 * 1024)
        print(f"Checkpoint model size: ~{model_size_mb:.2f} MB")
        
        return model
    except Exception as e:
        print(f"Error loading checkpoint model: {e}")
        return None

def load_saved_model():
    """Load model from SavedModel directory"""
    try:
        print(f"Loading model from {SAVED_MODEL_PATH}...")
        model = tf.saved_model.load(SAVED_MODEL_PATH)
        
        # Calculate approximate size
        model_size_mb = sum(os.path.getsize(os.path.join(SAVED_MODEL_PATH, f)) 
                          for f in os.listdir(SAVED_MODEL_PATH) 
                          if os.path.isfile(os.path.join(SAVED_MODEL_PATH, f))) / (1024 * 1024)
        print(f"SavedModel size: ~{model_size_mb:.2f} MB")
        
        return model
    except Exception as e:
        print(f"Error loading SavedModel: {e}")
        return None

def compare_models():
    """Compare predictions between checkpoint and SavedModel"""
    print("\n==== TRAFFIC LIGHT MODEL COMPARISON ====\n")
    
    # Load both models
    checkpoint_model = load_checkpoint_model()
    saved_model = load_saved_model()
    
    if not checkpoint_model or not saved_model:
        print("Failed to load one or both models. Cannot compare.")
        return
    
    # Find test images
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        print(f"Test directory '{test_dir}' not found!")
        return
    
    test_images = glob.glob(f"{test_dir}/*.jpg") + glob.glob(f"{test_dir}/*.png")
    if not test_images:
        print("No test images found!")
        return
    
    # Limit to 5 images
    if len(test_images) > 5:
        print(f"Found {len(test_images)} images, limiting to first 5")
        test_images = test_images[:5]
    else:
        print(f"Found {len(test_images)} test images")
    
    # Process each image
    for i, img_path in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] Testing image: {os.path.basename(img_path)}")
        
        # Get expected class from filename
        expected_class = None
        for state in STATES:
            if state in os.path.basename(img_path).lower():
                expected_class = state
                break
        if expected_class:
            print(f"Expected class (from filename): {expected_class}")
        
        # Process image
        original_img, img_batch = process_image(img_path)
        
        # Get prediction from checkpoint model
        checkpoint_pred = checkpoint_model.predict(img_batch, verbose=0)[0]
        checkpoint_class = STATES[np.argmax(checkpoint_pred)]
        
        # Get prediction from SavedModel
        infer = saved_model.signatures["serving_default"]
        saved_model_output = infer(tf.convert_to_tensor(img_batch, dtype=tf.float32))
        output_key = list(saved_model_output.keys())[0]
        saved_model_pred = saved_model_output[output_key].numpy()[0]
        saved_model_class = STATES[np.argmax(saved_model_pred)]
        
        # Compare predictions
        print(f"\nCheckpoint model prediction: {checkpoint_class}")
        print("Checkpoint probabilities:")
        for j, state in enumerate(STATES):
            print(f"  {state}: {checkpoint_pred[j]:.4f}")
        
        print(f"\nSavedModel prediction: {saved_model_class}")
        print("SavedModel probabilities:")
        for j, state in enumerate(STATES):
            print(f"  {state}: {saved_model_pred[j]:.4f}")
        
        # Check if predictions match
        if checkpoint_class != saved_model_class:
            print("\n⚠️ WARNING: Models gave different predictions!")
        
        # Visualize
        plt.figure(figsize=(12, 8))
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(original_img)
        plt.title(f"Traffic Light - {os.path.basename(img_path)}")
        plt.axis('off')
        
        # Checkpoint prediction
        plt.subplot(2, 2, 3)
        colors = ['green', 'red', 'yellow', 'gray']
        plt.barh(STATES, checkpoint_pred, color=colors)
        plt.xlim(0, 1)
        title = f"Checkpoint: {checkpoint_class}"
        if expected_class and expected_class != checkpoint_class:
            title += f" (Expected: {expected_class} ⚠️)"
        plt.title(title)
        
        # SavedModel prediction
        plt.subplot(2, 2, 4)
        plt.barh(STATES, saved_model_pred, color=colors)
        plt.xlim(0, 1)
        title = f"SavedModel: {saved_model_class}"
        if expected_class and expected_class != saved_model_class:
            title += f" (Expected: {expected_class})"
        plt.title(title)
        
        plt.tight_layout()
        plt.show()
    
    print("\n==== COMPARISON COMPLETE ====")

if __name__ == "__main__":
    compare_models()