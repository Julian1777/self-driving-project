import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import random
import argparse
from pathlib import Path
import shutil

# Constants
IMG_SIZE = (64, 64)
STATES = ["red", "yellow", "green"]
COLORS = ['red', 'yellow', 'green']
MERGED_DATASET_PATH = r"C:\Users\user\Documents\github\self-driving-car-simulation\traffic_lights\merged_dataset"
MISCLASSIFIED_DIR = r"C:\Users\user\Documents\github\self-driving-car-simulation\traffic_lights\misclassified_images"
MODEL_PATH = r"C:\Users\user\Documents\github\self-driving-car-simulation\traffic_lights\traffic_light_classification_savedmodel"
CROP_TOP_BOTTOM = 3
CROP_LEFT_RIGHT = 12

def load_image(image_path):
    """Load and preprocess an image for the model"""
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert to RGB
    img_resized = cv.resize(img, IMG_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0  # Normalize to [0,1]
    return img, img_normalized

def analyze_brightness_pattern(image):
    """Direct brightness-based classification"""
    # Convert to HSV
    hsv = cv.cvtColor((image * 255).astype(np.uint8), cv.COLOR_RGB2HSV)
    
    # Get cropped V channel
    height, width = image.shape[:2]
    cropped_v = hsv[CROP_TOP_BOTTOM:height-CROP_TOP_BOTTOM, CROP_LEFT_RIGHT:width-CROP_LEFT_RIGHT, 2]
    
    # Get average brightness per row
    row_brightness = np.mean(cropped_v, axis=1)
    
    # Split into three vertical sections
    section_size = len(row_brightness) // 3
    top_section = np.mean(row_brightness[:section_size])
    middle_section = np.mean(row_brightness[section_size:2*section_size])
    bottom_section = np.mean(row_brightness[2*section_size:])
    
    # Determine brightest section
    sections = [top_section, middle_section, bottom_section]
    brightest_section = np.argmax(sections)
    
    return STATES[brightest_section], sections

def predict_traffic_light(model, image):
    """Make a prediction using the SavedModel"""
    # Print image statistics
    print(f"Image shape: {image.shape}, min: {np.min(image):.4f}, max: {np.max(image):.4f}")
    
    # Show preprocessed image for visual inspection
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.title("Preprocessed input")
    plt.show()
    
    # Direct brightness analysis for comparison
    brightness_state, sections = analyze_brightness_pattern(image)
    print(f"Direct brightness analysis: {brightness_state}")
    print(f"Section values: top={sections[0]:.2f}, middle={sections[1]:.2f}, bottom={sections[2]:.2f}")
    
    # Model prediction
    img_batch = np.expand_dims(image, 0)
    
    # Get prediction from SavedModel
    infer = model.signatures["serving_default"]
    output = infer(tf.convert_to_tensor(img_batch, dtype=tf.float32))
    output_key = list(output.keys())[0]
    predictions = output[output_key].numpy()[0]
    
    # Print raw prediction values
    print(f"Raw predictions: {predictions}")
    print(f"Sum of predictions: {np.sum(predictions):.4f}")
    
    return predictions, brightness_state

def test_random_images(num_images=10, visualize=True):
    """Test the model on random images from the merged dataset"""
    # Create misclassified directory if it doesn't exist
    os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)
    
    # Check if merged dataset exists
    if not os.path.exists(MERGED_DATASET_PATH):
        print(f"Error: Dataset path {MERGED_DATASET_PATH} not found.")
        return
    
    # Load the saved model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.saved_model.load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get all image paths organized by state
    image_paths_by_state = {}
    for state in STATES:
        state_dir = os.path.join(MERGED_DATASET_PATH, state)
        if os.path.exists(state_dir):
            image_paths = [os.path.join(state_dir, f) for f in os.listdir(state_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_paths_by_state[state] = image_paths
            print(f"Found {len(image_paths)} images for state '{state}'")
    
    # Sample images from each state
    per_state = max(1, num_images // len(STATES))
    selected_images = []
    
    for state, paths in image_paths_by_state.items():
        state_samples = min(per_state, len(paths))
        selected_images.extend([(path, state) for path in random.sample(paths, state_samples)])
    
    # Process selected images
    correct_model = 0
    correct_brightness = 0
    total = len(selected_images)
    
    # For batch plotting
    if visualize:
        cols = min(4, total)
        rows = (total + cols - 1) // cols
        plt.figure(figsize=(cols * 4, rows * 4))
    
    for i, (img_path, true_state) in enumerate(selected_images):
        img_name = os.path.basename(img_path)
        print(f"\nTesting image {i+1}/{total}: {img_name}")
        print(f"True state: {true_state}")
        
        # Load and preprocess image
        original_img, preprocessed_img = load_image(img_path)
        
        # Get model prediction and brightness analysis
        predictions, brightness_state = predict_traffic_light(model, preprocessed_img)
        predicted_class = STATES[np.argmax(predictions)]
        confidence = np.max(predictions)
        
        print(f"Model prediction: {predicted_class} with confidence {confidence:.4f}")
        print(f"Brightness analysis: {brightness_state}")
        
        # Check if predictions are correct
        is_correct_model = predicted_class == true_state
        is_correct_brightness = brightness_state == true_state
        
        if is_correct_model:
            correct_model += 1
            print("Model prediction correct! ✓")
        else:
            print("Model prediction incorrect! ✗")
            
            # Save misclassified image
            misclassified_path = os.path.join(MISCLASSIFIED_DIR, 
                                             f"{true_state}_model_pred_{predicted_class}_{confidence:.2f}_{img_name}")
            cv.imwrite(misclassified_path, cv.cvtColor(original_img, cv.COLOR_RGB2BGR))
        
        if is_correct_brightness:
            correct_brightness += 1
            print("Brightness analysis correct! ✓")
        else:
            print("Brightness analysis incorrect! ✗")
        
        # Visualize the prediction
        if visualize:
            plt.subplot(rows, cols, i+1)
            
            # Original image
            plt.imshow(original_img)
            
            # Create title with both predictions
            title = f"True: {true_state}\nModel: {predicted_class} ({confidence:.2f})\nBright: {brightness_state}"
            color = 'green' if is_correct_model else 'red'
            plt.title(title, color=color, fontsize=9)
            
            # Bar chart overlay for model confidences
            ax2 = plt.gca().inset_axes([0.05, 0.05, 0.9, 0.2])
            y_pos = np.arange(len(STATES))
            ax2.barh(y_pos, predictions, color=COLORS)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(STATES)
            ax2.set_xlim(0, 1)
            plt.setp(ax2.get_xticklabels(), fontsize=8)
            plt.setp(ax2.get_yticklabels(), fontsize=8)
            
            plt.axis('off')
    
    # Show the visualization
    if visualize and total > 0:
        plt.tight_layout()
        plt.show()
    
    # Print accuracy results
    model_accuracy = correct_model / total if total > 0 else 0
    brightness_accuracy = correct_brightness / total if total > 0 else 0
    print(f"\nTest Results:")
    print(f"Model accuracy: {correct_model}/{total} correct, {model_accuracy:.2%}")
    print(f"Brightness analysis accuracy: {correct_brightness}/{total} correct, {brightness_accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test traffic light classification model on random images')
    parser.add_argument('--num', type=int, default=12, help='Number of random images to test')
    parser.add_argument('--no-plot', action='store_true', help='Disable visualization')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to SavedModel directory')
    
    args = parser.parse_args()
    if args.model:
        MODEL_PATH = args.model
    test_random_images(num_images=args.num, visualize=not args.no_plot)