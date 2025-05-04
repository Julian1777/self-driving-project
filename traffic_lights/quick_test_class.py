import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Lambda, Concatenate
from tensorflow.keras.models import Model

# Define constants
IMG_SIZE = (64, 64)
STATES = ["green", "red", "yellow", "off"]
CHECKPOINT_PATH = "traffic_light_classification_weights.h5"

def create_model():
    """Create the model architecture that exactly matches your checkpoint"""
    # Input layer
    inputs = Input(shape=(*IMG_SIZE, 3))
    
    # Data augmentation would have been here but we can skip it for inference
    
    # Create base model with EfficientNetB0
    base_model = EfficientNetB0(include_top=False, input_tensor=inputs, weights=None)
    
    # Add global pooling
    x = GlobalAveragePooling2D()(base_model.output)
    
    # Add dense layer with dropout (matching your training model)
    cnn_features = Dense(128, activation='relu')(x)
    cnn_features = Dropout(0.5)(cnn_features)
    
    # Create a dummy HSV feature extraction layer that matches your model
    # This is a placeholder and won't actually perform the extraction
    hsv_features = Lambda(lambda x: tf.zeros([tf.shape(x)[0], 28]))(inputs)
    hsv_branch = Dense(32, activation='relu')(hsv_features)
    
    # Combine features
    combined = Concatenate()([cnn_features, hsv_branch])
    
    # Final classification layer
    outputs = Dense(len(STATES), activation='softmax')(combined)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def process_image(image_path):
    """Process an image for prediction"""
    # Read image
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv.resize(img, IMG_SIZE)
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, 0)
    
    return img, img_batch

def predict_light(image_path):
    """Predict traffic light state from image"""
    # Create model
    model = create_model()
    
    # Load weights
    print(f"Loading weights from {CHECKPOINT_PATH}")
    model.load_weights(CHECKPOINT_PATH)
    
    # Process image
    original, processed = process_image(image_path)
    
    # Make prediction
    prediction = model.predict(processed, verbose=0)[0]
    
    # Get class with highest probability
    class_idx = np.argmax(prediction)
    class_name = STATES[class_idx]
    confidence = prediction[class_idx]
    
    # Display results
    plt.figure(figsize=(8, 4))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')
    
    # Display prediction
    plt.subplot(1, 2, 2)
    plt.imshow(cv.resize(original, (200, 200)))
    plt.title(f"Prediction: {class_name}")
    plt.xlabel(f"Confidence: {confidence:.2f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        "state": class_name,
        "confidence": float(confidence),
        "probabilities": {state: float(prob) for state, prob in zip(STATES, prediction)}
    }

# Update the main execution section at the bottom of the file

if __name__ == "__main__":
    # Debug output for class labels
    print("Using the following class labels:")
    for i, state in enumerate(STATES):
        print(f"  Index {i}: {state}")
    print()
    
    # Test with a known image
    test_image = "traffic_light2_go.jpg"
    
    if os.path.exists(test_image):
        print(f"Using test image: {test_image}")
        try:
            result = predict_light(test_image)
            print(f"\nPrediction: {result['state']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            print("\nAll class probabilities:")
            for state, prob in result['probabilities'].items():
                print(f"  {state}: {prob:.4f}")
                
            # Check for potential mismatch
            if result['confidence'] > 0.9:
                print("\nWARNING: High confidence prediction. If incorrect, class order might be wrong.")
                print("Try these alternative class orderings:")
                
                # Alternative 1: LISA dataset order
                alt1 = ["red", "green", "yellow", "off"]
                print("\nAlternative 1:", alt1)
                alt1_idx = np.argmax(result['probabilities'].values())
                print(f"With this ordering, prediction would be: {alt1[alt1_idx]}")
                
                # Alternative 2: Another common ordering
                alt2 = ["off", "red", "yellow", "green"]
                print("\nAlternative 2:", alt2)
                alt2_idx = np.argmax(result['probabilities'].values())
                print(f"With this ordering, prediction would be: {alt2[alt2_idx]}")
        except Exception as e:
            print(f"Error during prediction: {e}")
    else:
        print(f"Test image '{test_image}' not found.")
        
        # Try to find any traffic light images
        possible_dirs = ["./dataset", "./cropped_dataset", "./merged_dataset"]
        for dir in possible_dirs:
            if os.path.exists(dir):
                for state in STATES:
                    state_dir = os.path.join(dir, state)
                    if os.path.exists(state_dir):
                        files = [f for f in os.listdir(state_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if files:
                            sample = os.path.join(state_dir, files[0])
                            print(f"\nFound alternative test image: {sample}")
                            print(f"True class should be: {state}")
                            try:
                                result = predict_light(sample)
                                print(f"Prediction: {result['state']} (Confidence: {result['confidence']:.2f})")
                                print("This will help verify if the class ordering is correct.")
                            except Exception as e:
                                print(f"Error testing with alternative image: {e}")
                            break
                    break
                break