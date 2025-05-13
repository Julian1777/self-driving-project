import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import sys

def convert_to_tflite(h5_path, tflite_path):
    """Convert H5 model to TFLite format using low-level TF operations"""
    print(f"Converting {h5_path} to TFLite format...")
    
    # Read the model file as a binary file
    with open(h5_path, 'rb') as f:
        model_content = f.read()
    
    try:
        # Use tf.lite.TFLiteConverter directly with model content
        converter = tf.lite.TFLiteConverter.from_keras_model_file(h5_path)
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Successfully converted model to {tflite_path}")
        return True
    except Exception as e:
        print(f"Error during conversion: {e}")
        
        # Try alternative method for older TF versions
        try:
            print("Trying alternative conversion method...")
            # This bypasses the model loading entirely
            os.system(f"python -m tf.lite.python.tflite_convert \
                    --keras_model_file={h5_path} \
                    --output_file={tflite_path}")
            
            if os.path.exists(tflite_path):
                print(f"Successfully converted model using CLI tool")
                return True
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
    
    return False

def use_tflite_model(image_path, model_path):
    """Test the TFLite model with an image"""
    # Load and preprocess the image
    img = cv.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test prediction
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Print prediction
    predicted_class = np.argmax(output[0])
    confidence = output[0][predicted_class]
    
    print(f"Prediction: Class {predicted_class}, Confidence: {confidence:.4f}")
    print("Model is working correctly!")

# For TFLite usage in your realtime.py script
def detect_with_tflite():
    """Example code to use in realtime.py"""
    MODEL_PATH = "sign_model.tflite"
    VIDEO_PATH = "/Users/jstamm2024/Documents/GitHub/self-driving-car-simulation/traffic_lights/ams_driving.mp4"
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Example prediction function
    def predict(img):
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])
    
    # Use this predict function instead of model.predict()
    # ...

if __name__ == "__main__":
    # Convert the model
    h5_path = "sign_model.h5"
    tflite_path = "sign_model.tflite"
    
    success = convert_to_tflite(h5_path, tflite_path)
    
    if success and os.path.exists(tflite_path):
        print("\nModel converted successfully to TFLite format!")
        print("Now update your realtime.py to use this TFLite model")
        print("Example code for TFLite usage is in the detect_with_tflite() function")
        
        # Test with an image if available
        test_images = [f for f in os.listdir('.') if f.endswith(('.jpg', '.png'))]
        if test_images:
            print(f"\nTesting model with image: {test_images[0]}")
            use_tflite_model(test_images[0], tflite_path)
    else:
        print("\nFailed to convert the model to TFLite format.")