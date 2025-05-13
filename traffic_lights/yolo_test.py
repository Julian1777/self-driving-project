from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import time
import glob

def run_traffic_light_detection():
    # Paths
    MODEL_PATH = "best_model.pt"
    TEST_IMAGES_DIR = "yolo_test_images"
    
    # Load the model
    print(f"Loading YOLOv8 model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get list of test images with multiple extensions and case-insensitive
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        test_images.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext)))
    
    print(f"Found {len(test_images)} test images")
    
    # Debug: print the files in the directory
    if not test_images:
        print("\nNo test images found. Listing directory contents:")
        if os.path.exists(TEST_IMAGES_DIR):
            all_files = os.listdir(TEST_IMAGES_DIR)
            for file in all_files:
                print(f"  {file}")
        else:
            print(f"Directory '{TEST_IMAGES_DIR}' does not exist!")
            print(f"Current working directory: {os.getcwd()}")
            return
    
    # Metrics
    class_counter = Counter()
    confidence_scores = []
    processing_times = []
    
    # Process test images
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        print(f"\nProcessing {img_name}...")
        
        # Run inference
        start_time = time.time()
        results = model(img_path, conf=0.25)  # Confidence threshold 0.25
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # ms
        processing_times.append(processing_time)
        
        # Extract results for the current image
        result = results[0]
        
        # Count classes and print detections for this image
        print(f"Detections in {img_name}:")
        
        if len(result.boxes) == 0:
            print("  No traffic lights detected")
        
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 10))
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        
        # Draw bounding boxes
        for i, box in enumerate(result.boxes):
            cls = int(box.cls.item())
            class_name = result.names[cls]
            confidence = box.conf.item()
            
            # Add to metrics
            class_counter[class_name] += 1
            confidence_scores.append(confidence)
            
            # Print detection details
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  {i+1}. {class_name.upper()} traffic light (conf: {confidence:.2f}) at coordinates: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
            
            # Different colors for different classes
            if class_name == 'red':
                color = (1, 0, 0)  # Red
            elif class_name == 'yellow':
                color = (1, 1, 0)  # Yellow
            elif class_name == 'green':
                color = (0, 1, 0)  # Green
            else:
                color = (0, 0, 1)  # Blue
                
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor=color, linewidth=3)
            ax.add_patch(rect)
            
            # Add label with larger font and better visibility
            text_box = ax.text(x1, y1-10, f"{class_name.upper()} {confidence:.2f}", 
                  color='black', fontsize=14, fontweight='bold',
                  bbox=dict(facecolor=color, alpha=0.7, boxstyle='round'))
        
        ax.set_title(f"Traffic Light Detection: {img_name}", fontsize=16)
        plt.tight_layout()
        
        # Show the image with detections
        plt.show()
        
        # Print processing time for this image
        print(f"  Processing time: {processing_time:.2f} ms")
    
    # Display overall metrics
    if test_images:
        print("\n===== DETECTION METRICS =====")
        print(f"Total test images processed: {len(test_images)}")
        print(f"Average processing time: {np.mean(processing_times):.2f} ms")
        
        print("\nDetections by class:")
        for class_name, count in class_counter.items():
            print(f"  {class_name.upper()}: {count}")
        
        if confidence_scores:
            print(f"\nConfidence scores:")
            print(f"  Average: {np.mean(confidence_scores):.3f}")
            print(f"  Min: {min(confidence_scores):.3f}")
            print(f"  Max: {max(confidence_scores):.3f}")

if __name__ == "__main__":
    run_traffic_light_detection()