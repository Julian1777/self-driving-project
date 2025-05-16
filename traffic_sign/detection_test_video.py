import cv2
import numpy as np
import time
from ultralytics import YOLO
import random

# Load the YOLOv8 model
MODEL_PATH = "./traffic sign detection yolo/best_model.pt"
VIDEO_PATH = "ams_driving_cropped.mp4"

def process_video():
    # Load the YOLOv8 model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Open the video file
    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video {VIDEO_PATH}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Optionally, you can resize the display window to a custom size:
        window_width, window_height = 920, 300
        cv2.namedWindow('Traffic Sign Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Traffic Sign Detection', window_width, window_height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video loaded: {width}x{height} at {fps}fps")
        
    except Exception as e:
        print(f"Error opening video: {e}")
        return
    
    # Create random colors for different classes
    class_colors = {}
    
    # Start processing
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
            
        frame_count += 1
        
        # Run detection on the frame
        results = model(frame, conf=0.25, iou=0.45)
        
        # Process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get confidence score
                conf = float(box.conf[0])
                
                # Get class ID
                cls_id = int(box.cls[0])
                
                # Get or create a color for this class
                if cls_id not in class_colors:
                    # Generate random color
                    class_colors[cls_id] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                
                color = class_colors[cls_id]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f"Sign: {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    frame, 
                    (x1, y1 - text_size[1] - 5), 
                    (x1 + text_size[0], y1), 
                    color, 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    2
                )
        
        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps_text = f"FPS: {frame_count / elapsed_time:.1f}"
        cv2.putText(
            frame, 
            fps_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Display the frame
        cv2.imshow('Traffic Sign Detection', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"Average FPS: {frame_count / elapsed_time:.2f}")

if __name__ == "__main__":
    process_video()