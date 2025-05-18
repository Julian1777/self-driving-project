import cv2
import numpy as np
import time
from ultralytics import YOLO
import random

MODEL_PATH = "vehicle_pedestrian_detection.pt"
VIDEO_PATH = "ams_driving_cropped.mp4"

def process_video():
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        
        print(f"Model classes: {model.names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video {VIDEO_PATH}")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        window_width, window_height = 1280, 720
        cv2.namedWindow('Vehicle & Pedestrian Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vehicle & Pedestrian Detection', window_width, window_height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video loaded: {width}x{height} at {fps}fps")
        
    except Exception as e:
        print(f"Error opening video: {e}")
        return
    
    class_colors = {
        'car': (0, 0, 255),       # Red
        'person': (0, 255, 0),    # Green
        'truck': (255, 0, 0),     # Blue
        'bus': (255, 255, 0),     # Yellow
        'traffic light': (255, 0, 255),  # Magenta
        'traffic sign': (0, 255, 255),   # Cyan
        'bicycle': (255, 165, 0), # Orange
        'motorcycle': (128, 0, 128)  # Purple
    }
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
            
        frame_count += 1
        
        results = model(frame, conf=0.30, iou=0.45)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                conf = float(box.conf[0])
                
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                if cls_id not in class_colors:
                    if cls_name in class_colors:
                        class_colors[cls_id] = class_colors[cls_name]
                    else:
                        class_colors[cls_id] = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255)
                        )
                
                color = class_colors[cls_id]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                label = f"{cls_name}: {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(
                    frame, 
                    (x1, y1 - text_size[1] - 5), 
                    (x1 + text_size[0], y1), 
                    color, 
                    -1
                )
                
                cv2.putText(
                    frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
        
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
        
        cv2.imshow('Vehicle & Pedestrian Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"Average FPS: {frame_count / elapsed_time:.2f}")

if __name__ == "__main__":
    process_video()