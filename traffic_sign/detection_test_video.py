import cv2
import numpy as np
import time
from ultralytics import YOLO
import random

MODEL_PATH = "./traffic sign detection yolo/sign_detection.pt"
VIDEO_PATH = "ams_driving_cropped.mp4"

def process_video():
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
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
        window_width, window_height = 920, 300
        cv2.namedWindow('Traffic Sign Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Traffic Sign Detection', window_width, window_height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video loaded: {width}x{height} at {fps}fps")
        
    except Exception as e:
        print(f"Error opening video: {e}")
        return
    
    class_colors = {}
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
            
        frame_count += 1
        
        results = model(frame, conf=0.25, iou=0.45)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                conf = float(box.conf[0])
                
                cls_id = int(box.cls[0])
                
                if cls_id not in class_colors:
                    class_colors[cls_id] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                
                color = class_colors[cls_id]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"Sign: {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
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
                    0.5, 
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
        
        cv2.imshow('Traffic Sign Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"Average FPS: {frame_count / elapsed_time:.2f}")

if __name__ == "__main__":
    process_video()