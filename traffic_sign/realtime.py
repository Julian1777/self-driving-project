import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import os
import time

def load_class_names(csv_path):
    try:
        df = pd.read_csv(csv_path)
        class_names = {}
        for _, row in df.iterrows():
            class_id = row['id']
            name = row['description']
            class_names[str(class_id)] = name
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
        return {}

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_ordered_descriptions(class_names, ds_path):
    if os.path.exists(ds_path):
        class_dirs = sorted([d for d in os.listdir(ds_path) 
                            if os.path.isdir(os.path.join(ds_path, d))])
        ordered_descriptions = []
        for dir_name in class_dirs:
            description = class_names.get(dir_name, f"Class {dir_name}")
            ordered_descriptions.append(description)
        return ordered_descriptions, class_dirs
    else:
        print(f"Warning: Dataset path {ds_path} not found")
        return list(class_names.values()), list(class_names.keys())

def main():
    MODEL_PATH = "sign_inference.h5"
    CSV_PATH = "sign_dic.csv"
    DS_PATH = os.path.join("dataset", "Train")
    CONFIDENCE_THRESHOLD = 0.6
    
    VIDEO_SOURCE = "ams_driving_cropped.mp4"
    
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = load_model("sign_model.h5")
        print("Loaded full model as fallback")
    
    class_names = load_class_names(CSV_PATH)
    ordered_descriptions, class_dirs = get_ordered_descriptions(class_names, DS_PATH)
    print(f"Loaded {len(ordered_descriptions)} class descriptions")
    
    print(f"Opening video file: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_SOURCE}")
        print("Trying alternative path...")
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), VIDEO_SOURCE)
        cap = cv2.VideoCapture(alt_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {alt_path} either")
            return
        else:
            print(f"Successfully opened video at {alt_path}")
    else:
        print("Video file opened successfully")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    display_width = 1024
    display_height = int(height * (display_width / width))
    print(f"Display size: {display_width}x{display_height}")
    
    if total_frames <= 0:
        total_frames = float('inf')
        print("Warning: Could not determine total frames")
    else:
        print(f"Video loaded: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    frame_position = 0
    frame_skip = 0
    
    print("\nControls:")
    print("  ESC or Q: Quit")
    print("  RIGHT ARROW or D: Skip forward 30 frames")
    print("  LEFT ARROW or A: Skip backward 30 frames")
    print("  SPACE: Pause/Resume")
    print("  + / -: Adjust confidence threshold")
    
    paused = False
    processing_times = []
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached or error reading frame")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_position = 0
                continue
            
            frame_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            if frame_skip > 0 and frame_position % (frame_skip + 1) != 0:
                continue
            
            start_time = time.time()
            
            processed = preprocess_frame(frame)
            predictions = model.predict(np.expand_dims(processed, axis=0), verbose=0)[0]
            
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_predictions = [(ordered_descriptions[i], predictions[i] * 100) for i in top_indices]
            
            best_idx = top_indices[0]
            best_confidence = predictions[best_idx] * 100
            best_class = ordered_descriptions[best_idx]
            
            process_time = (time.time() - start_time) * 1000
            processing_times.append(process_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            avg_time = np.mean(processing_times)
            
            display = cv2.resize(frame, (display_width, display_height))
            scale_x = display_width / width
            scale_y = display_height / height
            
            cv2.rectangle(display, (0, 0), (display_width, int(40 * scale_y)), (0, 0, 0), -1)
            position_text = f"Frame: {frame_position}/{total_frames}"
            cv2.putText(display, position_text, (10, int(30 * scale_y)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale_y, (255, 255, 255), 1)
            
            fps_text = f"Processing: {process_time:.1f}ms ({1000/avg_time:.1f} FPS)"
            text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale_y, 2)[0]
            cv2.putText(display, fps_text, (display_width - text_size[0] - 20, int(30 * scale_y)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale_y, (255, 255, 255), 1)
            
            if best_confidence >= CONFIDENCE_THRESHOLD:
                cv2.rectangle(display, (0, int(50 * scale_y)), 
                             (display_width, int(150 * scale_y)), (0, 0, 0), -1)
                
                if best_confidence > 85:
                    color = (0, 255, 0)
                elif best_confidence > 70:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)
                
                text = f"Detected: {best_class}"
                cv2.putText(display, text, (int(20 * scale_x), int(100 * scale_y)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale_y, color, 1)
                
                conf_text = f"{best_confidence:.1f}%"
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale_y, 3)[0]
                cv2.putText(display, conf_text, (display_width - text_size[0] - 20, int(100 * scale_y)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale_y, color, 1)
            
            box_height = int(130 * scale_y)
            cv2.rectangle(display, (0, display_height - box_height), 
                         (display_width, display_height), (0, 0, 0), -1)
            
            cv2.putText(display, "Top Predictions:", (int(20 * scale_x), display_height - box_height + int(30 * scale_y)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale_y, (255, 255, 255), 1)
            
            for i, (desc, conf) in enumerate(top_predictions):
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                
                text = f"{i+1}. {desc}"
                conf_text = f"{conf:.1f}%"
                
                if i == 0:
                    color = (0, 255, 0)
                elif i == 1:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)
                
                y_pos = display_height - box_height + int(70 * scale_y) + int(i * 30 * scale_y)
                cv2.putText(display, text, (int(40 * scale_x), y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale_y, color, 1)
                
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale_y, 2)[0]
                conf_x = display_width - text_size[0] - int(40 * scale_x)
                cv2.putText(display, conf_text, (conf_x, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale_y, color, 1)
        
        cv2.imshow("Traffic Sign Recognition", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27 or key == ord('q'):
            break
        elif key == 32:
            paused = not paused
            if paused:
                print("Paused - press SPACE to resume")
            else:
                print("Resumed")
        elif key == 83 or key == ord('d'):
            new_pos = min(frame_position + 30, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            paused = True
        elif key == 81 or key == ord('a'):
            new_pos = max(frame_position - 30, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            paused = True
        elif key == ord('+') or key == ord('='):
            CONFIDENCE_THRESHOLD = min(CONFIDENCE_THRESHOLD + 0.05, 0.95)
            print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
        elif key == ord('-'):
            CONFIDENCE_THRESHOLD = max(CONFIDENCE_THRESHOLD - 0.05, 0.05)
            print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete")

if __name__ == "__main__":
    main()