import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import tensorflow as tf
import keras

def load_class_names(csv_path):
    df = pd.read_csv(csv_path)
    class_names = {}
    for _, row in df.iterrows():
        class_id = row['id']
        name = row['description']
        class_names[str(class_id)] = name
    return class_names

def get_ordered_descriptions(class_names):
    ds_path = os.path.join("dataset", "Train")
    if os.path.exists(ds_path):
        print(f"Loading class ordering from dataset at {ds_path}")
        train_ds = keras.preprocessing.image_dataset_from_directory(
            ds_path,
            batch_size=1,
            image_size=(224, 224),
            shuffle=False
        )
        original_class_names = train_ds.class_names
        
        ordered_descriptions = []
        for dir_name in original_class_names:
            description = class_names.get(dir_name, f"Class {dir_name}")
            ordered_descriptions.append(description)
            
        print(f"Loaded {len(ordered_descriptions)} class descriptions")
        return ordered_descriptions
    else:
        print(f"Warning: Training dataset directory {ds_path} not found")
        max_id = max([int(k) for k in class_names.keys() if k.isdigit()], default=42)
        ordered_descriptions = []
        for i in range(max_id + 1):
            description = class_names.get(str(i), f"Class {i}")
            ordered_descriptions.append(description)
        return ordered_descriptions

def preprocess_frame(frame):
    """
    Enhanced preprocessing to help model focus on sign features
    """
    # Basic resizing and color conversion
    processed = cv2.resize(frame, (224, 224))
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Optional: Enhance contrast to make signs more visible
    # lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # cl = clahe.apply(l)
    # enhanced_lab = cv2.merge((cl, a, b))
    # processed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # DO NOT normalize here - the inference model has a built-in normalization layer
    processed = processed.astype(np.float32)
    processed = np.expand_dims(processed, axis=0)
    return processed

def debug_predictions(predictions, ordered_descriptions, frame_num):
    """Helpful debug function to see prediction breakdown"""
    # Get top 5 predictions
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_confidences = predictions[top_indices]
    
    print(f"\nFrame {frame_num} - Top 5 predictions:")
    for i, (idx, conf) in enumerate(zip(top_indices, top_confidences)):
        sign_name = ordered_descriptions[idx] if idx < len(ordered_descriptions) else f"Class {idx}"
        print(f"  {i+1}. {sign_name}: {conf:.4f}")
    
    # Check problematic classes we've identified
    biased_classes = ["danger", "60 speed limit", "trucks prohibited"]
    for class_name in biased_classes:
        try:
            idx = ordered_descriptions.index(class_name)
            conf = predictions[idx]
            print(f"  '{class_name}' confidence: {conf:.4f} (Rank: {np.where(np.argsort(predictions)[::-1] == idx)[0][0]+1})")
        except ValueError:
            # Try partial matching
            matches = [i for i, name in enumerate(ordered_descriptions) if class_name.lower() in name.lower()]
            if matches:
                idx = matches[0]
                conf = predictions[idx]
                print(f"  '{ordered_descriptions[idx]}' confidence: {conf:.4f} (Rank: {np.where(np.argsort(predictions)[::-1] == idx)[0][0]+1})")
    
    print(f"  Prediction entropy: {-np.sum(predictions * np.log(predictions + 1e-10)):.4f}")

def process_realtime(video_path, model, ordered_descriptions, confidence_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = "output_" + os.path.basename(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create a logging panel to show multiple predictions
    panel_height = 180  # Height of the info panel in pixels
    
    print("Starting real-time processing with navigation controls:")
    print("  - Press 'q' to quit")
    print("  - Press → or 'n' to go forward 1 frame")
    print("  - Press → + Shift to go forward 10 frames")
    print("  - Press ← or 'p' to go backward 1 frame")
    print("  - Press ← + Shift to go backward 10 frames")
    print("  - Press Space to pause/resume playback")
    
    frame_count = 0
    last_debug_frame = 0
    
    # Track sign detections over time for temporal smoothing
    detection_history = []
    history_size = 5
    
    # Define known problematic classes for more aggressive correction
    biased_class_names = ["danger", "60 speed limit", "trucks prohibited", "No trucks"]
    biased_class_indices = []
    for name in biased_class_names:
        try:
            idx = ordered_descriptions.index(name)
            biased_class_indices.append(idx)
        except ValueError:
            # Try partial matching
            matches = [i for i, desc in enumerate(ordered_descriptions) if name.lower() in desc.lower()]
            if matches:
                biased_class_indices.extend(matches)
    
    print(f"Applying bias correction to these class indices: {biased_class_indices}")
    
    # Add playback control variables
    is_paused = False
    
    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                # End of video
                print("End of video reached")
                break
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        display_frame = frame.copy()
        
        # Process and predict
        processed_frame = preprocess_frame(frame)
        predictions = model.predict(processed_frame, verbose=0)[0]
        
        # Debug periodically
        if frame_count - last_debug_frame >= 30:  # Debug every 30 frames (about 1 second)
            debug_predictions(predictions, ordered_descriptions, frame_count)
            last_debug_frame = frame_count
        
        # Apply global bias correction for known problematic classes
        adjusted_predictions = predictions.copy()
        for idx in biased_class_indices:
            # Apply a 50% penalty to known biased classes
            adjusted_predictions[idx] *= 0.5
        
        # Renormalize
        adjusted_predictions /= np.sum(adjusted_predictions)
        
        # Get top predictions after bias correction
        top_indices = np.argsort(adjusted_predictions)[-5:][::-1]
        top_confidences = adjusted_predictions[top_indices]
        
        # Create info panel
        info_panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        # Add a black background for main text
        cv2.rectangle(display_frame, (0, 0), (width, 40), (0, 0, 0), -1)
        
        # Add frame counter to display
        frame_info = f"Frame: {frame_count+1}/{total_frames} ({'PAUSED' if is_paused else 'Playing'})"
        cv2.putText(display_frame, frame_info, (width - 320, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Add visualization of all class confidences as a bar
        vis_width = width
        for i, p in enumerate(adjusted_predictions):
            bar_x = int(i * vis_width / len(adjusted_predictions))
            bar_width = max(1, int(vis_width / len(adjusted_predictions)))
            bar_height = int(p * 50)  # Scale for visibility
            
            # Use different colors for biased classes
            color = (0, 0, 200) if i in biased_class_indices else (100, 100, 100)
            if p > confidence_threshold:
                color = (0, 255, 0)
                
            cv2.rectangle(info_panel, 
                          (bar_x, panel_height - 10 - bar_height), 
                          (bar_x + bar_width, panel_height - 10), 
                          color, -1)
        
        # ---- BIAS MITIGATION: Apply temporal smoothing ----
        # Store current frame's top prediction
        if top_confidences[0] > confidence_threshold:
            detection_history.append((top_indices[0], top_confidences[0]))
        else:
            detection_history.append((None, 0))
            
        # Keep only the most recent predictions
        if len(detection_history) > history_size:
            detection_history = detection_history[-history_size:]
            
        # Count occurrences of each class in recent history
        class_counts = {}
        for idx, conf in detection_history:
            if idx is not None:
                class_counts[idx] = class_counts.get(idx, 0) + 1
                
        # ---- BIAS MITIGATION: Penalize overly frequent predictions ----
        # Find the most common class in recent history
        most_common_class = None
        most_common_count = 0
        for idx, count in class_counts.items():
            if count > most_common_count:
                most_common_class = idx
                most_common_count = count
                
        # Apply penalty to predictions that appear too consistently
        if most_common_class is not None and most_common_count > history_size * 0.6:
            # If a class appears in >60% of recent frames, penalize it more
            
            # Extra penalty for biased classes
            if most_common_class in biased_class_indices:
                adjusted_predictions[most_common_class] *= 0.3  # 70% penalty
            else:
                adjusted_predictions[most_common_class] *= 0.7  # 30% penalty
                
            # Renormalize
            adjusted_predictions /= np.sum(adjusted_predictions)
            
            # Recalculate top indices with adjusted predictions
            top_indices = np.argsort(adjusted_predictions)[-5:][::-1]
            top_confidences = adjusted_predictions[top_indices]
        
        # Display top predictions
        has_detection = False
        for i, (idx, conf) in enumerate(zip(top_indices, top_confidences)):
            sign_name = ordered_descriptions[idx] if idx < len(ordered_descriptions) else f"Class {idx}"
            
            # Add to info panel
            y_pos = 25 + (i * 25)  # Increased spacing
            
            # Highlight biased classes
            if idx in biased_class_indices:
                text_color = (0, 0, 255)  # Red for biased classes
                sign_name = f"{sign_name} (BIAS CORRECTED)"
            else:
                text_color = (0, 255, 0) if i == 0 else (200, 200, 200)
                
            cv2.putText(info_panel, f"{i+1}. {sign_name}: {conf:.2f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            
            # Main detection on video frame
            if i == 0 and conf > confidence_threshold and idx not in biased_class_indices:
                has_detection = True
                text = f"Sign: {sign_name} ({conf:.2f})"
                cv2.putText(display_frame, text, (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if not has_detection:
            cv2.putText(display_frame, "No reliable traffic signs detected", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Combine frame and info panel
        combined_frame = np.vstack([display_frame, info_panel])
        
        # Display and save (just the original frame without info panel)
        cv2.imshow('Traffic Sign Detection', combined_frame)
        
        # Only save to output if not in paused mode
        if not is_paused:
            out.write(display_frame)
        
        # Handle keyboard controls
        key = cv2.waitKey(0 if is_paused else 1) & 0xFF
        
        if key == ord('q'):
            # Quit
            break
        elif key == ord(' '):
            # Toggle pause
            is_paused = not is_paused
            print(f"Playback {'paused' if is_paused else 'resumed'} at frame {frame_count+1}")
        elif key == ord('n') or key == 83:  # 'n' or right arrow
            # Next frame
            if is_paused:
                next_frame = min(frame_count + 1, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                ret, frame = cap.read()
                if ret:
                    frame_count = next_frame
                    print(f"Advanced to frame {frame_count+1}")
        elif key == ord('N') or key == 85:  # 'N' or shift+right arrow
            # Jump forward 10 frames
            if is_paused:
                next_frame = min(frame_count + 10, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                ret, frame = cap.read()
                if ret:
                    frame_count = next_frame
                    print(f"Advanced to frame {frame_count+1}")
        elif key == ord('p') or key == 81:  # 'p' or left arrow
            # Previous frame
            if is_paused:
                prev_frame = max(frame_count - 1, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
                ret, frame = cap.read()
                if ret:
                    frame_count = prev_frame
                    print(f"Went back to frame {frame_count+1}")
        elif key == ord('P') or key == 86:  # 'P' or shift+left arrow
            # Jump backward 10 frames
            if is_paused:
                prev_frame = max(frame_count - 10, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
                ret, frame = cap.read()
                if ret:
                    frame_count = prev_frame
                    print(f"Went back to frame {frame_count+1}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processed video saved to {output_path}")

def main():
    video_path = "ams_driving_cropped.mp4"
    model_path = "sign_inference.h5"
    csv_path = "sign_dic.csv"
    
    # Check files exist
    for path, desc in [(video_path, "Video"), (model_path, "Model"), (csv_path, "CSV")]:
        if not os.path.exists(path):
            print(f"Error: {desc} file not found: {path}")
            if path == model_path:
                alt_path = "sign_model.h5"
                if os.path.exists(alt_path):
                    print(f"Found alternative model at {alt_path}")
                    model_path = alt_path
                else:
                    return
            else:
                return
    
    # Load class names
    print("Loading class descriptions...")
    class_names = load_class_names(csv_path)
    print(f"Loaded {len(class_names)} class descriptions")
    
    # Get ordered descriptions matching the trained model
    ordered_descriptions = get_ordered_descriptions(class_names)
    
    # Print first few class mappings for verification
    print("\nClass mapping (first 10):")
    for i, desc in enumerate(ordered_descriptions[:10]):
        print(f"  {i}: {desc}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # Process video with higher confidence threshold
    process_realtime(video_path, model, ordered_descriptions, confidence_threshold=0.5)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()