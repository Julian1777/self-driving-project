from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import time

def run_traffic_light_detection_video():
    # Paths
    MODEL_PATH = "best_model.pt"
    VIDEO_PATH = "ams_driving.mp4"
    
    # Recording settings (optional)
    RECORD_OUTPUT = True
    OUTPUT_PATH = "ams_driving_detected.mp4"
    
    # Navigation settings
    SKIP_FRAMES = 30  # Number of frames to skip on arrow keys
    
    # Load the model
    print(f"Loading YOLOv8 model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Video file '{VIDEO_PATH}' not found!")
        return
    
    # Open the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video file {VIDEO_PATH}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create output video writer if recording
    output_writer = None
    if RECORD_OUTPUT:
        output_writer = cv2.VideoWriter(
            OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
    
    # Metrics
    class_counter = Counter()
    confidence_scores = []
    processing_times = []
    frames_with_detections = 0
    
    # Process video frames
    frame_count = 0
    detection_count = 0
    
    # Navigation state
    paused = False
    skip_message = ""
    skip_message_time = 0
    
    print("\nStarting live detection. Controls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Press SPACE to pause/resume")
    print("  - Press RIGHT ARROW to skip forward")
    print("  - Press LEFT ARROW to skip backward")
    
    # Create window for display
    cv2.namedWindow("Traffic Light Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Light Detection", 1280, 720)  # Resize for better viewing
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Skip further processing if paused (but still handle key events)
        if not paused:
            # Run inference
            start_time = time.time()
            results = model(frame, conf=0.25)  # Confidence threshold 0.25
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # ms
            processing_times.append(processing_time)
            
            # Extract results for the current frame
            result = results[0]
            
            # Frame has detections?
            has_detections = len(result.boxes) > 0
            if has_detections:
                frames_with_detections += 1
                
            # Draw bounding boxes on frame
            annotated_frame = frame.copy()
            
            for box in result.boxes:
                cls = int(box.cls.item())
                class_name = result.names[cls]
                confidence = box.conf.item()
                
                # Add to metrics
                class_counter[class_name] += 1
                confidence_scores.append(confidence)
                detection_count += 1
                
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Different colors for different classes
                if class_name == 'red':
                    color = (0, 0, 255)  # BGR: Red
                elif class_name == 'yellow':
                    color = (0, 255, 255)  # BGR: Yellow
                elif class_name == 'green':
                    color = (0, 255, 0)  # BGR: Green
                else:
                    color = (255, 0, 0)  # BGR: Blue
                    
                # Draw rectangle
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw filled rectangle for text background
                text = f"{class_name.upper()} {confidence:.2f}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, (x1, y1-25), (x1+text_size[0], y1), color, -1)
                
                # Draw text
                cv2.putText(annotated_frame, text, (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add frame number and processing time
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Time: {frame_count/fps:.1f}s", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Processing: {processing_time:.1f}ms", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display pause status
            if paused:
                cv2.putText(annotated_frame, "PAUSED", 
                            (width-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Display skip message if active
            if skip_message and time.time() - skip_message_time < 2:  # Show for 2 seconds
                cv2.putText(annotated_frame, skip_message, 
                            (width//2-200, height-50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 165, 255), 2)
            
            # Display counters for each class
            y_offset = 120
            for cls_name, count in class_counter.most_common():
                if cls_name == 'red':
                    color = (0, 0, 255)  # BGR: Red
                elif cls_name == 'yellow':
                    color = (0, 255, 255)  # BGR: Yellow
                elif cls_name == 'green':
                    color = (0, 255, 0)  # BGR: Green
                else:
                    color = (255, 0, 0)  # BGR: Blue
                    
                cv2.putText(annotated_frame, f"{cls_name.upper()}: {count}", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30
        
            # Write to output video if recording enabled
            if output_writer:
                output_writer.write(annotated_frame)
        
        # Always display the frame, even when paused
        # Display the frame in real-time
        cv2.imshow("Traffic Light Detection", annotated_frame if not paused else annotated_frame)
        
        # Check for keyboard input
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        # Handle navigation keys
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("Detection stopped by user")
            break
        elif key == 32:  # SPACE
            paused = not paused
            print("Video " + ("paused" if paused else "resumed"))
        elif key == 83 or key == 3:  # RIGHT ARROW (may vary by system)
            # Skip forward
            new_frame = min(frame_count + SKIP_FRAMES, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            skip_message = f"Skipped forward to frame {new_frame}"
            skip_message_time = time.time()
            print(skip_message)
            # Ensure we're not paused after skipping
            paused = False
        elif key == 81 or key == 2:  # LEFT ARROW (may vary by system)
            # Skip backward
            new_frame = max(frame_count - SKIP_FRAMES, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            skip_message = f"Skipped backward to frame {new_frame}"
            skip_message_time = time.time()
            print(skip_message)
            # Ensure we're not paused after skipping
            paused = False
    
    # Release resources
    cap.release()
    if output_writer:
        output_writer.release()
    cv2.destroyAllWindows()
    
    # Display overall metrics
    print("\n===== VIDEO DETECTION METRICS =====")
    print(f"Video duration: {total_frames/fps:.2f} seconds")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with detections: {frames_with_detections} ({frames_with_detections/frame_count*100:.1f}%)")
    print(f"Total detections: {detection_count}")
    
    if processing_times:
        avg_time = np.mean(processing_times)
        print(f"Average processing time: {avg_time:.2f} ms per frame")
        print(f"Effective processing speed: {1000/avg_time:.2f} FPS")
    
    print("\nDetections by class:")
    for class_name, count in class_counter.items():
        print(f"  {class_name.upper()}: {count}")
    
    if confidence_scores:
        print(f"\nConfidence scores:")
        print(f"  Average: {np.mean(confidence_scores):.3f}")
        print(f"  Min: {min(confidence_scores):.3f}")
        print(f"  Max: {max(confidence_scores):.3f}")
    
    if RECORD_OUTPUT:
        print(f"\nProcessed video saved to: {OUTPUT_PATH}")
    
    # Create a detection summary chart
    plt.figure(figsize=(14, 8))

    # Detection counts by class
    plt.subplot(2, 2, 1)
    labels = list(class_counter.keys())
    counts = list(class_counter.values())
    colors = ['red' if cls == 'red' else 'yellow' if cls == 'yellow' else 'green' if cls == 'green' else 'blue' 
              for cls in labels]
    plt.bar(labels, counts, color=colors)
    plt.title('Detection Count by Class')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Processing time distribution
    plt.subplot(2, 2, 2)
    plt.hist(processing_times, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(processing_times), color='red', linestyle='dashed', linewidth=1)
    plt.title('Processing Time Distribution')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency')
    
    # Confidence score distribution
    if confidence_scores:
        plt.subplot(2, 2, 3)
        plt.hist(confidence_scores, bins=30, color='lightgreen', edgecolor='black')
        plt.axvline(np.mean(confidence_scores), color='red', linestyle='dashed', linewidth=1)
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
    
    # Save and show the chart
    plt.tight_layout()
    plt.savefig('detection_summary.png')
    plt.show()

if __name__ == "__main__":
    run_traffic_light_detection_video()