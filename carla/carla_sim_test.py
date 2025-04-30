import tkinter as tk
import numpy as np
import random
from PIL import Image, ImageTk
import cv2 as cv
import os
import threading
import time
from queue import Queue

VIDEO_PATH = "test_video_ams_cut.mp4"
FRAME_SKIP = 2
window_width, window_height = 800, 600
frame_count = 0


root = tk.Tk()
root.title("Carla Self-Driving Car Simulation")

control_frame = tk.Frame(root)
control_frame.pack(side=tk.BOTTOM, fill=tk.X)

play_btn = tk.Button(control_frame, text="Play", command=lambda: toggle_play())
play_btn.pack(side=tk.LEFT, padx=10, pady=5)

reset_btn = tk.Button(control_frame, text="Reset", command=lambda: reset_video())
reset_btn.pack(side=tk.LEFT, padx=10, pady=5)

status_label = tk.Label(control_frame, text="Ready")
status_label.pack(side=tk.RIGHT, padx=10)

lane_hough_window = tk.Toplevel(root)
lane_hough_window.title("Lane Detection using Hough Transform")
lane_hough_window.geometry(f"{window_width}x{window_height}+{window_width}+0")

lane_ml_window = tk.Toplevel(root)
lane_ml_window.title("Lane Detection using Machine Learning")
lane_ml_window.geometry(f"{window_width}x{window_height}+{window_width}+0")

sign_window = tk.Toplevel(root)
sign_window.title("Sign Detection")
sign_window.geometry(f"{window_width}x{window_height}+0+{window_height}")

light_window = tk.Toplevel(root)
light_window.title("Traffic Light Detection")
light_window.geometry(f"{window_width}x{window_height}+{window_width}+{window_height}")

main_canvas = tk.Canvas(root, width=window_width, height=window_height)
main_canvas.pack(fill="both", expand=True)

lane_hough_canvas = tk.Canvas(lane_hough_window, width=window_width, height=window_height)
lane_hough_canvas.pack(fill="both", expand=True)

lane_ml_canvas = tk.Canvas(lane_ml_window, width=window_width, height=window_height)
lane_ml_canvas.pack(fill="both", expand=True)

sign_canvas = tk.Canvas(sign_window, width=window_width, height=window_height)
sign_canvas.pack(fill="both", expand=True)

light_canvas = tk.Canvas(light_window, width=window_width, height=window_height)
light_canvas.pack(fill="both", expand=True)


cap = None
playing = False
delay = 30

main_photo = None
lane_hough_photo = None
lane_ml_photo = None
sign_photo = None
light_detect_photo = None
light_class_photo = None

def numpy_to_tkinter(array):
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
    
    if array.shape[0] != window_height or array.shape[1] != window_width:
        array = cv.resize(array, (window_width, window_height))

    img = Image.fromarray(array)
    photo = ImageTk.PhotoImage(image=img)
    return photo

def show_image(frame):
    global main_photo, lane_ml_photo, sign_photo, light_detect_photo, light_class_photo, frame_count

    frame_count += 1
    
    rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    main_photo = numpy_to_tkinter(rgb_image)
    main_canvas.create_image(0, 0, image=main_photo, anchor="nw")
    
    if frame_count % 5 == 0:
        try:
            lane_results_hough = detect_lanes_hough(rgb_image)
            lane_image_hough = rgb_image.copy()
            
            if lane_results_hough is not None and isinstance(lane_results_hough, (list, tuple, np.ndarray)) and len(lane_results_hough) > 0:
                for line in lane_results_hough:
                    if isinstance(line, tuple) and len(line) == 2:
                        cv.line(lane_image_hough, line[0], line[1], (255, 0, 0), 2)
            
            cv.putText(lane_image_hough, f"Frame: {frame_count}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            lane_hough_photo = numpy_to_tkinter(lane_image_hough)
            lane_hough_canvas.create_image(0, 0, image=lane_hough_photo, anchor="nw")
        except Exception as e:
            print(f"Error in lane_detection_hough: {e}")

    elif frame_count % 5 == 1:
        try:
            lane_results_ml = detect_lanes_ml(rgb_image)
            lane_image_ml = rgb_image.copy()
            
            if lane_results_ml is not None and isinstance(lane_results_ml, (list, tuple, np.ndarray)) and len(lane_results_ml) > 0:
                for line in lane_results_ml:
                    if isinstance(line, tuple) and len(line) == 2:
                        cv.line(lane_image_ml, line[0], line[1], (0, 255, 0), 2)
            
            cv.putText(lane_image_ml, f"Frame: {frame_count}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            lane_ml_photo = numpy_to_tkinter(lane_image_ml)
            lane_ml_canvas.create_image(0, 0, image=lane_ml_photo, anchor="nw")
        except Exception as e:
            print(f"Error in lane_detection_ml: {e}")
    
    elif frame_count % 5 == 2:
        try:
            sign_results = detect_signs(rgb_image)
            sign_image = rgb_image.copy()
            
            if sign_results is not None and isinstance(sign_results, (list, tuple, np.ndarray)) and len(sign_results) > 0:
                for sign in sign_results:
                    if isinstance(sign, dict) and 'bbox' in sign:
                        x, y, w, h = sign['bbox']
                        cv.rectangle(sign_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        if 'label' in sign:
                            cv.putText(sign_image, sign['label'], (x, y-10), 
                                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv.putText(sign_image, f"Frame: {frame_count}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            sign_photo = numpy_to_tkinter(sign_image)
            sign_canvas.create_image(0, 0, image=sign_photo, anchor="nw")
        except Exception as e:
            print(f"Error in sign_detection: {e}")
    
    elif frame_count % 5 == 3:
        try:
            light_results_detect = detect_traffic_lights(rgb_image)
            light_detect_image = rgb_image.copy()
            
            if light_results_detect is not None and isinstance(light_results_detect, (list, tuple, np.ndarray)) and len(light_results_detect) > 0:
                for light in light_results_detect:
                    if isinstance(light, dict) and 'bbox' in light:
                        x1, y1, x2, y2 = light['bbox']
                        cv.rectangle(light_detect_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        if 'confidence' in light:
                            cv.putText(light_detect_image, f"Conf: {light['confidence']:.2f}", 
                                    (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv.putText(light_detect_image, f"Frame: {frame_count}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            light_detect_photo = numpy_to_tkinter(light_detect_image)
            light_canvas.create_image(0, 0, image=light_detect_photo, anchor="nw")
        except Exception as e:
            print(f"Error in traffic_light_detection: {e}")
    
    elif frame_count % 5 == 4:
        try:
            light_results_class = classify_traffic_lights(rgb_image)
            light_class_image = rgb_image.copy()
            
            if light_results_class is not None and isinstance(light_results_class, (list, tuple, np.ndarray)) and len(light_results_class) > 0:
                for light in light_results_class:
                    if isinstance(light, dict) and 'state' in light:
                        label = light['state']
                        confidence = light.get('confidence', 0)
                        cv.putText(light_class_image, f"{label} ({confidence:.2f})", 
                                (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            cv.putText(light_class_image, f"Frame: {frame_count}", (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            light_class_photo = numpy_to_tkinter(light_class_image)
            light_canvas.create_image(0, 0, image=light_class_photo, anchor="nw")
        except Exception as e:
            print(f"Error in traffic_light_classification: {e}")
    
    # Update the UI
    root.update()
    

def open_video():
    global cap
    if os.path.exists(VIDEO_PATH):
        cap = cv.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video file {VIDEO_PATH}")
            return False
        return True
    else:
        print(f"Error: Video file not found: {VIDEO_PATH}")
        return False

def read_frame():
    global cap
    if cap is None or not cap.isOpened():
        return False
        
    for _ in range(FRAME_SKIP):
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            return False
    
    start_time = time.time()
    show_image(frame)
    processing_time = time.time() - start_time
    
    status_label.config(text=f"Frame: {frame_count} - Processing: {processing_time:.3f}s")
    
    return True

def update_frame():
    global playing, status_label
    if playing:
        success = read_frame()
        if success:
            status_label.config(text="Playing...")
            root.after(delay, update_frame)
        else:
            playing = False
            play_btn.config(text="Play")
            status_label.config(text="End of video")

def toggle_play():
    global playing
    playing = not playing
    if playing:
        play_btn.config(text="Pause")
        update_frame()
    else:
        play_btn.config(text="Play")
        status_label.config(text="Paused")

def reset_video():
    global cap, playing
    if cap is not None:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        read_frame()
        status_label.config(text="Video reset")
        playing = False
        play_btn.config(text="Play")

def detect_lanes_hough(frames):
    from lane_detection_hough import lane_detection

    lane_hough_results = lane_detection(frames)

    return lane_hough_results

def detect_lanes_ml(frames):
    from lane_detection_model import predict_lane

    lane_ml_results = predict_lane(frames)

    return lane_ml_results

def detect_traffic_lights(frames):
    from traffic_light_detect import detect_traffic_light

    light_detect_results = detect_traffic_light(frames)

    return light_detect_results

def classify_traffic_lights(frames):
    from traffic_light_class import predict_traffic_light

    light_class_results = predict_traffic_light(frames)

    return light_class_results

def detect_signs(frames):
    from sign_detection import predict_sign

    sign_results = predict_sign(frames)

    return sign_results

if open_video():
    read_frame()
    status_label.config(text="Video loaded - Press Play to start")
else:
    status_label.config(text="Error: Could not load video")
    
def on_closing():
    global cap
    if cap is not None:
        cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()