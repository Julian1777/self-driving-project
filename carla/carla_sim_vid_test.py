import tkinter as tk
import numpy as np
import random
from PIL import Image, ImageTk
import cv2 as cv
import os
import threading
import time
from queue import Queue

VIDEO_PATH = os.path.join("test_videos", "test_video_ams_cut.mp4")
FRAME_SKIP = 2
MODELS = {}
last_lane_update = 0
last_light_update = 0
last_sign_update = 0
last_vehicle_update = 0
update_interval = 5
window_width, window_height = 800, 600
frame_count = 0

vehicle_window = None
lane_hough_window = None
sign_window = None
light_window = None

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
lane_ml_window.geometry(f"{window_width}x{window_height}+{2*window_width}+0")

sign_window = tk.Toplevel(root)
sign_window.title("Sign Detection")
sign_window.geometry(f"{window_width}x{window_height}+0+{window_height}")

light_window = tk.Toplevel(root)
light_window.title("Traffic Light Detection")
light_window.geometry(f"{window_width}x{window_height}+{window_width}+{window_height}")

vehicle_window = tk.Toplevel(root)
vehicle_window.title("Vehicle & Pedestrian Detection")
vehicle_window.geometry(f"{window_width}x{window_height}+{2*window_width}+{window_height}")

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

vehicle_canvas = tk.Canvas(vehicle_window, width=window_width, height=window_height)
vehicle_canvas.pack(fill="both", expand=True)


cap = None
playing = False
delay = 30

main_photo = None
lane_hough_photo = None
lane_ml_photo = None
sign_photo = None
light_detect_photo = None
light_class_photo = None
vehicle_ped_photo = None

photo_refs = {}
def numpy_to_tkinter(array, window_id="main"):
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
    
    if array.shape[0] != window_height or array.shape[1] != window_width:
        array = cv.resize(array, (window_width, window_height))

    img = Image.fromarray(array)
    photo = ImageTk.PhotoImage(image=img)

    photo_refs[window_id] = photo
    
    return photo

def load_all_models():
    try:
        print("Loading models, please wait...")
        
        from ultralytics import YOLO
        import tensorflow as tf
        
        # Load vehicle & pedestrian detection model
        MODELS['vehicle'] = YOLO(os.path.join("model", "vehicle_pedestrian_detection.pt"))
        print("Vehicle detection model loaded")
        
        # Load sign detection model
        MODELS['sign_detect'] = YOLO(os.path.join("model", "sign_detection.pt"))
        print("Sign detection model loaded")
        
        # Load sign classification model
        MODELS['sign_classify'] = tf.keras.models.load_model(os.path.join("model", "sign_classification.h5"))
        print("Sign classification model loaded")
        
        # Load traffic light detection model
        MODELS['traffic_light'] = YOLO(os.path.join("model", "traffic_light_detect_class.pt"))
        print("Traffic light model loaded")
        
        print("All models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_image(frame):
    global main_photo, lane_ml_photo, sign_photo, light_detect_photo, light_class_photo, vehicle_ped_photo
    global frame_count, last_lane_update, last_sign_update, last_light_update, last_vehicle_update

    frame_count += 1
    
    rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    main_photo = numpy_to_tkinter(rgb_image)
    main_canvas.create_image(0, 0, image=main_photo, anchor="nw")
    
    if frame_count % 5 == 0 and frame_count - last_lane_update >= update_interval:
        try:
            lane_results_hough = detect_lanes_hough(rgb_image)
            lane_image_hough = rgb_image.copy()
            
            if lane_results_hough is not None and isinstance(lane_results_hough, (list, tuple, np.ndarray)) and len(lane_results_hough) > 0:
                for line in lane_results_hough:
                    if isinstance(line, tuple) and len(line) == 2:
                        cv.line(lane_image_hough, line[0], line[1], (255, 0, 0), 2)
            
            cv.putText(lane_image_hough, f"Frame: {frame_count}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            lane_hough_photo = numpy_to_tkinter(lane_image_hough, "lane_hough")
            lane_hough_canvas.create_image(0, 0, image=photo_refs["lane_hough"], anchor="nw")
        except Exception as e:
            print(f"Error in lane_detection_hough: {e}")

    # elif frame_count % 5 == 1:
    #     try:
    #         lane_results_ml = detect_lanes_ml(rgb_image)
    #         lane_image_ml = rgb_image.copy()
            
    #         if lane_results_ml is not None and isinstance(lane_results_ml, (list, tuple, np.ndarray)) and len(lane_results_ml) > 0:
    #             for line in lane_results_ml:
    #                 if isinstance(line, tuple) and len(line) == 2:
    #                     cv.line(lane_image_ml, line[0], line[1], (0, 255, 0), 2)
            
    #         cv.putText(lane_image_ml, f"Frame: {frame_count}", (10, 30), 
    #                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
    #         lane_ml_photo = numpy_to_tkinter(lane_image_ml)
    #         lane_ml_canvas.create_image(0, 0, image=lane_ml_photo, anchor="nw")
    #     except Exception as e:
    #         print(f"Error in lane_detection_ml: {e}")
    
    elif frame_count % 5 == 2 and frame_count - last_sign_update >= update_interval:
        try:
            sign_results = detect_classify_signs(rgb_image)
            sign_image = rgb_image.copy()
            
            if sign_results is not None and isinstance(sign_results, (list, tuple, np.ndarray)) and len(sign_results) > 0:
                for sign in sign_results:
                    if isinstance(sign, dict) and 'bbox' in sign:
                        x1, y1, x2, y2 = sign['bbox']

                        if 'classification_confidence' in sign and sign['classification_confidence'] > 0.7:
                            color = (0, 255, 0)
                        elif 'classification_confidence' in sign and sign['classification_confidence'] > 0.4:
                            color = (0, 255, 255)
                        else:
                            color = (0, 160, 255)

                        cv.rectangle(sign_image, (x1, y1), (x2, y2), color, 2)

                        if 'classification' in sign:
                            class_text = sign['classification']
                            if len(class_text) > 20:
                                class_text = class_text[:20] + "..."
                            confidence = sign['classification_confidence']
                            label = f"{class_text}: {confidence:.2f}"

                            text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                            cv.rectangle(sign_image, (x1, y1 - text_size[1] - 5), 
                                         (x1 + text_size[0], y1), color, -1)
                            
                            cv.putText(sign_image, label, (x1, y1 - 5), 
                                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv.putText(sign_image, f"Frame: {frame_count}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            sign_photo = numpy_to_tkinter(sign_image, "sign")
            sign_canvas.create_image(0, 0, image=photo_refs["sign"], anchor="nw")
        except Exception as e:
            print(f"Error in sign_detection: {e}")
    
    elif frame_count % 5 == 3 and frame_count - last_light_update >= update_interval:
        try:
            light_results_detect = detect_class_traffic_lights(rgb_image)
            light_detect_image = rgb_image.copy()
            
            if light_results_detect is not None and isinstance(light_results_detect, (list, tuple, np.ndarray)) and len(light_results_detect) > 0:
                for light in light_results_detect:
                    if isinstance(light, dict) and 'bbox' in light:
                        x1, y1, x2, y2 = light['bbox']

                        if 'class' in light:
                            if light['class'] == 'red':
                                color = (0, 0, 255)
                            elif light['class'] == 'yellow':
                                color = (0, 255, 255)
                            elif light['class'] == 'green':
                                color = (0, 255, 0)
                            else:
                                color = (255, 255, 255)

                            cv.rectangle(light_detect_image, (x1, y1), (x2, y2), color, 2)

                            label = f"{light['class']}: {light['confidence']:.2f}"
                            cv.putText(light_detect_image, label, 
                                (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            cv.rectangle(light_detect_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            if 'confidence' in light:
                                cv.putText(light_detect_image, f"Conf: {light['confidence']:.2f}", 
                                    (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv.putText(light_detect_image, f"Frame: {frame_count}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            light_detect_photo = numpy_to_tkinter(light_detect_image, "light")
            light_canvas.create_image(0, 0, image=photo_refs["light"], anchor="nw")

            last_light_update = frame_count
        except Exception as e:
            print(f"Error in traffic_light_detection: {e}")

    elif frame_count % 5 == 4 and frame_count - last_vehicle_update >= update_interval:
        try:
            vehicle_ped_results = detect_vehicles_pedestrians(rgb_image)
            vehicle_ped_image = rgb_image.copy()

            if vehicle_ped_results and len(vehicle_ped_results) > 0:
                for detection in vehicle_ped_results:
                    if 'bbox' in detection and 'class' in detection:
                        x1, y1, x2, y2 = detection['bbox']
                        class_name = detection['class'].lower()
                        conf = detection['confidence']
                        
                        if class_name == 'pedestrian':
                            color = (0, 255, 0)
                        elif class_name in ['car', 'truck', 'bus']:
                            color = (0, 0, 255)
                        elif class_name in ['bicycle', 'motorcycle']:
                            color = (255, 165, 0)
                        else:
                            color = (255, 255, 255)

                        cv.rectangle(vehicle_ped_image, (x1, y1), (x2, y2), color, 2)

                        label = f"{class_name}: {conf:.2f}"
                        cv.putText(vehicle_ped_image, label, 
                                  (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv.putText(vehicle_ped_image, f"Frame: {frame_count}", (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            vehicle_ped_photo = numpy_to_tkinter(vehicle_ped_image, "vehicle")
            vehicle_canvas.create_image(0, 0, image=photo_refs["vehicle"], anchor="nw")

            last_vehicle_update = frame_count
        except Exception as e:
            print(f"Error in vehicle_pedestrian_detection: {e}")
            
    lane_hough_window.update()
    light_window.update()
    sign_window.update()
    vehicle_window.update()
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

    print(f"Lane detection input shape: {frames.shape}, dtype: {frames.dtype}")

    bgr_frames = cv.cvtColor(frames, cv.COLOR_RGB2BGR)

    lane_hough_results = lane_detection(bgr_frames)
    print(f"Got lane results: {lane_hough_results[:2] if lane_hough_results else 'None'}")

    return lane_hough_results

# def detect_lanes_ml(frames):
#     from lane_detection_model import predict_lane

#     lane_ml_results = predict_lane(frames)

#     return lane_ml_results

def detect_class_traffic_lights(frames):
    from traffic_light_detect_class import detect_classify_traffic_light

    bgr_frames = cv.cvtColor(frames, cv.COLOR_RGB2BGR)

    light_detect_results = detect_classify_traffic_light(bgr_frames)
    print(f"Got traffic light results: {light_detect_results[:2] if light_detect_results else 'None'}")

    return light_detect_results

def detect_classify_signs(frames):
    from sign_detection_classification import detect_classify_sign

    sign_results = detect_classify_sign(frames)
    print(f"Got sign results: {sign_results[:2] if sign_results else 'None'}")

    return sign_results

def detect_vehicles_pedestrians(frames):
    from vehicle_pedestrian_detection import detect_vehicles_pedestrians

    print(f"Vehicle detection input shape: {frames.shape}, dtype: {frames.dtype}")

    vehicle_ped_results = detect_vehicles_pedestrians(frames)
    print(f"Got vehicle results: {vehicle_ped_results[:2] if vehicle_ped_results else 'None'}")

    return vehicle_ped_results

if __name__ == "__main__":
    load_all_models()

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