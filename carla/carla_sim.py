import carla
import tkinter as tk
import numpy as np
import random
from PIL import Image, ImageTk
import cv2 as cv
import os


window_width, window_height = 800, 600


# 1. Connect to CARLA and load Town01
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town01')

# 2. Spawn the first vehicle at a random spawn point
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 3. Create an RGB camera and attach to the vehicle
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')


#Camera position and rotation
camera_transform = carla.Transform(carla.Location(x=-4, z=4), carla.Rotation(pitch=-15))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

root = tk.Tk()
root.title("Carla Self-Driving Car Simulation")
root.protocol("WM_DELETE_WINDOW", exit.handler)

# Create additional windows
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

# Create canvas for each window to display images
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


main_photo = None
lane_photo = None
sign_photo = None
light_photo = None

def numpy_to_tkinter(array):
    """Convert a numpy array to a tkinter PhotoImage"""
    img = Image.fromarray(array)
    photo = ImageTk.PhotoImage(image=img)
    return photo

def show_image(image):
    global main_photo, lane_photo, sign_photo, light_photo
    
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb_image = array[:, :, :3][:, :, ::-1]
    
    main_photo = numpy_to_tkinter(rgb_image)
    main_canvas.create_image(0, 0, image=main_photo, anchor="nw")
    
    lane_results_hough = detect_lanes_hough(rgb_image)
    lane_results_ml = detect_lanes_ml(rgb_image)
    sign_results = detect_signs(rgb_image)
    light_results_detect = detect_traffic_lights(rgb_image)
    light_results_class = classify_traffic_lights(rgb_image)
    
    lane_image_hough = rgb_image.copy()
    lane_image_ml = rgb_image.copy()
    sign_image = rgb_image.copy()
    light_detect_image = rgb_image.copy()
    light_class_image = rgb_image.copy()
    
    if lane_results_hough:
        for line in lane_results_hough:
            if isinstance(line, tuple) and len(line) == 2:
                cv.line(lane_image_hough, line[0], line[1], (255, 0, 0), 2)

    if lane_results_ml:
        for line in lane_results_ml:
            if isinstance(line, tuple) and len(line) == 2:
                cv.line(lane_image_ml, line[0], line[1], (255, 0, 0), 2)
    
    if sign_results:
        for sign in sign_results:
            if isinstance(sign, dict) and 'bbox' in sign:
                x, y, w, h = sign['bbox']
                cv.rectangle(sign_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if 'label' in sign:
                    cv.putText(sign_image, sign['label'], (x, y-10), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if light_results_detect:
        pass

    if light_results_class:
        pass
    
    lane_hough_photo = numpy_to_tkinter(lane_image_hough)
    lane_hough_canvas.create_image(0, 0, image=lane_hough_photo, anchor="nw")

    lane_ml_photo = numpy_to_tkinter(lane_image_ml)
    lane_ml_canvas.create_image(0, 0, image=lane_ml_photo, anchor="nw")
    
    sign_photo = numpy_to_tkinter(sign_image)
    sign_canvas.create_image(0, 0, image=sign_photo, anchor="nw")
    
    light_detect_photo = numpy_to_tkinter(light_detect_image)
    light_canvas.create_image(0, 0, image=light_detect_photo, anchor="nw")

    light_class_photo = numpy_to_tkinter(light_class_image)
    light_canvas.create_image(0, 0, image=light_class_photo, anchor="nw")
    
    root.update()



def exit_handler():
    print("Cleaning up resources...")
    camera.stop()
    vehicle.destroy()
    root.quit()
    root.destroy()


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


camera.listen(lambda image: show_image(image))

try:
    root.mainloop()
except KeyboardInterrupt:
    pass
finally:
    exit_handler()

