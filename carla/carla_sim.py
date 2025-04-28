import carla
import tkinter as tk
import numpy as np
import random
from PIL import Image, ImageTk
import cv2 as cv

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
root.protocol("WM_DELETE_WINDOW", lambda: exit.handler())

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
    
    # Convert raw CARLA image to a (H,W,3) NumPy array in RGB order
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb_image = array[:, :, :3][:, :, ::-1]  # BGRA to RGB
    
    # Update main window with original image
    main_photo = numpy_to_tkinter(rgb_image)
    main_canvas.create_image(0, 0, image=main_photo, anchor="nw")
    
    # Process detections
    lane_results_hough = detect_lanes_hough(rgb_image)
    lane_results_ml = detect_lanes_ml(rgb_image)
    sign_results = detect_signs(rgb_image)
    light_results = detect_traffic_lights(rgb_image)
    
    # Create copies for each window to draw on
    lane_image_hough = rgb_image.copy()
    lane_image_ml = rgb_image.copy()
    sign_image = rgb_image.copy()
    light_image = rgb_image.copy()
    
    # Draw lane detection results (example - modify based on your actual results)
    if lane_results_hough:
        # Example: Draw lines on the image
        for line in lane_results_hough:
            if isinstance(line, tuple) and len(line) == 2:
                # Assuming line is ((x1,y1), (x2,y2))
                cv.line(lane_image_hough, line[0], line[1], (255, 0, 0), 2)
            # Handle other result formats as needed

    if lane_results_ml:
        # Example: Draw lines on the image
        for line in lane_results_ml:
            if isinstance(line, tuple) and len(line) == 2:
                # Assuming line is ((x1,y1), (x2,y2))
                cv.line(lane_image_ml, line[0], line[1], (255, 0, 0), 2)
            # Handle other result formats as needed
    
    # Draw sign detection results (example)
    if sign_results:
        # Example: Draw boxes around signs
        for sign in sign_results:
            if isinstance(sign, dict) and 'bbox' in sign:
                # Assuming bbox is (x, y, w, h)
                x, y, w, h = sign['bbox']
                cv.rectangle(sign_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Add text if label exists
                if 'label' in sign:
                    cv.putText(sign_image, sign['label'], (x, y-10), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw traffic light detection results (example)
    if light_results:
        # Similar to sign detection
        pass
    
    # Update all windows with their respective images
    lane_hough_photo = numpy_to_tkinter(lane_image_hough)
    lane_hough_canvas.create_image(0, 0, image=lane_hough_photo, anchor="nw")

    lane_ml_photo = numpy_to_tkinter(lane_image_ml)
    lane_ml_canvas.create_image(0, 0, image=lane_ml_photo, anchor="nw")
    
    sign_photo = numpy_to_tkinter(sign_image)
    sign_canvas.create_image(0, 0, image=sign_photo, anchor="nw")
    
    light_photo = numpy_to_tkinter(light_image)
    light_canvas.create_image(0, 0, image=light_photo, anchor="nw")
    
    # Schedule the next update
    root.update()



def exit_handler():
    print("Cleaning up resources...")
    camera.stop()
    vehicle.destroy()
    root.quit()
    root.destroy()


def detect_lanes_hough(frames):
    from lane_detection_hough import lane_detection

    results = lane_detection(frames)

    return results

def detect_lanes_ml(frames):
    from lane_detection_hough import lane_detection

    results = lane_detection(frames)

    return results

def detect_traffic_lights(frames):
    from traffic_light import traffic_light_detection

    results = traffic_light_detection(frames)

    return results

def detect_signs(frames):
    from sign_detection import predict_sign

    results = predict_sign(frames)

    return results


camera.listen(lambda image: show_image(image))

try:
    root.mainloop()
except KeyboardInterrupt:
    pass
finally:
    exit_handler()

