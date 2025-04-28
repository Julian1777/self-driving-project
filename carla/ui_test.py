import tkinter as tk
import numpy as np
import random
from PIL import Image, ImageTk
import cv2 as cv
import time
from dataclasses import dataclass

window_width, window_height = 800, 600

# Mock image generator
def generate_test_image():
    """Generate a test image with a road and some simple shapes"""
    # Create a base image (sky and ground)
    img = np.ones((window_height, window_width, 3), dtype=np.uint8) * 255
    
    # Draw road (dark gray)
    cv.rectangle(img, (0, 300), (window_width, window_height), (80, 80, 80), -1)
    
    # Draw lane markings (white dashed lines)
    for i in range(0, window_width, 50):
        cv.rectangle(img, (i, 450), (i+30, 460), (255, 255, 255), -1)
    
    # Draw a center line (yellow)
    cv.line(img, (0, 400), (window_width, 400), (0, 255, 255), 5)
    
    # Add some "signs" as colored rectangles
    sign_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
    for i in range(3):
        x = random.randint(50, window_width-100)
        y = random.randint(100, 250)
        color = random.choice(sign_colors)
        cv.rectangle(img, (x, y), (x+50, y+50), color, -1)
    
    # Add a traffic light
    light_color = random.choice([(0, 0, 255), (0, 255, 255), (0, 255, 0)])
    cv.circle(img, (700, 150), 20, light_color, -1)
    
    return img

# Mock detection results
def mock_lane_detection_hough(image):
    """Return simulated lane detection results"""
    # Return a list of lines as ((x1,y1), (x2,y2)) tuples
    lines = [
        ((100, 450), (300, 450)),
        ((400, 450), (600, 450)),
        ((200, 400), (400, 400)),
        ((500, 400), (700, 400))
    ]
    return lines

def mock_lane_detection_ml(image):
    """Return simulated ML lane detection results (slightly different)"""
    # Return a list of lines as ((x1,y1), (x2,y2)) tuples
    lines = [
        ((150, 450), (350, 450)),
        ((450, 450), (650, 450)),
        ((150, 400), (350, 400)),
        ((450, 400), (650, 400)),
        ((150, 500), (650, 500))  # Extra line to differentiate from Hough
    ]
    return lines

def mock_sign_detection(image):
    """Return simulated sign detection results"""
    # Return a list of dictionaries with bbox and label
    signs = [
        {'bbox': (150, 150, 50, 50), 'label': 'Stop'},
        {'bbox': (400, 200, 50, 50), 'label': 'Yield'}
    ]
    return signs

def mock_traffic_light(image):
    """Return simulated traffic light detection results"""
    states = ['Red', 'Yellow', 'Green']
    lights = [
        {'bbox': (680, 130, 40, 40), 'state': random.choice(states)}
    ]
    return lights

# Create a class to simulate CARLA images
class MockImage:
    def __init__(self, array):
        self.height, self.width = array.shape[:2]
        # Convert to BGRA format like CARLA provides
        bgra = cv.cvtColor(array, cv.COLOR_RGB2BGRA)
        self.raw_data = bgra.tobytes()

# Function to convert numpy array to Tkinter PhotoImage
def numpy_to_tkinter(array):
    """Convert a numpy array to a tkinter PhotoImage"""
    # Make sure the array is uint8
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(array)
    # Convert to PhotoImage
    photo = ImageTk.PhotoImage(image=img)
    return photo

# Create windows
root = tk.Tk()
root.title("Carla Simulation (TEST MODE)")
root.geometry(f"{window_width}x{window_height}+0+0")

# Create additional windows
lane_hough_window = tk.Toplevel(root)
lane_hough_window.title("Lane Detection (Hough)")
lane_hough_window.geometry(f"{window_width}x{window_height}+{window_width}+0")

lane_ml_window = tk.Toplevel(root)
lane_ml_window.title("Lane Detection (ML)")
lane_ml_window.geometry(f"{window_width}x{window_height}+0+{window_height}")

sign_window = tk.Toplevel(root)
sign_window.title("Sign Detection")
sign_window.geometry(f"{window_width}x{window_height}+{window_width}+{window_height}")

# Create canvas for each window
main_canvas = tk.Canvas(root, width=window_width, height=window_height)
main_canvas.pack(fill="both", expand=True)

lane_hough_canvas = tk.Canvas(lane_hough_window, width=window_width, height=window_height)
lane_hough_canvas.pack(fill="both", expand=True)

lane_ml_canvas = tk.Canvas(lane_ml_window, width=window_width, height=window_height)
lane_ml_canvas.pack(fill="both", expand=True)

sign_canvas = tk.Canvas(sign_window, width=window_width, height=window_height)
sign_canvas.pack(fill="both", expand=True)

# Variables to store PhotoImage references
main_photo = None
lane_hough_photo = None
lane_ml_photo = None
sign_photo = None

# Callback function to update all windows
def show_image(image):
    global main_photo, lane_hough_photo, lane_ml_photo, sign_photo
    
    # Convert raw image data to RGB numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb_image = array[:, :, :3][:, :, ::-1]  # BGRA to RGB
    
    # Update main window with original image
    main_photo = numpy_to_tkinter(rgb_image)
    main_canvas.create_image(0, 0, image=main_photo, anchor="nw")
    
    # Process detections
    lane_hough_results = mock_lane_detection_hough(rgb_image)
    lane_ml_results = mock_lane_detection_ml(rgb_image)
    sign_results = mock_sign_detection(rgb_image)
    light_results = mock_traffic_light(rgb_image)
    
    # Create copies for each window
    lane_hough_image = rgb_image.copy()
    lane_ml_image = rgb_image.copy()
    sign_image = rgb_image.copy()
    
    # Draw lane detection results (Hough)
    for line in lane_hough_results:
        cv.line(lane_hough_image, line[0], line[1], (255, 0, 0), 2)
    
    # Draw lane detection results (ML)
    for line in lane_ml_results:
        cv.line(lane_ml_image, line[0], line[1], (0, 255, 0), 2)
    
    # Draw sign detection results
    for sign in sign_results:
        x, y, w, h = sign['bbox']
        cv.rectangle(sign_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv.putText(sign_image, sign['label'], (x, y-10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Update all windows
    lane_hough_photo = numpy_to_tkinter(lane_hough_image)
    lane_hough_canvas.create_image(0, 0, image=lane_hough_photo, anchor="nw")
    
    lane_ml_photo = numpy_to_tkinter(lane_ml_image)
    lane_ml_canvas.create_image(0, 0, image=lane_ml_photo, anchor="nw")
    
    sign_photo = numpy_to_tkinter(sign_image)
    sign_canvas.create_image(0, 0, image=sign_photo, anchor="nw")
    
    # Update the display
    root.update()

# Exit handler
def exit_handler():
    print("Exiting test mode...")
    root.quit()
    root.destroy()

# Configure window close events
root.protocol("WM_DELETE_WINDOW", exit_handler)
lane_hough_window.protocol("WM_DELETE_WINDOW", exit_handler)
lane_ml_window.protocol("WM_DELETE_WINDOW", exit_handler)
sign_window.protocol("WM_DELETE_WINDOW", exit_handler)

# Update frame function
def update_frame():
    # Generate a new test image
    test_img = generate_test_image()
    # Create a mock CARLA image
    mock_carla_img = MockImage(test_img)
    # Call the show_image function
    show_image(mock_carla_img)
    # Schedule the next update (33ms ≈ 30fps)
    root.after(33, update_frame)

# Start the update loop
print("Starting UI test...")
update_frame()

# Start the main loop
try:
    root.mainloop()
except KeyboardInterrupt:
    print("Keyboard interrupt received")
finally:
    exit_handler()