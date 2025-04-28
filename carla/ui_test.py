import tkinter as tk
import numpy as np
import random
from PIL import Image, ImageTk
import cv2 as cv
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

#############################################
# CONFIGURATION
#############################################
# Display settings
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
FPS = 30  # Frames per second

# Color constants (BGR format for OpenCV)
class Colors:
    # Environment
    SKY = (135, 206, 235)      # Light blue
    GROUND = (34, 139, 34)     # Forest green
    ROAD = (80, 80, 80)        # Dark gray
    LANE_MARKING = (255, 255, 255)  # White
    CENTER_LINE = (0, 215, 255)     # Yellow
    
    # Detection visualization
    HOUGH_LANE = (255, 0, 0)       # Red
    ML_LANE = (0, 255, 0)          # Green
    SIGN_BOX = (0, 0, 255)         # Blue
    TEXT_COLOR = (255, 255, 255)   # White
    
    # Traffic lights
    RED_LIGHT = (0, 0, 255)
    YELLOW_LIGHT = (0, 255, 255)
    GREEN_LIGHT = (0, 255, 0)

#############################################
# MOCK SIMULATION CLASSES
#############################################
@dataclass
class VehicleState:
    """Current state of the test vehicle"""
    speed: float = 50.0  # km/h
    steering_angle: float = 0.0  # degrees, positive is right
    position_x: float = WINDOW_WIDTH // 2
    position_y: float = WINDOW_HEIGHT - 150

class MockImage:
    """Simulates a CARLA camera image"""
    def __init__(self, array):
        self.height, self.width = array.shape[:2]
        # Convert to BGRA format like CARLA provides
        bgra = cv.cvtColor(array, cv.COLOR_BGR2BGRA)
        self.raw_data = bgra.tobytes()

class SceneGenerator:
    """Generates test scenes with road, signs, and traffic lights"""
    def __init__(self):
        self.vehicle_state = VehicleState()
        self.frame_count = 0
    
    def update_vehicle_state(self):
        """Update the vehicle state for the next frame"""
        # Slightly vary steering angle for realistic movement
        self.vehicle_state.steering_angle = 5 * np.sin(self.frame_count / 50)
        # Slightly vary speed
        self.vehicle_state.speed = 50 + 10 * np.sin(self.frame_count / 100)
        self.frame_count += 1
    
    def generate_image(self) -> np.ndarray:
        """Generate a test image with a road and some simple shapes"""
        # Create base image (sky and ground)
        img = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        img[:WINDOW_HEIGHT//2, :] = Colors.SKY
        img[WINDOW_HEIGHT//2:, :] = Colors.GROUND
        
        # Draw road (perspective effect)
        road_pts = np.array([
            [WINDOW_WIDTH//2 - 300, WINDOW_HEIGHT],  # Bottom left
            [WINDOW_WIDTH//2 + 300, WINDOW_HEIGHT],  # Bottom right
            [WINDOW_WIDTH//2 + 100, WINDOW_HEIGHT//2],  # Top right
            [WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT//2],  # Top left
        ], np.int32)
        cv.fillPoly(img, [road_pts], Colors.ROAD)
        
        # Draw lane markings (perspective effect)
        for offset in [-150, 0, 150]:
            pts = np.array([
                [WINDOW_WIDTH//2 + offset - 10, WINDOW_HEIGHT],
                [WINDOW_WIDTH//2 + offset + 10, WINDOW_HEIGHT],
                [WINDOW_WIDTH//2 + offset/3 + 5, WINDOW_HEIGHT//2],
                [WINDOW_WIDTH//2 + offset/3 - 5, WINDOW_HEIGHT//2],
            ], np.int32)
            
            # Center line is solid yellow, others are dashed white
            if offset == 0:
                cv.fillPoly(img, [pts], Colors.CENTER_LINE)
            else:
                # Create dashed lines by only drawing portions
                for i in range(0, 10):
                    y_start = WINDOW_HEIGHT//2 + i * (WINDOW_HEIGHT//2)//10
                    y_end = y_start + (WINDOW_HEIGHT//2)//20
                    if i % 2 == 0:  # Only draw every other segment
                        mask = np.zeros_like(img)
                        cv.fillPoly(mask, [pts], (255, 255, 255))
                        # Fix the broadcasting error - extract only one channel
                        road_mask = (img[:,:,0] == Colors.ROAD[0]) & (img[:,:,1] == Colors.ROAD[1]) & (img[:,:,2] == Colors.ROAD[2])
                        for c in range(3):
                            mask_channel = (mask[:,:,c] > 0) & road_mask
                            img[y_start:y_end,:,c][mask_channel[y_start:y_end,:]] = Colors.LANE_MARKING[c]
        
        # Add some "signs" as colored rectangles with shapes
        sign_positions = [
            (WINDOW_WIDTH//4, WINDOW_HEIGHT//4, "STOP"),
            (3*WINDOW_WIDTH//4, WINDOW_HEIGHT//3, "YIELD")
        ]
        
        for x, y, label in sign_positions:
            if label == "STOP":
                # Red octagon for stop sign
                cv.circle(img, (x, y), 30, (0, 0, 200), -1)
                cv.putText(img, "STOP", (x-25, y+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            elif label == "YIELD":
                # Yellow triangle for yield
                triangle_pts = np.array([
                    [x, y-30], [x-25, y+15], [x+25, y+15]
                ], np.int32)
                cv.fillPoly(img, [triangle_pts], (0, 215, 255))
                
        # Add a traffic light
        tl_state = (self.frame_count // 50) % 3  # 0=red, 1=yellow, 2=green
        tl_colors = [Colors.RED_LIGHT, Colors.YELLOW_LIGHT, Colors.GREEN_LIGHT]
        
        # Traffic light housing
        cv.rectangle(img, (700, 120), (740, 200), (40, 40, 40), -1)
        
        # Individual lights (highlight the active one)
        for i, color in enumerate(tl_colors):
            y_pos = 140 + i * 25
            intensity = 255 if i == tl_state else 50
            adjusted_color = tuple(min(c * intensity // 255, 255) for c in color)
            cv.circle(img, (720, y_pos), 10, adjusted_color, -1)
        
        return img

class Detectors:
    """Collection of mock detection algorithms"""
    
    @staticmethod
    def lane_detection_hough(image: np.ndarray) -> List[Tuple]:
        """Return simulated lane detection results"""
        # Create perspective-aware lines
        hw, hh = WINDOW_WIDTH//2, WINDOW_HEIGHT//2
        lines = [
            # Left lane
            ((hw-150, WINDOW_HEIGHT), (hw-50, hh)),
            # Right lane
            ((hw+150, WINDOW_HEIGHT), (hw+50, hh)),
            # Center line segments
            ((hw-10, WINDOW_HEIGHT-100), (hw-5, WINDOW_HEIGHT-200)),
            ((hw-5, WINDOW_HEIGHT-250), (hw, WINDOW_HEIGHT-350)),
        ]
        return lines

    @staticmethod
    def lane_detection_ml(image: np.ndarray) -> List[Tuple]:
        """Return simulated ML lane detection results"""
        # Similar to Hough but smoother/more accurate
        hw, hh = WINDOW_WIDTH//2, WINDOW_HEIGHT//2
        lines = [
            # Left lane (smoother)
            ((hw-150, WINDOW_HEIGHT), (hw-60, hh+50)),
            ((hw-60, hh+50), (hw-40, hh)),
            # Right lane (smoother)
            ((hw+150, WINDOW_HEIGHT), (hw+60, hh+50)),
            ((hw+60, hh+50), (hw+40, hh)),
            # Center line (continuous)
            ((hw, WINDOW_HEIGHT), (hw, hh)),
        ]
        return lines

    @staticmethod
    def sign_detection(image: np.ndarray) -> List[Dict]:
        """Return simulated sign detection results"""
        signs = [
            {'bbox': (WINDOW_WIDTH//4-30, WINDOW_HEIGHT//4-30, 60, 60), 'label': 'Stop', 'confidence': 0.92},
            {'bbox': (3*WINDOW_WIDTH//4-25, WINDOW_HEIGHT//3-30, 50, 60), 'label': 'Yield', 'confidence': 0.87}
        ]
        return signs

    @staticmethod
    def traffic_light(image: np.ndarray, frame_count: int) -> List[Dict]:
        """Return simulated traffic light detection results"""
        tl_state = (frame_count // 50) % 3  # 0=red, 1=yellow, 2=green
        states = ['Red', 'Yellow', 'Green']
        lights = [
            {'bbox': (700, 120, 40, 80), 'state': states[tl_state], 'confidence': 0.95}
        ]
        return lights

#############################################
# UI COMPONENTS
#############################################
class UIManager:
    """Manages all UI windows and visualization"""
    def __init__(self):
        self.scene_generator = SceneGenerator()
        self.detectors = Detectors()
        
        # Store photo references (must be stored to prevent garbage collection)
        self.photos = {}
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Carla Simulation (TEST MODE)")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+0+0")
        
        # Create additional windows
        self.windows = {
            'main': self.root,
            'lane_hough': self._create_window("Lane Detection (Hough)", 
                                            WINDOW_WIDTH, 0),
            'lane_ml': self._create_window("Lane Detection (ML)", 
                                         0, WINDOW_HEIGHT),
            'sign_detection': self._create_window("Sign & Traffic Light Detection", 
                                               WINDOW_WIDTH, WINDOW_HEIGHT)
        }
        
        # Create canvas for each window
        self.canvases = {}
        for name, window in self.windows.items():
            canvas = tk.Canvas(window, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
            canvas.pack(fill="both", expand=True)
            self.canvases[name] = canvas
            
        # Set up close handlers
        for window in self.windows.values():
            window.protocol("WM_DELETE_WINDOW", self.exit_handler)
    
    def _create_window(self, title, x, y):
        """Helper to create a new window at the specified position"""
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")
        return window
    
    def _numpy_to_tkinter(self, array):
        """Convert a numpy array to a tkinter PhotoImage"""
        # Make sure the array is uint8
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        
        # Convert BGR to RGB for PIL
        rgb_array = cv.cvtColor(array, cv.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        img = Image.fromarray(rgb_array)
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image=img)
        return photo
    
    def _draw_dashboard(self, image, vehicle_state):
        """Draw a dashboard with speed and steering information"""
        # Draw dashboard background
        cv.rectangle(image, (50, WINDOW_HEIGHT-100), (WINDOW_WIDTH-50, WINDOW_HEIGHT-20), 
                    (30, 30, 30), -1)
        cv.rectangle(image, (60, WINDOW_HEIGHT-90), (WINDOW_WIDTH-60, WINDOW_HEIGHT-30), 
                    (50, 50, 50), -1)
        
        # Speed display
        speed_text = f"Speed: {vehicle_state.speed:.1f} km/h"
        cv.putText(image, speed_text, (100, WINDOW_HEIGHT-60), 
                 cv.FONT_HERSHEY_SIMPLEX, 0.7, Colors.TEXT_COLOR, 2)
        
        # Steering display
        steer_text = f"Steering: {vehicle_state.steering_angle:.1f}°"
        cv.putText(image, steer_text, (400, WINDOW_HEIGHT-60), 
                 cv.FONT_HERSHEY_SIMPLEX, 0.7, Colors.TEXT_COLOR, 2)
        
        # Draw steering wheel indicator
        center_x, center_y = WINDOW_WIDTH-120, WINDOW_HEIGHT-60
        cv.circle(image, (center_x, center_y), 30, (100, 100, 100), 2)
        
        # Draw indicator line based on steering angle
        angle_rad = np.radians(vehicle_state.steering_angle)
        end_x = int(center_x - 25 * np.sin(angle_rad))
        end_y = int(center_y - 25 * np.cos(angle_rad))
        cv.line(image, (center_x, center_y), (end_x, end_y), (0, 255, 255), 3)
        
        return image
    
    def _draw_detection_info(self, image, detections, detection_type):
        """Add detection information to the image based on type"""
        if detection_type == "sign":
            for sign in detections:
                x, y, w, h = sign['bbox']
                cv.rectangle(image, (x, y), (x+w, y+h), Colors.SIGN_BOX, 2)
                label = f"{sign['label']} ({sign['confidence']:.2f})"
                cv.putText(image, label, (x, y-10), 
                         cv.FONT_HERSHEY_SIMPLEX, 0.6, Colors.SIGN_BOX, 2)
        
        elif detection_type == "traffic_light":
            for light in detections:
                x, y, w, h = light['bbox']
                # Color based on state
                if light['state'] == 'Red':
                    color = Colors.RED_LIGHT
                elif light['state'] == 'Yellow':
                    color = Colors.YELLOW_LIGHT
                else:
                    color = Colors.GREEN_LIGHT
                
                cv.rectangle(image, (x, y), (x+w, y+h), color, 2)
                label = f"{light['state']} ({light['confidence']:.2f})"
                cv.putText(image, label, (x, y-10), 
                         cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        elif detection_type == "hough_lane":
            for line in detections:
                cv.line(image, line[0], line[1], Colors.HOUGH_LANE, 2)
        
        elif detection_type == "ml_lane":
            for line in detections:
                cv.line(image, line[0], line[1], Colors.ML_LANE, 2)
        
        return image
    
    def update_displays(self):
        """Update all visualization windows"""
        # Update vehicle state
        self.scene_generator.update_vehicle_state()
        
        # Generate base image
        base_image = self.scene_generator.generate_image()
        base_with_dash = self._draw_dashboard(base_image.copy(), 
                                             self.scene_generator.vehicle_state)
        
        # Run detections
        hough_lines = self.detectors.lane_detection_hough(base_image)
        ml_lines = self.detectors.lane_detection_ml(base_image)
        signs = self.detectors.sign_detection(base_image)
        lights = self.detectors.traffic_light(base_image, 
                                             self.scene_generator.frame_count)
        
        # Prepare visualization for each window
        lane_hough_image = base_with_dash.copy()
        lane_ml_image = base_with_dash.copy()
        sign_traffic_image = base_with_dash.copy()
        
        # Draw detections
        lane_hough_image = self._draw_detection_info(lane_hough_image, hough_lines, "hough_lane")
        lane_ml_image = self._draw_detection_info(lane_ml_image, ml_lines, "ml_lane")
        sign_traffic_image = self._draw_detection_info(sign_traffic_image, signs, "sign")
        sign_traffic_image = self._draw_detection_info(sign_traffic_image, lights, "traffic_light")
        
        # Update all windows
        self._update_canvas('main', base_with_dash)
        self._update_canvas('lane_hough', lane_hough_image)
        self._update_canvas('lane_ml', lane_ml_image)
        self._update_canvas('sign_detection', sign_traffic_image)
    
    def _update_canvas(self, name, image):
        """Update a specific canvas with the given image"""
        photo = self._numpy_to_tkinter(image)
        self.photos[name] = photo  # Store reference to prevent garbage collection
        self.canvases[name].create_image(0, 0, image=photo, anchor="nw")
    
    def exit_handler(self):
        """Handle exit for all windows"""
        print("Exiting test mode...")
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the simulation and UI update loop"""
        def update_loop():
            self.update_displays()
            self.root.after(int(1000/FPS), update_loop)
        
        print("Starting UI test...")
        update_loop()
        
        # Run the main loop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Keyboard interrupt received")
        finally:
            self.exit_handler()

#############################################
# MAIN PROGRAM
#############################################
if __name__ == "__main__":
    ui = UIManager()
    ui.run()