import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.perception.yolop.yolop import detect_yolop
from src.perception.lane_detection.cv.perspective import perspective_warp, get_src_points
from src.perception.lane_detection.cv.multi_lane.multi_lane_finder import detect_multiple_lanes

default_threshold = 0.3

def process_frame(img, confidence_threshold=default_threshold, model=None, device=None, transforms=None, speed=0, calibration_data=None, vehicle_model='etk800'):
    """
    Process frame with YOLOP to get drivable area and detected lanes.
    
    Args:
        img: Input image (BGR numpy array)
        confidence_threshold: threshold
        model: Pre-loaded YOLOP model
        device: Torch device
        transforms: Image transforms
        speed: Vehicle speed for perspective calibration
        calibration_data: Perspective transform calibration
        vehicle_model: Vehicle model for calibration
    
    Returns:
        tuple: (detections, drivable_area, lane_mask)
    """
    try:
        detections, drivable_area, lane_mask = detect_yolop(img, confidence_threshold, model=model, device=device, transforms=transforms)
        
        if not detections:
            detections = []
            
        if lane_mask is not None and lane_mask.size > 0:
            lane_mask = (lane_mask > 0).astype(np.uint8)
            
        return detections, drivable_area, lane_mask
    except Exception as e:
        print(f"Error processing YOLOP frame: {e}")
        return [], None, None
