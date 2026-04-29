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
    Uses same multi-lane detection algorithm as CV pipeline for consistency.
    
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
        tuple: (detections, drivable_area, lanes)
              where lanes is list of detected lane dicts with ploty, left_fitx, right_fitx
    """
    try:
        detections, drivable_area, lane_mask = detect_yolop(img, confidence_threshold, model=model, device=device, transforms=transforms)
        
        if not detections:
            detections = []
        
        lanes = None
        if lane_mask is not None and lane_mask.size > 0:
            try:
                print(f"[YOLOP] Lane mask shape: {lane_mask.shape}, dtype: {lane_mask.dtype}, unique values: {np.unique(lane_mask)}")
                
                lane_mask_binary = (lane_mask > 0).astype(np.uint8) * 255
                
                src_points = get_src_points(img.shape, speed, 0, vehicle_model=vehicle_model, calibration_data=calibration_data)
                
                roi_mask = np.zeros(lane_mask_binary.shape[:2], dtype=np.uint8)
                src_poly = np.array(src_points, dtype=np.int32)
                cv2.fillPoly(roi_mask, [src_poly], 255)
                lane_mask_roi = cv2.bitwise_and(lane_mask_binary, roi_mask)
                
                print(f"[YOLOP] Lane mask after ROI: min={np.min(lane_mask_roi)}, max={np.max(lane_mask_roi)}, mean={np.mean(lane_mask_roi):.1f}")
                
                lane_warped, _ = perspective_warp(
                    lane_mask_roi, 
                    speed=speed, 
                    calibration_data=calibration_data, 
                    vehicle_model=vehicle_model,
                    is_mask=True
                )
                
                print(f"[YOLOP] Warped lane mask: shape={lane_warped.shape}, min={np.min(lane_warped)}, max={np.max(lane_warped)}")
                
                if lane_warped.dtype != np.uint8:
                    lane_warped = lane_warped.astype(np.uint8)
                
                _, lane_warped_binary = cv2.threshold(lane_warped, 127, 255, cv2.THRESH_BINARY)
                
                print(f"[YOLOP] Binary warped mask: min={np.min(lane_warped_binary)}, max={np.max(lane_warped_binary)}, non-zero pixels: {np.count_nonzero(lane_warped_binary)}")
                
                detected_num_lanes = 3
                for attempt_lanes in [3, 2, 1]:
                    lanes = detect_multiple_lanes(lane_warped_binary, num_lanes=attempt_lanes)
                    if lanes is not None:
                        detected_num_lanes = attempt_lanes
                        print(f"[YOLOP] Successfully detected {detected_num_lanes} lanes from mask")
                        break
                    else:
                        print(f"[YOLOP] Failed to detect {attempt_lanes} lanes, retrying with {attempt_lanes-1}...")
                
                if lanes is None:
                    print("[YOLOP] Lane detection from mask failed - no valid lane boundaries found")
            except Exception as e:
                print(f"[YOLOP] Error processing lane mask: {e}")
                lanes = None
            
        return detections, drivable_area, lanes
    except Exception as e:
        print(f"Error processing YOLOP frame: {e}")
        return [], None, None
