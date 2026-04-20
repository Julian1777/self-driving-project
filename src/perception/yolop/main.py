import cv2

from src.perception.yolop.yolop import detect_yolop

default_threshold = 0.3

def process_frame(img, confidence_threshold=default_threshold):
    """
    Process frame with YOLOP to get drivable area and lane lines.
    Args:
        img: Input image (BGR numpy array)
        confidence_threshold: threshold
    Returns:
        tuple: (detections, drivable_area, lane_lines)
    """
    try:
        detections, drivable_area, lane_lines = detect_yolop(img, confidence_threshold)
        
        if not detections:
            detections = []
            
        return detections, drivable_area, lane_lines
    except Exception as e:
        print(f"Error processing YOLOP frame: {e}")
        return [], None, None
