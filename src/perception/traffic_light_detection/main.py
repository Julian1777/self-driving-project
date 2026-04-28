from src.perception.traffic_light_detection.detect_classify import detect_traffic_lights

default_threshold = 0.2

def process_frame(img, confidence_threshold=default_threshold, draw_detections=True, model=None):
    """
    Process traffic light detection with config-driven threshold.
    Args:
        img: Input image (numpy array)
        perception_cfg: Perception configuration dictionary
        draw_detections: Whether to draw bounding boxes on the image
        model: Pre-loaded traffic light detection model
    Returns:
        tuple: (detections, result_img)
    """
    
    try:
        detections = detect_traffic_lights(img, model=model)
        
        if not detections:
            detections = []
        
        filtered_detections = [det for det in detections if det['confidence'] >= confidence_threshold]
        
        result_img = img
        
        if draw_detections:
            import cv2
            result_img = img.copy()
            for det in filtered_detections:
                x1, y1, x2, y2 = det['bbox']
                
                class_name = det.get('class', 'Unknown')
                confidence = det.get('confidence', 0.0)
                
                label = f"{class_name} ({confidence:.2f})"
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        return filtered_detections, result_img
    except Exception as e:
        print(f"Error processing traffic light frame: {e}")
        return [], img
