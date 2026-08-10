from src.perception.object_detection.object_detection import detect_objects
import cv2

default_threshold = 0.55

def process_frame(img, confidence_threshold=default_threshold, draw_detections=True, model=None):
    try:
        detections = detect_objects(img, model=model, confidence_threshold= confidence_threshold)

        if not detections:
            detections = []
        
        result_img = img
        
        if draw_detections:
            result_img = img.copy()
            for det in detections:
                bbox = det['bbox']
                label = f"{det['class']} ({det['confidence']:.2f})"
                cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(result_img, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections, result_img
    except Exception as e:
        print(f"Error processing obstacle frame: {e}")
        return [], img
