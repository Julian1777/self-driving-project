import os
import sys
import numpy as np
import base64
from flask import Flask, request, jsonify
from ultralytics import YOLO

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core traffic light detection logic from source
from src.perception.traffic_light_detection.main import process_frame

app = Flask(__name__)

MODELS = {}

def load_models():
    """Pre-load detection model at startup"""
    global MODELS
    model_path = os.getenv('MODEL_PATH')
    if not model_path:
        print("[Traffic Light Detection Service] ERROR: MODEL_PATH environment variable not set")
        return False
    
    if not os.path.exists(model_path):
        print(f"[Traffic Light Detection Service] ERROR: Model file not found at {model_path}")
        return False
    
    try:
        print(f"[Traffic Light Detection Service] Loading model from {model_path}...")
        MODELS['traffic_light'] = YOLO(model_path)
        # Make MODELS accessible to imported modules
        sys.modules['__main__'].MODELS = MODELS
        print("[Traffic Light Detection Service] Model loaded successfully")
        return True
    except Exception as e:
        print(f"[Traffic Light Detection Service] ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/process', methods=['POST'])
def process_detection():
    """
    Process a camera frame for traffic light detection and classification
    
    Expected JSON payload:
    {
        "frame": [list of pixel values],
        "frame_shape": [height, width, channels],
        "confidence_threshold": 0.2,
        "frame_id": "frame_123"
    }
    """
    data = None
    try:
        data = request.get_json()
        
        # Decode frame from request (base64 encoded)
        frame_b64 = data['frame']
        frame_bytes = base64.b64decode(frame_b64)
        frame_shape = data.get('frame_shape', [1080, 1920, 3])
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_shape)
        
        confidence_threshold = data.get('confidence_threshold', 0.2)
        frame_id = data.get('frame_id', 'unknown')
        
        print(f"[Traffic Light Detection Service] Processing frame {frame_id}: {frame.shape}, threshold: {confidence_threshold}")
        
        # Convert RGB to BGR for YOLO (expects OpenCV format)
        import cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Call core inference logic from source
        detections, result_img = process_frame(
            frame_bgr,
            confidence_threshold=confidence_threshold,
            draw_detections=False
        )
        
        # Format detections for response
        formatted_detections = []
        if detections:
            for det in detections:
                bbox = det.get('bbox', [0, 0, 0, 0])
                # Convert bbox to Python ints (handle numpy int64)
                bbox_list = [int(x) for x in bbox]
                formatted_detections.append({
                    'class': str(det.get('class', 'unknown')),
                    'confidence': float(det.get('confidence', 0.0)),
                    'bbox': bbox_list
                })
        
        response = {
            'frame_id': frame_id,
            'service': 'traffic_light_detection',
            'status': 'success',
            'detections': formatted_detections,
            'detection_count': len(formatted_detections)
        }
        
        return response, 200
        
    except Exception as e:
        print(f"[Traffic Light Detection Service] Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        frame_id = data.get('frame_id', 'unknown') if data else 'unknown'
        return {'status': 'error', 'message': str(e), 'frame_id': frame_id}, 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_ready = 'traffic_light' in MODELS and MODELS['traffic_light'] is not None
    return {
        'status': 'healthy' if model_ready else 'initializing',
        'service': 'traffic_light_detection',
        'model_ready': model_ready
    }, 200 if model_ready else 503

if __name__ == '__main__':
    if not load_models():
        print("[Traffic Light Detection Service] Failed to load models. Exiting.")
        sys.exit(1)
    print("[Traffic Light Detection Service] Starting on 0.0.0.0:6777")
    app.run(host='0.0.0.0', port=6777, debug=False)
