import os
import sys
import numpy as np
import base64
from flask import Flask, request, jsonify
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core sign classification logic from source
from src.perception.sign_detection.detect_classify import sign_classification_only

app = Flask(__name__)

MODELS = {}

def load_models():
    """Pre-load classification model at startup"""
    global MODELS
    model_path = os.getenv('MODEL_PATH')
    if not model_path:
        print("[Sign Classification Service] ERROR: MODEL_PATH environment variable not set")
        return False
    
    if not os.path.exists(model_path):
        print(f"[Sign Classification Service] ERROR: Model file not found at {model_path}")
        return False
    
    try:
        print(f"[Sign Classification Service] Loading model from {model_path}...")
        MODELS['sign_classify'] = tf.keras.models.load_model(model_path)
        # Make MODELS accessible to imported modules
        sys.modules['__main__'].MODELS = MODELS
        print("[Sign Classification Service] Model loaded successfully")
        return True
    except Exception as e:
        print(f"[Sign Classification Service] ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/process', methods=['POST'])
def process_classification():
    """
    Process bounding boxes for traffic sign classification only
    
    Expected JSON payload:
    {
        "frame": [list of pixel values],
        "frame_shape": [height, width, channels],
        "bboxes": [[x1, y1, x2, y2], ...],  # Optional: pre-detected bboxes
        "frame_id": "frame_123"
    }
    """
    data = None
    try:
        data = request.get_json()
        
        # decode frame from request (base64 encoded)
        frame_b64 = data['frame']
        frame_bytes = base64.b64decode(frame_b64)
        frame_shape = data.get('frame_shape', [1080, 1920, 3])
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_shape)
        
        # try to use pre-detected bboxes if provided
        bboxes = data.get('bboxes', None)
        if bboxes:
            bboxes = [tuple(bbox) for bbox in bboxes]
        
        frame_id = data.get('frame_id', 'unknown')
        
        print(f"[Sign Classification Service] Processing frame {frame_id}: {frame.shape}, bboxes: {len(bboxes) if bboxes else 'auto-detect'}")
        
        # Convert RGB to BGR for TensorFlow/OpenCV (expects OpenCV format)
        import cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # call classification logic
        classifications = sign_classification_only(frame_bgr, bboxes=bboxes)
        
        # format classifications for response
        formatted_classifications = []
        if classifications:
            for cls in classifications:
                bbox = cls.get('bbox', [0, 0, 0, 0])
                # Convert bbox to Python ints (handle numpy int64)
                bbox_list = [int(x) for x in bbox]
                formatted_classifications.append({
                    'sign_type': str(cls.get('classification', 'unknown')),
                    'confidence': float(cls.get('classification_confidence', 0.0)),
                    'class_index': int(cls.get('class_index', -1)),
                    'bbox': bbox_list
                })
        
        response = {
            'frame_id': frame_id,
            'service': 'sign_classification',
            'status': 'success',
            'classifications': formatted_classifications,
            'classification_count': len(formatted_classifications)
        }
        
        return response, 200
        
    except Exception as e:
        print(f"[Sign Classification Service] Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        frame_id = data.get('frame_id', 'unknown') if data else 'unknown'
        return {'status': 'error', 'message': str(e), 'frame_id': frame_id}, 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_ready = 'sign_classify' in MODELS and MODELS['sign_classify'] is not None
    return {
        'status': 'healthy' if model_ready else 'initializing',
        'service': 'sign_classification',
        'model_ready': model_ready
    }, 200 if model_ready else 503

if __name__ == '__main__':
    if not load_models():
        print("[Sign Classification Service] Failed to load models. Exiting.")
        sys.exit(1)
    print("[Sign Classification Service] Starting on 0.0.0.0:8777")
    app.run(host='0.0.0.0', port=8777, debug=False)
