import os
import sys
import numpy as np
import torch
import cv2
import base64
from flask import Flask, request, jsonify
import torchvision.transforms as transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, '/app/yolop_repo')

# Import YOLOP model utilities (from inference example)
try:
    from lib.config import cfg
    from lib.models import get_net
    from lib.core.general import non_max_suppression, scale_coords
    from lib.core.postprocess import morphological_process, connect_lane
    print("[YOLOP Service] Successfully imported YOLOP libraries")
except ImportError as e:
    print(f"[YOLOP Service] Warning: YOLOP utilities not found: {e}")
    print("[YOLOP Service] Trying alternative import paths...")
    try:
        from yolop.lib.config import cfg
        from yolop.lib.models import get_net
        from yolop.lib.core.general import non_max_suppression, scale_coords
        from yolop.lib.core.postprocess import morphological_process, connect_lane
        print("[YOLOP Service] Successfully imported YOLOP libraries from yolop package")
    except ImportError as e2:
        print(f"[YOLOP Service] Alternative import also failed: {e2}")
        print("[YOLOP Service] Starting without YOLOP utilities - will fail at inference")

app = Flask(__name__)

MODELS = {}
DEVICE = None
TRANSFORMS = None


def load_models():
    """Initialize YOLOP model"""
    global MODELS, DEVICE, TRANSFORMS
    
    model_path = os.getenv('MODEL_PATH')
    if not model_path:
        print("[YOLOP Service] ERROR: MODEL_PATH environment variable not set")
        return False
    
    if not os.path.exists(model_path):
        print(f"[YOLOP Service] ERROR: Model file not found at {model_path}")
        return False
    
    try:
        # Setup device
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"[YOLOP Service] Using device: {DEVICE}")
        
        # Load model
        model = get_net(cfg)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(DEVICE)
        model.eval()
        
        MODELS['model'] = model
        
        # Setup transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        sys.modules['__main__'].MODELS = MODELS
        
        print(f"[YOLOP Service] Model loaded from {model_path}")
        print("[YOLOP Service] Ready to process frames")
        return True
        
    except Exception as e:
        print(f"[YOLOP Service] ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/process', methods=['POST'])
def process_yolop():
    """
    Process a camera frame with YOLOP for unified detection
    
    Expected JSON payload:
    {
        "frame": [list of pixel values],
        "frame_shape": [height, width, channels],
        "confidence_threshold": 0.3,
        "frame_id": "frame_123"
    }
    
    Returns:
    {
        "frame_id": "frame_123",
        "service": "yolop",
        "status": "success",
        "detections": [objects detected],
        "drivable_area": [segmentation mask],
        "lane_lines": [lane segmentation mask],
        "detection_count": N
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
        
        confidence_threshold = data.get('confidence_threshold', 0.3)
        frame_id = data.get('frame_id', 'unknown')
        
        print(f"[YOLOP Service] Processing frame {frame_id}: {frame.shape}, threshold: {confidence_threshold}")
        
        # Convert RGB to BGR for YOLOP (model expects OpenCV format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Preprocess image: resize to standard YOLOP input size (640x640)
        img_ori = frame_bgr
        # Resize to 640x640 which is standard YOLOP input size
        img_resized = cv2.resize(img_ori, (640, 640), interpolation=cv2.INTER_LINEAR)
        img_tensor = TRANSFORMS(img_resized).to(DEVICE)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension: (1, 3, 640, 640)
        
        # Run YOLOP inference
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = MODELS['model'](img_tensor)
        
        # Parse detection output
        inf_out, _ = det_out
        det_pred = non_max_suppression(
            inf_out,
            conf_thres=confidence_threshold,
            iou_thres=0.45,
            classes=None,
            agnostic=False
        )
        det = det_pred[0]  # First (only) batch
        
        # Scale detections back to original image size
        if len(det):
            # Scale from 640x640 back to original image dimensions
            det[:, :4] = scale_coords((640, 640), det[:, :4], img_ori.shape).round()
        
        # Parse drivable area segmentation
        da_seg_out = torch.softmax(da_seg_out, dim=1)
        da_seg_mask = torch.argmax(da_seg_out, dim=1)
        da_seg_mask = da_seg_mask.squeeze().cpu().numpy().astype(np.uint8)
        
        # Parse lane line segmentation
        ll_seg_out = torch.softmax(ll_seg_out, dim=1)
        ll_seg_mask = torch.argmax(ll_seg_out, dim=1)
        ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy().astype(np.uint8)
        
        da_seg_mask = da_seg_mask.astype(np.uint8)
        da_seg_mask = morphological_process(da_seg_mask)
        
        ll_seg_mask = ll_seg_mask.astype(np.uint8)
        ll_seg_mask = connect_lane(ll_seg_mask)
        
        original_h, original_w = img_ori.shape[:2]
        da_seg_mask = cv2.resize(da_seg_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        ll_seg_mask = cv2.resize(ll_seg_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        formatted_detections = []
        if len(det) > 0:
            for *xyxy, conf, cls in reversed(det):
                formatted_detections.append({
                    'class_id': int(cls.item()),
                    'confidence': float(conf.item()),
                    'bbox': [float(x.item()) for x in xyxy]
                })
        
        response = {
            'frame_id': frame_id,
            'service': 'yolop',
            'status': 'success',
            'detections': formatted_detections,
            'detection_count': len(formatted_detections),
            'drivable_area': da_seg_mask.tolist(),
            'lane_lines': ll_seg_mask.tolist(),
            'drivable_area_shape': list(da_seg_mask.shape),
            'lane_lines_shape': list(ll_seg_mask.shape)
        }
        
        return response, 200
        
    except Exception as e:
        print(f"[YOLOP Service] Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        frame_id = data.get('frame_id', 'unknown') if data else 'unknown'
        return {
            'status': 'error',
            'message': str(e),
            'frame_id': frame_id
        }, 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'yolop',
        'model_loaded': 'model' in MODELS,
        'device': str(DEVICE) if DEVICE else 'unknown'
    }, 200


if __name__ == '__main__':
    print("[YOLOP Service] Starting...")
    
    if not load_models():
        print("[YOLOP Service] Failed to load models, exiting")
        sys.exit(1)
    
    print("[YOLOP Service] Listening on 0.0.0.0:9777")
    app.run(host='0.0.0.0', port=9777, debug=False, threaded=True)
