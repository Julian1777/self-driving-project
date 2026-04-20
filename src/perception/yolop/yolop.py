import sys
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from config.config import YOLOP_MODEL

# Try importing YOLOP libraries
try:
    from yolop.lib.config import cfg
    from yolop.lib.models import get_net
    from yolop.lib.core.general import non_max_suppression, scale_coords
    from yolop.lib.core.postprocess import morphological_process, connect_lane
except ImportError:
    try:
        from lib.config import cfg
        from lib.models import get_net
        from lib.core.general import non_max_suppression, scale_coords
        from lib.core.postprocess import morphological_process, connect_lane
    except ImportError as e:
        print(f"Warning: YOLOP utilities not found: {e}")

def get_models_dict():
    try:
        main_module = sys.modules['__main__']
        if hasattr(main_module, 'MODELS'):
            return main_module.MODELS
        return None
    except:
        return None

def detect_yolop(frame_bgr, confidence_threshold=0.3):
    models_dict = get_models_dict()
    
    if models_dict is not None and 'yolop_model' in models_dict:
        model = models_dict['yolop_model']
        device = models_dict.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        transforms_comp = models_dict.get('yolop_transforms')
    else:
        print("Warning: Loading YOLOP model from scratch - slower!")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = get_net(cfg)
        checkpoint = torch.load(str(YOLOP_MODEL), map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        transforms_comp = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    img_ori = frame_bgr
    img_resized = cv2.resize(img_ori, (640, 640), interpolation=cv2.INTER_LINEAR)
    img_tensor = transforms_comp(img_resized).to(device)
    img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = model(img_tensor)
        
    inf_out, _ = det_out
    det_pred = non_max_suppression(
        inf_out,
        conf_thres=confidence_threshold,
        iou_thres=0.45,
        classes=None,
        agnostic=False
    )
    det = det_pred[0]
    
    if len(det):
        det[:, :4] = scale_coords((640, 640), det[:, :4], img_ori.shape).round()
        
    da_seg_out = torch.softmax(da_seg_out, dim=1)
    da_seg_mask = torch.argmax(da_seg_out, dim=1)
    da_seg_mask = da_seg_mask.squeeze().cpu().numpy().astype(np.uint8)
    
    ll_seg_out = torch.softmax(ll_seg_out, dim=1)
    ll_seg_mask = torch.argmax(ll_seg_out, dim=1)
    ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy().astype(np.uint8)
    
    da_seg_mask = da_seg_mask.astype(np.uint8)
    try:
        da_seg_mask = morphological_process(da_seg_mask)
    except NameError:
        pass
        
    ll_seg_mask = ll_seg_mask.astype(np.uint8)
    try:
        ll_seg_mask = connect_lane(ll_seg_mask)
    except NameError:
        pass
        
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
            
    return formatted_detections, da_seg_mask, ll_seg_mask
