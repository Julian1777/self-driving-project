import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.perception.lane_detection.cv.thresholding import apply_thresholds_with_voting
from src.perception.lane_detection.cv.perspective import debug_perspective_live, get_src_points, perspective_warp
from src.perception.lane_detection.cv.lane_finder import get_histogram, detect_lane_type, fill_dashed_lane_gaps
from src.perception.lane_detection.metrics import calculate_curvature_and_deviation, process_deviation
from src.perception.lane_detection.visualization import draw_multiple_lanes_overlay, add_text_overlay

from src.perception.lane_detection.cv.multi_lane.multi_lane_finder import detect_multiple_lanes
from src.perception.lane_detection.cv.multi_lane.lane_selector import get_current_lane
from src.perception.lane_detection.ufld.main import UFLDv2Inference
from config.config import MODELS_DIR

import numpy as np
import cv2
from pathlib import Path


_ufld_model = None


def _get_ufld_model():
    """Lazy-load UFLDv2 model on first use."""
    global _ufld_model
    if _ufld_model is None:
        model_path = MODELS_DIR / "ufld" / "culane_res18.pth"
        config_path = Path(__file__).parent.parent.parent.parent / "ufldv2" / "configs" / "culane_res18.py"

        try:
            _ufld_model = UFLDv2Inference(str(model_path), str(config_path))
            print("[MAIN] UFLDv2 model loaded successfully")
        except Exception as e:
            print(f"[MAIN] Failed to load UFLDv2 model: {e}")
            _ufld_model = False

    return _ufld_model if _ufld_model is not False else None


def process_frame_cv(img, speed=0, previous_steering=0, debug_display=False, perspective_debug_display=False, calibration_data=None, vehicle_model='q8_andronisk', num_lanes=3):
        
    previous_fit = None
    confidence = 0.0
    try:

        src_points = get_src_points(img.shape, speed, previous_steering, vehicle_model=vehicle_model, calibration_data=calibration_data)

        ufld_model = _get_ufld_model()
        ufld_mask = None
        if ufld_model is not None:
            try:
                ufld_mask = ufld_model.infer(img)
            except Exception as e:
                print(f"[MAIN] UFLDv2 inference failed: {e}")
                ufld_mask = None

        # Apply thresholding to full image with UFLD as voting feature
        binary_image, avg_brightness = apply_thresholds_with_voting(
            img,
            src_points=None,
            debug_display=debug_display,
            ufld_mask=ufld_mask
        )

        if debug_display:
            binary_display_resized = cv2.resize(binary_image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Binary Image', binary_display_resized)

        # Apply roi mask
        mask = np.zeros(binary_image.shape[:2], dtype=np.uint8)
        src_poly = np.array(src_points, dtype=np.int32)
        cv2.fillPoly(mask, [src_poly], 1)
        binary_image = binary_image * mask
        
        # perspective warp for BEV
        binary_warped, Minv = perspective_warp(binary_image, speed=speed, calibration_data=calibration_data, vehicle_model=vehicle_model)

        is_dashed_lane = detect_lane_type(binary_warped)
        if is_dashed_lane:
            binary_warped = fill_dashed_lane_gaps(binary_warped, gap_size=20)
        
        if perspective_debug_display:
            debug_perspective_live(img, speed, previous_steering=0, vehicle_model=vehicle_model, calibration_data=calibration_data)
        
        if debug_display:
            warped_display = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            warped_display_resized = cv2.resize(warped_display, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Warped Binary CV', warped_display_resized)

        lanes = None
        detected_num_lanes = num_lanes

        for attempt_lanes in [num_lanes, num_lanes-1, 2, 1]:
            if attempt_lanes < 1:
                break
            lanes = detect_multiple_lanes(binary_warped, num_lanes=attempt_lanes, debug=debug_display)
            if lanes is not None:
                detected_num_lanes = attempt_lanes
                print(f"[MULTI_LANE] Successfully detected {detected_num_lanes} lanes")
                break
            else:
                print(f"[MULTI_LANE] failed to detect {attempt_lanes} lanes, trying with {attempt_lanes-1} lanes")
        

        if lanes is None:
            print("[Multi-Lane] detection failed")
            result = img.copy()
            metrics = {
                'current_lane': None,
                'all_lanes': None,
                'confidence': 0.0,
                'error': 'Multi-lane detection failed'
            }
            return result, metrics, 0.0
        
        # show lanes in warped space
        if debug_display:
            warped_lane_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            
            for lane_idx, lane in enumerate(lanes):
                ploty_viz = lane['ploty'].astype(np.int32)
                left_fitx_viz = lane['left_fitx'].astype(np.int32)
                right_fitx_viz = lane['right_fitx'].astype(np.int32)
                color = colors[lane_idx % len(colors)]
                
                # clip
                h, w = binary_warped.shape[:2]
                left_fitx_viz = np.clip(left_fitx_viz, 0, w - 1)
                right_fitx_viz = np.clip(right_fitx_viz, 0, w - 1)
                ploty_viz = np.clip(ploty_viz, 0, h - 1)
                
                if len(left_fitx_viz) > 0 and len(ploty_viz) > 0:
                    left_points = np.array([np.transpose(np.vstack([left_fitx_viz, ploty_viz]))], dtype=np.int32)
                    cv2.polylines(warped_lane_img, left_points, isClosed=False, color=color, thickness=2)
                
                if len(right_fitx_viz) > 0 and len(ploty_viz) > 0:
                    right_points = np.array([np.transpose(np.vstack([right_fitx_viz, ploty_viz]))], dtype=np.int32)
                    cv2.polylines(warped_lane_img, right_points, isClosed=False, color=color, thickness=2)

            warped_lanes_display_resized = cv2.resize(warped_lane_img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Warped Lanes Detected', warped_lanes_display_resized)
        
        # classify lanes and get current lane
        warped_width = binary_warped.shape[1]
        lane_info = get_current_lane(lanes, vehicle_center=None, image_width=img.shape[1], warped_width=warped_width, debug=debug_display)
        current_lane_data = lane_info['current_lane']
        all_lanes = lane_info['all_lanes']

        # extract left and right fitx
        # Use current lane or fallback to first
        if current_lane_data:
            lane_data = current_lane_data['lane_data']
            if debug_display:
                print(f"[MAIN] Using selected lane: id={current_lane_data['lane_id']}, class={current_lane_data['lane_class']}, pos_in_lane={current_lane_data['position_in_lane']:.2f}")
        else:
            first_lane = list(all_lanes.values())[0] if all_lanes else None
            if first_lane:
                lane_data = first_lane['lane_data']
                print("[MAIN] No lane selected, using first detected lane")
            else:
                print("[MAIN] No lane data available")
                result = img.copy()
                metrics = {
                    'current_lane': None,
                    'all_lanes': None,
                    'confidence': 0.0,
                    'error': 'No lane data'
                }
                return result, metrics, 0.0
        
        # extract data
        ploty = lane_data['ploty']
        left_fitx = lane_data['left_fitx']
        right_fitx = lane_data['right_fitx']
        left_fit = lane_data['left_fit']
        right_fit = lane_data['right_fit']
        
        # drawl all detected lanes
        result = draw_multiple_lanes_overlay(img, binary_warped, Minv, lanes, all_lanes_classified=all_lanes)
        
        if debug_display:
            warped_lane_img_debug = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            h, w = binary_warped.shape[:2]
            
            for lane_idx, lane in enumerate(lanes):
                left_fitx_viz = lane['left_fitx'].astype(np.int32)
                right_fitx_viz = lane['right_fitx'].astype(np.int32)
                ploty_viz = lane['ploty'].astype(np.int32)
                lane_id = lane['lane_id']
                
                left_fitx_viz = np.clip(left_fitx_viz, 0, w - 1)
                right_fitx_viz = np.clip(right_fitx_viz, 0, w - 1)
                ploty_viz = np.clip(ploty_viz, 0, h - 1)
                
                is_current = current_lane_data and lane_id == current_lane_data['lane_id']
                color = (0, 255, 0) if is_current else (255, 255, 255)
                thickness = 3 if is_current else 2
                
                if len(left_fitx_viz) > 0 and len(ploty_viz) > 0:
                    left_points = np.array([np.transpose(np.vstack([left_fitx_viz, ploty_viz]))], dtype=np.int32)
                    cv2.polylines(warped_lane_img_debug, left_points, isClosed=False, color=color, thickness=thickness)
                
                if len(right_fitx_viz) > 0 and len(ploty_viz) > 0:
                    right_points = np.array([np.transpose(np.vstack([right_fitx_viz, ploty_viz]))], dtype=np.int32)
                    cv2.polylines(warped_lane_img_debug, right_points, isClosed=False, color=color, thickness=thickness)
                
                if len(left_fitx_viz) > 0 and len(right_fitx_viz) > 0:
                    label_x = int((left_fitx_viz[-1] + right_fitx_viz[-1]) / 2)
                    label_y = int(ploty_viz[-1])
                    label = f"L{lane_id}"
                    if is_current:
                        label += " *TRACKED*"
                    cv2.putText(warped_lane_img_debug, label, (label_x - 30, label_y + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.line(warped_lane_img_debug, (w//2, 0), (w//2, h), (0, 0, 255), 2)
            cv2.putText(warped_lane_img_debug, "VEHICLE CENTER", (w//2 + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            warped_lanes_display_resized = cv2.resize(warped_lane_img_debug, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Warped Lanes Detected', warped_lanes_display_resized)
        
        # Calculate metrics
        current_fit = (left_fit, right_fit)
        metrics_result = calculate_curvature_and_deviation(ploty, left_fitx, right_fitx, binary_warped, original_image_width=img.shape[1])

        previous_fit = current_fit

        if metrics_result is None or (isinstance(metrics_result, tuple) and all(x is None for x in metrics_result)):
            left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = None, None, None, None, None, None
            print("[METRICS] Lane detection metrics calculation returned None values")
        else:
            if len(metrics_result) == 6:
                left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = metrics_result
            elif len(metrics_result) == 5:
                left_curverad, right_curverad, deviation, lane_center, vehicle_center = metrics_result
                lane_width = None
            else:
                left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = None, None, None, None, None, None

        smoothed_deviation, effective_deviation = process_deviation(deviation if deviation is not None else 0.0)
        
        if debug_display:
            print(f"[METRICS] raw_dev={deviation:.4f}m, smoothed_dev={smoothed_deviation:.4f}m, effective_dev={effective_deviation:.4f}m")
            print(f"[METRICS] lane_center={lane_center:.1f}px, vehicle_center={vehicle_center:.1f}px, lane_width={lane_width:.1f}px")
            if left_curverad and right_curverad:
                print(f"[METRICS] curvature: left={left_curverad:.1f}m, right={right_curverad:.1f}m")
        
        result = add_text_overlay(result, left_curverad, right_curverad, deviation, avg_brightness, speed, confidence=confidence)
        
        if debug_display:
            debug_y = 150
            cv2.putText(result, f"Raw Dev: {deviation:.3f}m", (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(result, f"Smoothed: {smoothed_deviation:.3f}m", (10, debug_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(result, f"Effective: {effective_deviation:.3f}m", (10, debug_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if current_lane_data:
                cv2.putText(result, f"Lane: {current_lane_data['lane_class']} (ID:{current_lane_data['lane_id']}) pos={current_lane_data['position_in_lane']:.2f}", 
                           (10, debug_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f"Curv L:{left_curverad:.0f}m R:{right_curverad:.0f}m" if left_curverad else "Curv: N/A", 
                       (10, debug_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        metrics = {
            'left_curverad': left_curverad,
            'right_curverad': right_curverad,
            'deviation': deviation,  # raw deviation
            'smoothed_deviation': smoothed_deviation,
            'effective_deviation': effective_deviation,
            'lane_center': lane_center,
            'vehicle_center': vehicle_center,
            'lane_width': lane_width,
            'confidence': confidence,
            'current_lane': current_lane_data,
            'all_lanes': all_lanes,
            'detected_num_lanes': detected_num_lanes
        }
        
        return result, metrics, confidence
        
    except Exception as e:
        print(f"Lane detection error CV: {e}")
        import traceback
        traceback.print_exc()
        result = img.copy()
        metrics = {
            'left_curverad': 0,
            'right_curverad': 0,
            'deviation': 0,
            'lane_center': 0,
            'vehicle_center': 0,
            'lane_width': 0,
            'confidence': 0,
            'current_lane': None,
            'all_lanes': None,
            'error': str(e)
        }
        return result, metrics, 0.0