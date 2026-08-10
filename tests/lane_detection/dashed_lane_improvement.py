import cv2
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.perception.lane_detection.cv.thresholding import apply_thresholds_with_voting
from src.perception.lane_detection.cv.perspective import debug_perspective_live, get_src_points, perspective_warp
from src.perception.lane_detection.cv.lane_finder import get_histogram, sliding_window_search, detect_lane_type, fill_dashed_lane_gaps
from src.perception.lane_detection.cv.multi_lane.multi_lane_finder import detect_multiple_lanes, find_lane_boundaries
from src.perception.lane_detection.cv.multi_lane.lane_selector import get_current_lane
from src.perception.lane_detection.metrics import calculate_curvature_and_deviation



def process_frame_cv(img, speed=0, previous_steering=0, debug_display=False, perspective_debug_display=False, calibration_data=None, vehicle_model='q8_andronisk'):
        
    previous_fit = None
    confidence = 0.0
    try:
        # manual points
        src_points = np.float32([
            [50, 819],      # Bottom-Left
            [716, 675],    # Bottom-Right
            [1223, 670],   # Top-Right
            [1870, 803]    # Top-Left
        ])

        binary_image, avg_brightness = apply_thresholds_with_voting(
            img, 
            src_points=src_points,
            debug_display=debug_display,
            use_gradient=False
        )
        if debug_display:
            frame_with_roi = img.copy()
            roi_polygon = np.array(src_points, dtype=np.int32)
            cv2.polylines(frame_with_roi, [roi_polygon], True, (0, 255, 0), 3)
            cv2.imshow('Original Frame with ROI', frame_with_roi)
            
            cv2.imshow('Binary Image CV', binary_image*255 if binary_image.max()<=1 else binary_image)
            cv2.waitKey(1)
        
        mask = np.zeros(binary_image.shape[:2], dtype=np.uint8)
        src_poly = np.array(src_points, dtype=np.int32)
        cv2.fillPoly(mask, [src_poly], 1)
        binary_image = binary_image * mask
        
        img_size = (binary_image.shape[1], binary_image.shape[0])
        w, h = img_size
        
        dst_points = np.float32([
            [w*0.2, h],       # BL
            [w*0.8, h],       # BR
            [w*0.8, 0],       # TR
            [w*0.2, 0]        # TL
        ])
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        Minv = cv2.getPerspectiveTransform(dst_points, src_points)
        binary_warped = cv2.warpPerspective(binary_image, M, img_size, flags=cv2.INTER_LINEAR)
        
        binary_warped = cv2.rotate(binary_warped, cv2.ROTATE_90_CLOCKWISE)

        is_dashed_lane = detect_lane_type(binary_warped)
        if is_dashed_lane:
            binary_warped = fill_dashed_lane_gaps(binary_warped, gap_size=20)
        
        if debug_display:
            warped_display = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            cv2.imshow('Warped Binary CV', warped_display)
        
        histogram = get_histogram(binary_warped)
        lane_centers = find_lane_boundaries(histogram, num_peaks=4, min_distance=150)
        print(f"Found {len(lane_centers) if lane_centers else 0} peaks at: {lane_centers}")

        if debug_display:
            hist_display = np.zeros((256, histogram.shape[0], 3), dtype=np.uint8)
            for i, val in enumerate(histogram):
                height = int(val / histogram.max() * 256) if histogram.max() > 0 else 0
                cv2.line(hist_display, (i, 256), (i, 256 - height), (255, 255, 0), 1)
            cv2.imshow('Histogram', hist_display)
            cv2.waitKey(1)

        lanes = detect_multiple_lanes(binary_warped, num_lanes=3)
        
        if lanes is None or len(lanes) == 0:
            print("multi-lane detection failed")
            return {
                'error': 'multi-lane detection failed',
                'confidence': 0
            }
        
        if debug_display:
            warped_lane_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Left, Center, Right
            lane_names = ['left', 'center', 'right']
            
            for lane_idx, lane in enumerate(lanes):
                ploty = lane['ploty']
                left_fitx = lane['left_fitx']
                right_fitx = lane['right_fitx']
                color = colors[lane_idx]
                
                if len(left_fitx) > 0 and len(ploty) > 0:
                    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                    cv2.polylines(warped_lane_img, np.int32([left_points]), False, color, 2)
                
                if len(right_fitx) > 0 and len(ploty) > 0:
                    right_points = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                    cv2.polylines(warped_lane_img, np.int32([right_points]), False, color, 2)
            
            cv2.imshow('Warped Lanes Detected', warped_lane_img)
            
            lane_overlay = img.copy()
            
            for lane_idx, lane in enumerate(lanes):
                ploty = lane['ploty']
                left_fitx = lane['left_fitx']
                right_fitx = lane['right_fitx']
                color = colors[lane_idx]
                
                try:
                    warped_h, warped_w = binary_warped.shape
                    
                    left_x_orig = ploty
                    left_y_orig = warped_w - left_fitx
                    
                    right_x_orig = ploty
                    right_y_orig = warped_w - right_fitx
                    
                    left_pts_unrot = np.array([np.transpose(np.vstack([left_x_orig, left_y_orig]))], dtype=np.float32)
                    right_pts_unrot = np.array([np.transpose(np.vstack([right_x_orig, right_y_orig]))], dtype=np.float32)
                    
                    left_pts_orig = cv2.perspectiveTransform(left_pts_unrot, Minv)
                    right_pts_orig = cv2.perspectiveTransform(right_pts_unrot, Minv)
                    
                    cv2.polylines(lane_overlay, np.int32([left_pts_orig]), False, color, 4)
                    cv2.polylines(lane_overlay, np.int32([right_pts_orig]), False, color, 4)
                    
                except Exception as lane_err:
                    print(f"Lane {lane_idx} transform error: {lane_err}")
            
            cv2.imshow('Lane Lines CV', lane_overlay)

        lane_info = get_current_lane(lanes, image_width=binary_warped.shape[1])
        current_lane_data = lane_info['current_lane']
        all_lanes_classified = lane_info['all_lanes']
        
        if debug_display and current_lane_data:
            lane_overlay_shaded = img.copy()
            lane_overlay_alpha = np.zeros_like(img, dtype=np.uint8)
            
            lane_colors = {
                'left': (255, 100, 0),      # Orange
                'center': (0, 255, 0),     # Green
                'right': (0, 100, 255)     # Red
            }
            
            for lane_class, lane_dict in all_lanes_classified.items():
                lane = lane_dict['lane_data']
                ploty = lane['ploty']
                left_fitx = lane['left_fitx']
                right_fitx = lane['right_fitx']
                color = lane_colors.get(lane_class, (100, 100, 100))
                
                try:
                    warped_h, warped_w = binary_warped.shape
                    
                    left_x_orig = ploty
                    left_y_orig = warped_w - left_fitx
                    right_x_orig = ploty
                    right_y_orig = warped_w - right_fitx
                    
                    left_pts_unrot = np.array([np.transpose(np.vstack([left_x_orig, left_y_orig]))], dtype=np.float32)
                    right_pts_unrot = np.array([np.transpose(np.vstack([right_x_orig, right_y_orig]))], dtype=np.float32)
                    
                    left_pts_orig = cv2.perspectiveTransform(left_pts_unrot, Minv)
                    right_pts_orig = cv2.perspectiveTransform(right_pts_unrot, Minv)
                    
                    left_pts_list = left_pts_orig[0].astype(np.int32)
                    right_pts_list = right_pts_orig[0].astype(np.int32)
                    
                    lane_polygon = np.vstack([left_pts_list, right_pts_list[::-1]])
                    
                    cv2.fillPoly(lane_overlay_alpha, [lane_polygon], color)
                    
                except Exception as shade_err:
                    print(f"Lane shade error for {lane_class}: {shade_err}")
            
            lane_overlay_shaded = cv2.addWeighted(lane_overlay_shaded, 1.0, lane_overlay_alpha, 0.3, 0)
            
            for lane_class, lane_dict in all_lanes_classified.items():
                lane = lane_dict['lane_data']
                ploty = lane['ploty']
                left_fitx = lane['left_fitx']
                right_fitx = lane['right_fitx']
                color = lane_colors.get(lane_class, (100, 100, 100))
                
                try:
                    warped_h, warped_w = binary_warped.shape
                    
                    left_x_orig = ploty
                    left_y_orig = warped_w - left_fitx
                    right_x_orig = ploty
                    right_y_orig = warped_w - right_fitx
                    
                    left_pts_unrot = np.array([np.transpose(np.vstack([left_x_orig, left_y_orig]))], dtype=np.float32)
                    right_pts_unrot = np.array([np.transpose(np.vstack([right_x_orig, right_y_orig]))], dtype=np.float32)
                    
                    left_pts_orig = cv2.perspectiveTransform(left_pts_unrot, Minv)
                    right_pts_orig = cv2.perspectiveTransform(right_pts_unrot, Minv)
                    
                    cv2.polylines(lane_overlay_shaded, np.int32([left_pts_orig]), False, color, 3)
                    cv2.polylines(lane_overlay_shaded, np.int32([right_pts_orig]), False, color, 3)
                    
                    lane_bottom_x = int((left_pts_orig[-1][0][0] + right_pts_orig[-1][0][0]) / 2)
                    lane_bottom_y = int((left_pts_orig[-1][0][1] + right_pts_orig[-1][0][1]) / 2)
                    
                    label_text = f"{lane_class.upper()}"
                    if lane_class == current_lane_data['lane_class']:
                        label_text += " (CURRENT)"
                    
                    cv2.putText(lane_overlay_shaded, label_text, (lane_bottom_x - 40, lane_bottom_y + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                except Exception as label_err:
                    print(f"Lane label error for {lane_class}: {label_err}")
            
            cv2.imshow('Lanes with Shading', lane_overlay_shaded)

        all_metrics = []
        for lane_idx, lane in enumerate(lanes):
            metrics_result = calculate_curvature_and_deviation(
                lane['ploty'], 
                lane['left_fitx'], 
                lane['right_fitx'], 
                binary_warped
            )
            
            if metrics_result is not None:
                if len(metrics_result) == 6:
                    left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = metrics_result
                else:
                    left_curverad, right_curverad, deviation, lane_center, vehicle_center = metrics_result[:5]
                    lane_width = None
            else:
                left_curverad, right_curverad, deviation, lane_center, vehicle_center, lane_width = None, None, None, None, None, None
            
            lane_metrics = {
                'lane_id': lane_idx,
                'left_curverad': left_curverad,
                'right_curverad': right_curverad,
                'deviation': deviation,
                'lane_center': lane_center,
                'vehicle_center': vehicle_center,
                'lane_width': lane_width,
            }
            all_metrics.append(lane_metrics)

        previous_fit = None

        metrics = {
            'lanes': all_metrics,
            'confidence': confidence,
        }
        
        return metrics
        
    except Exception as e:
        print(f"Lane detection error CV: {e}")
        metrics = {
            'left_curverad': 0,
            'right_curverad': 0,
            'deviation': 0,
            'lane_center': 0,
            'vehicle_center': 0,
            'lane_width': 0,
            'confidence': 0,
            'error': str(e)
        }
        return metrics
    

if __name__ == "__main__":
    video_path = "C:\\Users\\user\\Documents\\github\\self-driving-car-simulation\\media\\nl_highway_trim.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        metrics = process_frame_cv(frame, speed=100, debug_display=True, perspective_debug_display=True)
        print(f"Frame {frame_count}: {metrics}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
