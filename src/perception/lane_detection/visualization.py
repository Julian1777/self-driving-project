import numpy as np
import cv2


def draw_multiple_lanes_overlay(original_image, warped_image, Minv, lanes, all_lanes_classified=None):
    """
    Draw multiple detected lanes on the original image with different colors.
    
    Args:
        original_image: The original undistorted image
        warped_image: The warped binary image
        Minv: Inverse perspective transform matrix
        lanes: List of lane dictionaries containing 'left_fitx', 'right_fitx', 'ploty'
        all_lanes_classified: Dictionary of classified lanes (left/center/right)
    
    Returns:
        Image with lane overlays
    """
    if lanes is None or len(lanes) == 0:
        return original_image
    
    try:
        h, w = warped_image.shape[:2]
        lane_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
        lane_names = ['left', 'center', 'right']
        
        for lane_idx, lane in enumerate(lanes):
            ploty = lane['ploty']
            left_fitx = lane['left_fitx']
            right_fitx = lane['right_fitx']
            color = colors[lane_idx % len(colors)]
            
            left_fitx = np.clip(left_fitx, 0, w - 1).astype(np.int32)
            right_fitx = np.clip(right_fitx, 0, w - 1).astype(np.int32)
            ploty = ploty.astype(np.int32)
            
            if len(left_fitx) > 0 and len(ploty) > 0:
                left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
                cv2.polylines(lane_overlay, left_points, isClosed=False, color=color, thickness=3)
            
            if len(right_fitx) > 0 and len(ploty) > 0:
                right_points = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
                cv2.polylines(lane_overlay, right_points, isClosed=False, color=color, thickness=3)
            
            if len(left_fitx) > 1 and len(right_fitx) > 1:
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))], dtype=np.int32)
                pts = np.hstack((pts_left, pts_right))
        
        warped_overlay = cv2.warpPerspective(lane_overlay, Minv, (original_image.shape[1], original_image.shape[0]))
        result = cv2.addWeighted(original_image, 1, warped_overlay, 0.3, 0)
        
        if all_lanes_classified is not None:
            lane_colors = {
                'left': (255, 100, 0),      # orange
                'center': (0, 255, 0),     # green
                'right': (0, 100, 255)     # red
            }
            
            for lane_class, lane_dict in all_lanes_classified.items():
                lane = lane_dict['lane_data']
                ploty = lane['ploty']
                left_fitx = lane['left_fitx']
                right_fitx = lane['right_fitx']
                color = lane_colors.get(lane_class, (100, 100, 100))
                
                try:
                    left_fitx = np.clip(left_fitx, 0, original_image.shape[1] - 1).astype(np.int32)
                    right_fitx = np.clip(right_fitx, 0, original_image.shape[1] - 1).astype(np.int32)
                    ploty = np.clip(ploty, 0, h - 1).astype(np.int32)
                    
                    left_pts_unrot = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.float32)
                    right_pts_unrot = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.float32)
                    
                    left_pts_orig = cv2.perspectiveTransform(left_pts_unrot, Minv)
                    right_pts_orig = cv2.perspectiveTransform(right_pts_unrot, Minv)
                    
                    cv2.polylines(result, np.int32([left_pts_orig]), isClosed=False, color=color, thickness=2)
                    cv2.polylines(result, np.int32([right_pts_orig]), isClosed=False, color=color, thickness=2)
                    
                    lane_bottom_x = int((left_pts_orig[-1][0][0] + right_pts_orig[-1][0][0]) / 2)
                    lane_bottom_y = int((left_pts_orig[-1][0][1] + right_pts_orig[-1][0][1]) / 2)
                    
                    label_text = f"{lane_class.upper()}"
                    cv2.putText(result, label_text, (lane_bottom_x - 40, lane_bottom_y + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                except Exception as lane_err:
                    print(f"Error drawing lane {lane_class}: {lane_err}")
        
        return result
        
    except Exception as e:
        print(f"Error in draw_multiple_lanes_overlay: {e}")
        import traceback
        traceback.print_exc()
        return original_image


def add_text_overlay(image, left_curverad, right_curverad, deviation, avg_brightness, speed, confidence):
    """
    Add text overlay with lane curvature, deviation, average brightness, and speed.
    Args:
        image: Image to add text overlay on
        left_curverad: Left lane line curvature in meters
        right_curverad: Right lane line curvature in meters
        deviation: Vehicle deviation from lane center in meters
        avg_brightness: Average brightness of the image
        speed: Vehicle speed in km/h
        confidence: Confidence score of lane detection (0.0 to 1.0)
    Returns:
        Image with text overlay
    """

    fontType = cv2.FONT_HERSHEY_SIMPLEX
    
    if deviation is None:
        deviation_text = "Deviation: N/A"
    else:
        try:
            direction = '+' if deviation > 0 else '-'
            deviation_text = f"Deviation: {direction}{abs(float(deviation)):.2f}m"
        except (TypeError, ValueError):
            deviation_text = "Deviation: ERROR"
    
    cv2.putText(image, deviation_text, (30, 50), fontType, 0.4, (0, 0, 0), 1)

    if avg_brightness is not None:
        try:
            cv2.putText(image, f"Avg Brightness: {float(avg_brightness):.1f}", (30, 80), fontType, 0.4, (0, 0, 0), 1)
        except (TypeError, ValueError):
            cv2.putText(image, "Avg Brightness: ERROR", (30, 80), fontType, 0.4, (0, 0, 0), 1)
    else:
        cv2.putText(image, "Avg Brightness: N/A", (30, 80), fontType, 0.4, (0, 0, 0), 1)

    if confidence is not None:
        try:
            cv2.putText(image, f"Confidence: {float(confidence):.2f}", (30, 110), fontType, 0.4, (0, 0, 0), 1)
        except (TypeError, ValueError):
            cv2.putText(image, "Confidence: ERROR", (30, 110), fontType, 0.4, (0, 0, 0), 1)
    else:
        cv2.putText(image, "Confidence: N/A", (30, 110), fontType, 0.4, (0, 0, 0), 1)

    return image

def create_mask_overlay(img, mask, alpha=0.4, color=(0, 255, 0)):
    """
    Create an overlay of a binary mask on the original image.
    
    Args:
        img: The original image (BGR)
        mask: Binary mask (0s and 1s)
        alpha: Transparency of the overlay (0.0 to 1.0)
        color: Color of the mask overlay (BGR tuple)
    
    Returns:
        Image with mask overlay
    """
    try:
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
            
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        
        colored_mask = np.zeros_like(img)
        colored_mask[mask > 0] = color
        
        overlay = img.copy()
        mask_bool = mask > 0
        
        for c in range(3):
            overlay[..., c] = np.where(
                mask_bool,
                (1 - alpha) * overlay[..., c] + alpha * color[c],
                overlay[..., c]
            )
        
        result = overlay.astype(np.uint8)
        
        return result
        
    except Exception as e:
        print(f"Error in create_mask_overlay: {e}")
        return img
