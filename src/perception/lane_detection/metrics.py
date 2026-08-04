import numpy as np


def smooth_deviation(raw_deviation, alpha=0.65):
    """
    Apply exponential smoothing to deviation values to reduce noise.
    
    Args:
        raw_deviation: Current frame's raw deviation value
        alpha: Smoothing factor (0-1), higher values = less smoothing
        
    Returns:
        smoothed_deviation: Filtered deviation value
    """

    if not hasattr(smooth_deviation, 'smoothed_deviation'):
        smooth_deviation.smoothed_deviation = raw_deviation if raw_deviation is not None else 0.0
    
    if raw_deviation is not None:
        smooth_deviation.smoothed_deviation = (
            alpha * raw_deviation + (1 - alpha) * smooth_deviation.smoothed_deviation
        )
    
    return smooth_deviation.smoothed_deviation


def apply_deviation_deadzone_and_scaling(smoothed_deviation, dead_zone=0.1, max_dev=1.5):
    """
    Apply deadzone and scaling to smoothed deviation for control purposes.
    
    Args:
        smoothed_deviation: Smoothed deviation value
        dead_zone: Minimum deviation threshold below which output is zero
        max_dev: Maximum deviation for scaling calculation
        
    Returns:
        effective_deviation: Processed deviation ready for control input
    """

    if smoothed_deviation is None:
        return 0.0
    
    if abs(smoothed_deviation) < dead_zone:
        effective_deviation = 0.0
    else:
        ramp = (abs(smoothed_deviation) - dead_zone) / (max_dev - dead_zone)
        ramp = np.clip(ramp, 0.0, 1.0)
        effective_deviation = np.sign(smoothed_deviation) * ramp * min(abs(smoothed_deviation), max_dev)
        if abs(smoothed_deviation) > max_dev:
            print(f"Large deviation detected: {smoothed_deviation:.2f}m")
    
    return effective_deviation


def validate_lane_geometry(left_fitx, right_fitx, ploty):
    """
    Validate lane geometry for reasonable lane width and positioning.
    
    Returns:
        bool: True if geometry is valid, False otherwise
        lane_width_pix: Lane width in pixels (if valid)
    """

    if len(left_fitx) == 0 or len(right_fitx) == 0:
        print("Empty lane fit arrays detected")
        return False, None

    leftx = left_fitx[::-1]
    rightx = right_fitx[::-1]

    left_bottom = leftx[-1]
    right_bottom = rightx[-1]
    lane_width_pix = right_bottom - left_bottom

    # Check for crossed lanes (removed - allow lanes to converge)
    if left_bottom > right_bottom:
        print(f"Lane lines reversed: left={left_bottom:.1f}, right={right_bottom:.1f}")
        return False, None

    return True, lane_width_pix


def check_lane_width_outliers(lane_width_pix):
    """
    Check if current lane width is an outlier based on historical data.
    
    Returns:
        bool: True if lane width is acceptable, False if it's an outlier
    """

    if not hasattr(check_lane_width_outliers, 'lane_width_history'):
        check_lane_width_outliers.lane_width_history = []
    if not hasattr(check_lane_width_outliers, 'outlier_count'):
        check_lane_width_outliers.outlier_count = 0
    if not hasattr(check_lane_width_outliers, 'consecutive_outliers'):
        check_lane_width_outliers.consecutive_outliers = 0
        
    lane_width_history = check_lane_width_outliers.lane_width_history

    lane_width_history.append(lane_width_pix)
    if len(lane_width_history) > 30:
        lane_width_history.pop(0)
        
    if len(lane_width_history) >= 15:
        avg_lane_width = np.mean(lane_width_history)
        # Tighter tolerance: only allow 10% variation instead of 15%
        max_lane_width = avg_lane_width * 1.10
        min_lane_width = avg_lane_width * 0.90
        if lane_width_pix < min_lane_width or lane_width_pix > max_lane_width:
            check_lane_width_outliers.outlier_count += 1
            print(f"Lane width outlier: {lane_width_pix:.1f} pixels (avg={avg_lane_width:.1f}), grace count: {check_lane_width_outliers.outlier_count}")
            
            check_lane_width_outliers.consecutive_outliers += 1
            
            if check_lane_width_outliers.outlier_count >= 2:
                check_lane_width_outliers.outlier_count = 0
                
                if check_lane_width_outliers.consecutive_outliers >= 7:
                    print("Resetting lane width history due to outliers")
                    lane_width_history.clear()
                    check_lane_width_outliers.consecutive_outliers = 0
                    
                return False
        else:
            check_lane_width_outliers.outlier_count = 0
            check_lane_width_outliers.consecutive_outliers = 0
    else:
        if lane_width_pix > 700:
            print(f"Unreasonably wide lane width: {lane_width_pix:.1f} pixels")
            return False

    return True


def calculate_curvature_and_deviation(ploty, left_fitx, right_fitx, binary_warped, original_image_width=None):
    """
    Calculate lane curvature and vehicle deviation from lane center.
    
    Args:
        ploty: Y coordinates for lane fitting
        left_fitx: X coordinates for left lane line
        right_fitx: X coordinates for right lane line
        binary_warped: Warped binary image for reference
        original_image_width: Width of original (unwarped) image. If None, uses binary_warped width.
    Returns:
        tuple: (left_curverad, right_curverad, deviation_m, lane_center, vehicle_center)
               Returns (None, None, None, None, None) if validation fails
    """
    ym_per_pix = 30/720  # meters per pixel in y dimension

    # Validate lane geometry
    is_valid, lane_width_pix = validate_lane_geometry(left_fitx, right_fitx, ploty)
    if not is_valid:
        return None, None, None, None, None

    # Check for lane width outliers
    if not check_lane_width_outliers(lane_width_pix):
        return None, None, None, None, None

    # Get geometry values
    leftx = left_fitx[::-1]
    rightx = right_fitx[::-1]
    y_eval = np.max(ploty)
    left_bottom = leftx[-1]
    right_bottom = rightx[-1]

    # Calculate scaling factor
    xm_per_pix = 3.55 / lane_width_pix

    if xm_per_pix <= 0 or xm_per_pix > 0.1:
        print(f"Unreasonable xm_per_pix: {xm_per_pix:.4f}")
        return None, None, None, None, None

    try:
        # Calculate curvature in real world coordinates
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
                        / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) \
                         / np.absolute(2*right_fit_cr[0])

        # Limit unreasonably large curvature values
        max_reasonable_curve = 1000  # meters
        if left_curverad > max_reasonable_curve or right_curverad > max_reasonable_curve:
            left_curverad = min(left_curverad, max_reasonable_curve)
            right_curverad = min(right_curverad, max_reasonable_curve)

        # Calculate lane center and vehicle deviation
        # Both lane_center and vehicle_center must be in WARPED pixel coordinates
        lane_center = (left_bottom + right_bottom) / 2.0
        vehicle_center = binary_warped.shape[1] / 2.0  # Warped image center
        
        deviation_pixels = vehicle_center - lane_center
        deviation_m = deviation_pixels * xm_per_pix
        
        # print(f"DEBUG: vehicle_center={vehicle_center:.1f}px, lane_center={lane_center:.1f}px, deviation_pixels={deviation_pixels:.1f}px")
        # print(f"DEBUG: xm_per_pix={xm_per_pix:.4f}, lane_width_pix={lane_width_pix:.1f}px")
        # print(f"DEBUG: deviation_m={deviation_m:.3f}m")

        # Limit unreasonable deviation values
        max_reasonable_deviation = 1.0
        if abs(deviation_m) > max_reasonable_deviation:
            print(f"Unreasonable deviation detected: {deviation_m:.2f}m, clipping to {max_reasonable_deviation:.2f}m")
            deviation_m = np.clip(deviation_m, -max_reasonable_deviation, max_reasonable_deviation)

        return left_curverad, right_curverad, deviation_m, lane_center, vehicle_center, lane_width_pix
        
    except Exception as e:
        print(f"Error in curvature calculation: {e}")
        return None, None, None, None, None, None


def process_deviation(raw_deviation, alpha=0.85, dead_zone=0.02, max_dev=1.5):
    """
    Process raw deviation for use in control systems.
    Applies deadzone/scaling FIRST, then smoothing.
    
    Args:
        raw_deviation: Raw deviation value from lane detection
        alpha: Smoothing factor for exponential smoothing (higher = less smoothing, more responsive)
        dead_zone: Minimum deviation threshold (reduced for earlier response)
        max_dev: Maximum deviation for scaling
        
    Returns:
        tuple: (smoothed_deviation, effective_deviation)
    """
    if raw_deviation is None:
        raw_deviation = 0.0
    
    if not hasattr(process_deviation, 'previous_smoothed_deviation'):
        process_deviation.previous_smoothed_deviation = 0.0
    
    # Apply deadzone and scaling FIRST to the raw deviation
    effective_deviation = apply_deviation_deadzone_and_scaling(raw_deviation, dead_zone, max_dev)
    
    # Then smooth the effective deviation
    smoothed_deviation = smooth_deviation(effective_deviation, alpha)
    
    process_deviation.previous_smoothed_deviation = smoothed_deviation
    
    return smoothed_deviation, effective_deviation
