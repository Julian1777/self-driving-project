import numpy as np
from scipy.signal import find_peaks
import cv2 as cv

def filter_close_peaks(peaks, histogram, min_distance=40):
    """
    Filter peaks that are too close together, keeping only the strongest one.
    
    This prevents double-lane detections from double-lines or noise.
    Uses physical constraint: lanes cannot be closer than ~2 meters.
    At 20 pixels/meter (from perspective warp), this means 40 pixels minimum.
    
    args:
        peaks: Array of peak indices
        histogram: 1D array of histogram values
        min_distance: Minimum distance between kept peaks in pixels (default 40 for 2m)
    
    returns:
        Filtered list of peak indices with minimum spacing maintained
    """
    if len(peaks) == 0:
        return peaks
    
    sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
    
    filtered_peaks = []
    for peak in sorted_peaks:
        is_too_close = False
        for selected_peak in filtered_peaks:
            if abs(peak - selected_peak) < min_distance:
                is_too_close = True
                break
        
        if not is_too_close:
            filtered_peaks.append(peak)
    
    return sorted(filtered_peaks)


def find_lane_boundaries(histogram, num_peaks=4, min_distance=80, min_physical_distance=40, height_threshold=None, debug=False):
    """
    Find lane boundaries using peaks in the histogram with physical distance constraints.

    args:
        histogram: 1D array representing the histogram of pixel intensities.
        num_peaks: Maximum number of peaks to identify (default is 4).
        min_distance: Minimum distance between peaks for find_peaks in pixels (default is 80).
        min_physical_distance: Minimum physical distance (2 meters = 40 pixels at 20px/m scale).
        height_threshold: Minimum height of peaks to be considered valid (default is None, which means no threshold).
        debug: Enable debug output

    returns:
        List of lane boundary positions (x-coordinates) in the image.
    
    
    """

    if height_threshold is None:
        height_threshold = 0.2 * np.max(histogram)
    
    peaks, properties = find_peaks(histogram, distance=min_distance, height=height_threshold)
    
    if debug:
        peak_heights = [histogram[p] for p in peaks] if len(peaks) > 0 else []
        print(f"[MULTI_LANE] Initial peaks: {list(peaks)}, heights: {peak_heights}, threshold={height_threshold:.1f}")
    
    filtered_peaks = filter_close_peaks(peaks, histogram, min_distance=min_physical_distance)
    
    if debug:
        print(f"[MULTI_LANE] After filter_close_peaks: {filtered_peaks}")
    
    if len(filtered_peaks) >= num_peaks:
        top_peaks = sorted(filtered_peaks, key=lambda x: histogram[x], reverse=True)[:num_peaks]
    else:
        top_peaks = filtered_peaks
    
    lane_boundaries = sorted(top_peaks)

    if debug:
        print(f"[MULTI_LANE] Final lane boundaries: {lane_boundaries}")
    
    return lane_boundaries

def sliding_window_search(binary_warped, start_x, histogram, num_windows=9, window_height=80, margin=100, minpix=50, debug_display=False):
    """
    Perform sliding window search to find lane pixels and fit polynomial.

    args:
        binary_warped: Warped binary image where lane lines are highlighted.
        histogram: 1D array representing the histogram of pixel intensities.
        num_windows: Number of sliding windows (default is 9).
        margin: Width of the windows +/- margin (default is 100).
        minpix: Minimum number of pixels found to recenter window (default is 50).
        debug_display: If True, display intermediate results for debugging (default is False).
    returns:
        ploty: Array of y-coordinates for plotting.
    """

    height, width = binary_warped.shape

    ploty = np.linspace(height -1, 0, height)

    nonzero = cv.findNonZero(binary_warped.astype(np.uint8))
    if nonzero is None:
        return None
    
    nonzero_y = nonzero[:, 0, 1]
    nonzero_x = nonzero[:, 0, 0]

    lane_x = []
    current_x = start_x

    for window in range(len(ploty) // window_height):
        window_y_low = height - (window + 1) * window_height
        window_y_high = height - window * window_height
        
        window_x_low = current_x - margin
        window_x_high = current_x + margin
        
        good_inds = (
            (nonzero_y >= window_y_low) &
            (nonzero_y < window_y_high) &
            (nonzero_x >= window_x_low) &
            (nonzero_x < window_x_high)
        )
        
        if np.sum(good_inds) > minpix:
            current_x = int(np.mean(nonzero_x[good_inds]))
        
        lane_x.append(current_x)
    
    lane_y_positions = [height - (i + 0.5) * window_height for i in range(len(lane_x))]
    
    try:
        fit = np.polyfit(lane_y_positions, lane_x, 2)
        fitx = np.poly1d(fit)(ploty)
    except:
        fitx = np.array([start_x] * len(ploty))
        fit = None
    
    return {
        'fitx': fitx,
        'fit': fit,
        'ploty': ploty,
        'x_positions': lane_x
    }


def detect_multiple_lanes(binary_warped, num_lanes=3, debug=False):
    """
    Detect multiple lanes (current ± 1) from binary warped image.
    
    Detects 4 boundaries which create 3 lanes between them.
    Each lane has left_fitx (inner boundary) and right_fitx (outer boundary).
    
    Args:
        binary_warped: Binary warped perspective image
        num_lanes: Number of lanes to detect (default 3, creates 4 boundaries)
        debug: Enable debug output
    
    Returns:
        List of lane data dicts with left_fitx and right_fitx, or None if detection fails
    """
    histogram = np.sum(binary_warped, axis=0)
    
    if debug:
        print(f"[MULTI_LANE] Histogram shape: {histogram.shape}, max: {np.max(histogram):.1f}")
    
    num_peaks = num_lanes + 1
    lane_boundaries = find_lane_boundaries(histogram, num_peaks=num_peaks, debug=debug)
    
    if lane_boundaries is not None and len(lane_boundaries) >= num_peaks:
        print(f"[MULTI_LANE] Detected {len(lane_boundaries)} boundaries at: {lane_boundaries}")
    
    if lane_boundaries is None or len(lane_boundaries) < num_peaks:
        print(f"[MULTI_LANE] First attempt found {len(lane_boundaries) if lane_boundaries is not None else 0} peaks, retrying with lower threshold...")
        lane_boundaries = find_lane_boundaries(histogram, num_peaks=num_peaks, height_threshold=0.1 * np.max(histogram), debug=debug)
    
    if lane_boundaries is not None and len(lane_boundaries) >= num_peaks:
        print(f"[MULTI_LANE] Detected {len(lane_boundaries)} boundaries at: {lane_boundaries} (with lower threshold)")
    
    if lane_boundaries is None or len(lane_boundaries) < num_peaks:
        return None
    
    boundary_fitx = []
    for boundary_x in lane_boundaries:
        result = sliding_window_search(binary_warped, start_x=boundary_x, histogram=histogram, debug_display=debug)
        
        if result is None:
            return None
        
        boundary_fitx.append(result)
    
    lanes = []
    for i in range(num_lanes):
        if i + 1 >= len(boundary_fitx):
            break
        lane_data = {
            'lane_id': i,
            'left_fitx': boundary_fitx[i]['fitx'],      # Left boundary
            'right_fitx': boundary_fitx[i + 1]['fitx'],  # Right boundary
            'left_fit': boundary_fitx[i]['fit'],
            'right_fit': boundary_fitx[i + 1]['fit'],
            'ploty': boundary_fitx[i]['ploty'],
            'left_boundary_x': lane_boundaries[i],
            'right_boundary_x': lane_boundaries[i + 1],
        }
        lanes.append(lane_data)
    
    if debug:
        for lane in lanes:
            print(f"[MULTI_LANE] Lane {lane['lane_id']}: left_boundary={lane['left_boundary_x']}, right_boundary={lane['right_boundary_x']}, width={lane['right_boundary_x'] - lane['left_boundary_x']}")
    
    return lanes if len(lanes) == num_lanes else None
