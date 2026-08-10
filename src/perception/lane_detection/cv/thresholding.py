import numpy as np
import cv2

# Brightness thresholds for adaptive behavior
DARK_THRESHOLD = 80
MEDIUM_LOW_THRESHOLD = 100
MEDIUM_BRIGHT_THRESHOLD = 180
BRIGHT_THRESHOLD = 200

# LAB L-channel thresholds (white lanes) recalibrated
L_THRESH_DARK = (160, 255)
L_THRESH_MEDIUM = (210, 255)
L_THRESH_BRIGHT = (210, 255)
L_THRESH_DEFAULT = (200, 255)

# LAB B-channel thresholds (yellow lanes)
B_THRESH_DARK = (145, 200)
B_THRESH_MEDIUM = (150, 200)
B_THRESH_BRIGHT = (155, 200)
B_THRESH_DEFAULT = (150, 200)

# If B-channel has fewer pixels than this, exclude it from voting since b channel is good for yellow lanes but if there are none it disrupts the voting
B_PIXEL_THRESHOLD = 100000


def color_threshold(image, avg_brightness=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    w_h_min, w_h_max = 0, 180
    w_s_min, w_s_max = 0, 30
    w_v_min, w_v_max = 210, 255

    y_h_min, y_h_max = 15, 30
    y_s_min, y_s_max = 100, 255
    y_v_min, y_v_max = 130, 255


    if not hasattr(color_threshold, "brightness_history"):
        color_threshold.brightness_history = []

    if avg_brightness is not None:
        color_threshold.brightness_history.append(avg_brightness)
        if len(color_threshold.brightness_history) > 5:
            color_threshold.brightness_history.pop(0)
            
        avg_recent = np.mean(color_threshold.brightness_history)
        variance = np.var(color_threshold.brightness_history) if len(color_threshold.brightness_history) > 1 else 0
        
        print(f"[Thresholding] Avg brightness: {avg_brightness:.1f}, Recent avg: {avg_recent:.1f}, Variance: {variance:.1f}")
        
        if avg_recent > BRIGHT_THRESHOLD:
            w_s_max = 30
            w_v_min = 210
            y_s_min = 90
            
        elif avg_recent > MEDIUM_BRIGHT_THRESHOLD:
            w_v_min = 205
            w_s_max = 35
            
        elif MEDIUM_LOW_THRESHOLD < avg_recent < MEDIUM_BRIGHT_THRESHOLD:
            w_v_min = 200
            w_s_max = 40

        elif DARK_THRESHOLD < avg_recent <= MEDIUM_LOW_THRESHOLD:
            w_v_min = 180
            w_s_max = 45

        elif avg_brightness <= DARK_THRESHOLD:
            w_v_min = 160
            w_s_max = 50
            y_v_min = 90
            y_s_min = 70

    # Apply white mask
    white_lower = np.array([w_h_min, w_s_min, w_v_min])
    white_upper = np.array([w_h_max, w_s_max, w_v_max])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Apply yellow mask
    yellow_lower = np.array([y_h_min, y_s_min, y_v_min])
    yellow_upper = np.array([y_h_max, y_s_max, y_v_max])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
    binary = np.zeros_like(hsv[:,:,0])
    binary[combined_mask > 0] = 1
    
    return binary


def majority_vote(binaries, n_vote):
    """
    Combine multiple thresholds: requires n_vote out of total filters to agree.
    
    Args:
        binaries (list): List of binary threshold results (numpy arrays)
        n_vote (int): Number of filters that must agree
    
    Returns:
        numpy array: Binary image where pixels passed majority vote
    """
    binaries = [b.astype(np.uint8) for b in binaries]
    stacked = np.stack(binaries, axis=-1)
    sum_binary = np.sum(stacked, axis=-1)
    print(f"[Thresholding] Majority vote feature sums: {[np.sum(b) for b in binaries]}")
    print(f"[Thresholding] Voting threshold: {n_vote} out of {len(binaries)} features")
    vote_binary = np.zeros_like(sum_binary)
    vote_binary[sum_binary >= n_vote] = 1
    return vote_binary.astype(np.uint8)


def adaptive_majority_vote(image, avg_brightness, ufld_mask=None):
    """
    Adaptive voting combining HSV, LAB L-channel, conditional LAB B-channel, and optional UFLD.
    Requires 2 votes when 2 or 3 color features are active.

    Args:
        image (numpy array): RGB image
        avg_brightness (float): Average brightness of the image
        ufld_mask (numpy array): Optional UFLD neural network lane mask

    Returns:
        numpy array: Binary image from majority voting
    """
    hsv_binary = color_threshold(image, avg_brightness=avg_brightness)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    b_channel = lab[:,:,2]

    # L-channel for white lanes
    if avg_brightness > BRIGHT_THRESHOLD:
        l_thresh = L_THRESH_BRIGHT
    elif avg_brightness < DARK_THRESHOLD:
        l_thresh = L_THRESH_DARK
    else:
        l_thresh = L_THRESH_MEDIUM
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    # B-channel for yellow lanes
    if avg_brightness > BRIGHT_THRESHOLD:
        b_thresh = B_THRESH_BRIGHT
    elif avg_brightness < DARK_THRESHOLD:
        b_thresh = B_THRESH_DARK
    else:
        b_thresh = B_THRESH_MEDIUM

    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

    b_pixel_count = np.sum(b_binary)
    include_b_channel = b_pixel_count >= B_PIXEL_THRESHOLD

    features = [hsv_binary, l_binary]

    if include_b_channel:
        features.append(b_binary)
        print(f"[Thresholding] Including LAB B: {b_pixel_count} pixels (>= {B_PIXEL_THRESHOLD} threshold)")
    else:
        print(f"[Thresholding] Excluding LAB B: {b_pixel_count} pixels (< {B_PIXEL_THRESHOLD} threshold)")

    if ufld_mask is not None and ufld_mask.size > 0:
        ufld_binary = (ufld_mask > 0).astype(np.uint8)
        features.append(ufld_binary)
        print(f"[Thresholding] Including UFLD: {np.sum(ufld_binary)} pixels")

    n_features = len(features)
    n_vote = 2

    result = majority_vote(features, n_vote)

    print(f"[Thresholding] Final combined pixels: {np.sum(result)}")
    feature_names = ["HSV", "L"]
    if include_b_channel:
        feature_names.append("B")
    if ufld_mask is not None and ufld_mask.size > 0:
        feature_names.append("UFLD")
    feature_sums = [np.sum(f) for f in features]
    print(f"  Feature votes: {dict(zip(feature_names, feature_sums))}")
    return result


def apply_thresholds_with_voting(image, src_points=None, debug_display=False, ufld_mask=None):
    """
    Apply adaptive majority voting thresholds with optional ROI masking.

    Args:
        image (numpy array): RGB image
        src_points (numpy array): Source points for ROI masking (optional)
        debug_display (bool): Whether to show debug windows
        ufld_mask (numpy array): Optional UFLD neural network lane mask

    Returns:
        tuple: (combined_binary, avg_brightness)
    """
    mask = None
    if src_points is not None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        src_poly = np.array(src_points, dtype=np.int32)
        cv2.fillPoly(mask, [src_poly], 1)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray[mask == 1])
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)

    combined_binary = adaptive_majority_vote(image, avg_brightness, ufld_mask=ufld_mask)
    
    if mask is not None:
        combined_binary = combined_binary * mask
    
    if debug_display:
        hsv_binary = color_threshold(image, avg_brightness=avg_brightness)
        hsv_binary_uint8 = hsv_binary.astype(np.uint8)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        b_channel = lab[:,:,2]

        if avg_brightness > BRIGHT_THRESHOLD:
            l_thresh = L_THRESH_BRIGHT
        elif avg_brightness < DARK_THRESHOLD:
            l_thresh = L_THRESH_DARK
        else:
            l_thresh = L_THRESH_MEDIUM
        l_binary = np.zeros_like(l_channel, dtype=np.uint8)
        l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

        if avg_brightness > BRIGHT_THRESHOLD:
            b_thresh = B_THRESH_BRIGHT
        elif avg_brightness < DARK_THRESHOLD:
            b_thresh = B_THRESH_DARK
        else:
            b_thresh = B_THRESH_MEDIUM

        b_binary = np.zeros_like(b_channel, dtype=np.uint8)
        b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

        debug_img = np.zeros((combined_binary.shape[0], combined_binary.shape[1], 3), dtype=np.uint8)
        debug_img[hsv_binary_uint8 == 1] = [255, 0, 255]
        debug_img[(hsv_binary_uint8 == 0) & (l_binary == 1)] = [0, 0, 255]
        debug_img[(hsv_binary_uint8 == 0) & (l_binary == 0) & (b_binary == 1)] = [255, 0, 0]
        debug_img[(hsv_binary_uint8 == 1) & (l_binary == 1)] = [0, 255, 255]

        hsv_display = np.dstack((hsv_binary_uint8, hsv_binary_uint8, hsv_binary_uint8)) * 255
        hsv_display_resized = cv2.resize(hsv_display, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('HSV', hsv_display_resized)

        l_display = np.dstack((l_binary, l_binary, l_binary)) * 255
        l_display_resized = cv2.resize(l_display, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('LAB L', l_display_resized)

        b_display = np.dstack((b_binary, b_binary, b_binary)) * 255
        b_display_resized = cv2.resize(b_display, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('LAB B', b_display_resized)

        if ufld_mask is not None and ufld_mask.size > 0:
            ufld_display_uint8 = (ufld_mask > 0).astype(np.uint8)
            ufld_display = np.dstack((ufld_display_uint8, ufld_display_uint8, ufld_display_uint8)) * 255
            ufld_display_resized = cv2.resize(ufld_display, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('UFLD', ufld_display_resized)

        combined_display = np.dstack((combined_binary, combined_binary, combined_binary)) * 255
        combined_display_resized = cv2.resize(combined_display, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('Combined', combined_display_resized)

        debug_display_resized = cv2.resize(debug_img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('Debug', debug_display_resized)
    
    return combined_binary, avg_brightness
