import numpy as np
import cv2

# Brightness thresholds for adaptive behavior
DARK_THRESHOLD = 80
MEDIUM_LOW_THRESHOLD = 100
MEDIUM_BRIGHT_THRESHOLD = 180
BRIGHT_THRESHOLD = 200

# LAB L-channel thresholds (white lanes)
L_THRESH_DARK = (170, 255)
L_THRESH_MEDIUM = (180, 255)
L_THRESH_BRIGHT = (220, 255)
L_THRESH_DEFAULT = (200, 255)

# LAB B-channel thresholds (yellow lanes)
B_THRESH_DARK = (145, 200)
B_THRESH_MEDIUM = (150, 200)
B_THRESH_BRIGHT = (155, 200)
B_THRESH_DEFAULT = (150, 200)

# HLS S-channel thresholds (saturation)
S_THRESH_DARK = (120, 255)
S_THRESH_BRIGHT = (180, 255)
S_THRESH_DEFAULT = (230, 255)

# If B-channel has fewer pixels than this, exclude it from voting since b channel is good for yellow lanes but if there are none it dirupts the voting
B_PIXEL_THRESHOLD = 100000


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dx = 1 if orient == 'x' else 0
    dy = 0 if orient == 'x' else 1
    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) 
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) 
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary


def gradient_thresholds(image, ksize=3, avg_brightness=None):
    x_low, x_high = 40, 120
    y_low, y_high = 40, 120
    mag_low, mag_high = 50, 120
    
    if avg_brightness is not None:
        if avg_brightness < DARK_THRESHOLD:
            x_low = 30
            y_low = 30
            mag_low = 40
        elif avg_brightness > BRIGHT_THRESHOLD:
            x_high = 160
            y_high = 160
            mag_high = 160
    
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(x_low, x_high))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(y_low, y_high))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(mag_low, mag_high))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) | (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined


def color_threshold(image, avg_brightness=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    w_h_min, w_h_max = 0, 180
    w_s_min, w_s_max = 0, 40
    w_v_min, w_v_max = 200, 255

    y_h_min, y_h_max = 10, 45
    y_s_min, y_s_max = 80, 255
    y_v_min, y_v_max = 120, 255

    s_h_min, s_h_max = 0, 180
    s_s_min, s_s_max = 0, 20
    s_v_min, s_v_max = 110, 150


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
            s_v_max = 160

        elif avg_brightness <= DARK_THRESHOLD:
            w_v_min = 160
            w_s_max = 50
            y_v_min = 90
            y_s_min = 70
            s_v_max = 150

    # Apply white mask
    white_lower = np.array([w_h_min, w_s_min, w_v_min])
    white_upper = np.array([w_h_max, w_s_max, w_v_max])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Apply yellow mask
    yellow_lower = np.array([y_h_min, y_s_min, y_v_min])
    yellow_upper = np.array([y_h_max, y_s_max, y_v_max])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    shadow_mask = np.zeros_like(white_mask)
    if avg_brightness is not None and 50 < avg_brightness < 150:
        shadow_lower = np.array([s_h_min, s_s_min, s_v_min])
        shadow_upper = np.array([s_h_max, s_s_max, s_v_max])
        shadow_mask = cv2.inRange(hsv, shadow_lower, shadow_upper)
    
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    if avg_brightness is not None and 60 < avg_brightness < 120:
        combined_mask = cv2.bitwise_or(combined_mask, shadow_mask)
        
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


def adaptive_majority_vote(image, avg_brightness, include_gradient=False):
    """
    Adaptive voting combining HSV, LAB L/B, HLS S, and optional gradient.
    Voting threshold adjusts for lighting: dark/medium (3/n), bright (4/n).
    
    Args:
        image (numpy array): RGB image
        avg_brightness (float): Average brightness of the image
        include_gradient (bool): Whether to include gradient features
    
    Returns:
        numpy array: Binary image from majority voting
    """
    hsv_binary = color_threshold(image, avg_brightness=avg_brightness)
    
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    b_channel = lab[:,:,2]
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # l channel for white
    if avg_brightness > BRIGHT_THRESHOLD:
        l_thresh = L_THRESH_BRIGHT
    elif avg_brightness < DARK_THRESHOLD:
        l_thresh = L_THRESH_DARK
    else:
        l_thresh = L_THRESH_MEDIUM
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    # b channel for yellow
    if avg_brightness > BRIGHT_THRESHOLD:
        b_thresh = B_THRESH_BRIGHT
    elif avg_brightness < DARK_THRESHOLD:
        b_thresh = B_THRESH_DARK
    else:
        b_thresh = B_THRESH_MEDIUM
    
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
    
    # s channel for saturation
    if avg_brightness > BRIGHT_THRESHOLD:
        s_thresh = S_THRESH_BRIGHT
    elif avg_brightness < DARK_THRESHOLD:
        s_thresh = S_THRESH_DARK
    else:
        s_thresh = S_THRESH_DEFAULT
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    if include_gradient:
        grad_binary = gradient_thresholds(image, avg_brightness=avg_brightness)
    
    b_pixel_count = np.sum(b_binary)
    include_b_channel = b_pixel_count >= B_PIXEL_THRESHOLD
    
    # l channel weighted 2x for white lane detection
    features = [hsv_binary, l_binary, l_binary, s_binary]
    
    if include_b_channel:
        features.append(b_binary)
        print(f"[Thresholding] Including LAB B: {b_pixel_count} pixels (>= {B_PIXEL_THRESHOLD} threshold)")
    else:
        print(f"[Thresholding] Excluding LAB B: {b_pixel_count} pixels (< {B_PIXEL_THRESHOLD} threshold) - likely no yellow lanes on this map")
    
    if include_gradient:
        features.append(grad_binary)
    
    n_features = len(features)
    
    if avg_brightness < MEDIUM_LOW_THRESHOLD:
        n_vote = max(2, n_features // 2)
        print(f"[Thresholding] Dark mode: voting {n_vote}/{n_features}")
    elif avg_brightness < MEDIUM_BRIGHT_THRESHOLD:
        n_vote = max(3, n_features // 2 + 1)
        print(f"[Thresholding] Medium mode: voting {n_vote}/{n_features}")
    else:
        n_vote = max(3, n_features // 2 + 1)
        print(f"[Thresholding] Bright mode: voting {n_vote}/{n_features}")
    
    result = majority_vote(features, n_vote)
    print(f"[Thresholding] Majority vote pixels: {np.sum(result)}")
    print(f"  HSV: {np.sum(hsv_binary)}, L (x2): {np.sum(l_binary)}, S: {np.sum(s_binary)}, B: {np.sum(b_binary)}" + (f", Grad: {np.sum(grad_binary)}" if include_gradient else ""))
    return result


def apply_thresholds_with_voting(image, src_points=None, debug_display=False, use_gradient=False):
    """
    Apply adaptive majority voting thresholds with optional ROI masking.
    
    Args:
        image (numpy array): RGB image
        src_points (numpy array): Source points for ROI masking (optional)
        debug_display (bool): Whether to show debug windows
        use_gradient (bool): Whether to include gradient features in voting
    
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
    
    combined_binary = adaptive_majority_vote(image, avg_brightness, include_gradient=use_gradient)
    
    if mask is not None:
        combined_binary = combined_binary * mask
    
    if debug_display:
        hsv_binary = color_threshold(image, avg_brightness=avg_brightness)
        hsv_binary_uint8 = hsv_binary.astype(np.uint8)
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        b_channel = lab[:,:,2]
        
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        
        if avg_brightness > BRIGHT_THRESHOLD:
            l_thresh = L_THRESH_BRIGHT
        elif avg_brightness < DARK_THRESHOLD:
            l_thresh = L_THRESH_DARK
        else:
            l_thresh = L_THRESH_MEDIUM
        l_binary = np.zeros_like(l_channel, dtype=np.uint8)
        l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
        
        if avg_brightness > BRIGHT_THRESHOLD:
            s_thresh = S_THRESH_BRIGHT
        elif avg_brightness < DARK_THRESHOLD:
            s_thresh = S_THRESH_DARK
        else:
            s_thresh = S_THRESH_DEFAULT
        
        s_binary = np.zeros_like(s_channel, dtype=np.uint8)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        if avg_brightness > BRIGHT_THRESHOLD:
            b_thresh = B_THRESH_BRIGHT
        elif avg_brightness < DARK_THRESHOLD:
            b_thresh = B_THRESH_DARK
        else:
            b_thresh = B_THRESH_MEDIUM
        
        b_binary = np.zeros_like(b_channel, dtype=np.uint8)
        b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
        
        if use_gradient:
            grad_binary = gradient_thresholds(image, avg_brightness=avg_brightness)
            grad_binary_uint8 = grad_binary.astype(np.uint8)
        
        debug_img = np.zeros((combined_binary.shape[0], combined_binary.shape[1], 3), dtype=np.uint8)
        debug_img[hsv_binary_uint8 == 1] = [255, 0, 255]
        debug_img[(hsv_binary_uint8 == 0) & (l_binary == 1)] = [0, 0, 255]
        debug_img[(hsv_binary_uint8 == 0) & (l_binary == 0) & (s_binary == 1)] = [0, 255, 0]
        debug_img[(hsv_binary_uint8 == 0) & (l_binary == 0) & (s_binary == 0) & (b_binary == 1)] = [255, 0, 0]
        debug_img[(hsv_binary_uint8 == 1) & ((l_binary == 1) | (s_binary == 1))] = [0, 255, 255]
        
        hsv_display = np.dstack((hsv_binary_uint8, hsv_binary_uint8, hsv_binary_uint8)) * 255
        hsv_display_resized = cv2.resize(hsv_display, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('HSV', hsv_display_resized)
        
        l_display = np.dstack((l_binary, l_binary, l_binary)) * 255
        l_display_resized = cv2.resize(l_display, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('LAB L', l_display_resized)
        
        s_display = np.dstack((s_binary, s_binary, s_binary)) * 255
        s_display_resized = cv2.resize(s_display, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('HLS S', s_display_resized)
        
        b_display = np.dstack((b_binary, b_binary, b_binary)) * 255
        b_display_resized = cv2.resize(b_display, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('LAB B', b_display_resized)
        
        if use_gradient:
            grad_display = np.dstack((grad_binary_uint8, grad_binary_uint8, grad_binary_uint8)) * 255
            grad_display_resized = cv2.resize(grad_display, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Gradient', grad_display_resized)
        
        combined_display = np.dstack((combined_binary, combined_binary, combined_binary)) * 255
        combined_display_resized = cv2.resize(combined_display, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('Combined', combined_display_resized)
        
        debug_display_resized = cv2.resize(debug_img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('Debug', debug_display_resized)
    
    return combined_binary, avg_brightness
