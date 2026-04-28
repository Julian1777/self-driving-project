import numpy as np
import cv2
import pickle
import os
from pathlib import Path


INTRINSIC_MATRIX = np.array([
    [771.21, 0.0, 960.0],
    [0.0, 771.21, 540.0],
    [0.0, 0.0, 1.0]
])

DISTORTION_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # assuming negligible distortion for simplicity

CAMERA_CONFIGS = {
    'q8_andronisk': {
        'height': 1.466,
        'pitch': 0,
        'resolution': (1920, 1080),
        'mtx': INTRINSIC_MATRIX,
        'dist': DISTORTION_COEFFS
    },
    'etk800': {
        'height': 1.36,
        'pitch': 0,
        'resolution': (1920, 1080),
        'mtx': INTRINSIC_MATRIX,
        'dist': DISTORTION_COEFFS
    }
}

def undistort_image(img, calibration_data):
    """
    Undistort an image using calibration parameters.
    
    Args:
        img: Input image
        calibration_data: Dictionary containing 'mtx' and 'dist' keys
    
    Returns:
        Undistorted image
    """
    if calibration_data is None:
        return img
    
    mtx = calibration_data.get('mtx')
    dist = calibration_data.get('dist')
    
    if mtx is None or dist is None:
        return img
    
    return cv2.undistort(img, mtx, dist)


def get_camera_extrinsics(cam_height, cam_pitch_deg):
    """
    Creates the Rotation and Translation matrices for the camera.
    Assumes camera is at (0,0,0) and road is at y = cam_height.
    """
    pitch_rad = np.deg2rad(cam_pitch_deg)
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])

    return R_x

def project_world_to_image(world_points, mtx, R, cam_height):
    """
    Projects 3D world coordinates (meters) to 2D image coordinates (pixels).
    """
    image_points = []
    
    for pt in world_points:

        X_w, Z_w = pt # We only pass X and Z, Y is fixed
        
        # Original vector: [X, cam_height, Z]
        vec_world = np.array([X_w, cam_height, Z_w])
        
        # Apply Rotation
        vec_cam = R @ vec_world
        
        # Extract intrinsic parameters
        f_x, f_y = mtx[0, 0], mtx[1, 1]
        c_x, c_y = mtx[0, 2], mtx[1, 2]
        
        # Prevent division by zero if point is behind camera
        if vec_cam[2] <= 0.1: 
            continue
            
        u = (f_x * vec_cam[0] / vec_cam[2]) + c_x
        v = (f_y * vec_cam[1] / vec_cam[2]) + c_y
        
        image_points.append([u, v])
        
    return np.float32(image_points)

def get_dynamic_src_points(calibration_data, speed=0, cam_config=None):
    """
    Generates SRC points based on PHYSICAL parameters, not hardcoded pixels.
    """
    if cam_config is None:
        cam_config = {'height': 1.4, 'pitch': 10} # meters, degrees
        
    mtx = calibration_data['mtx']
    
    speed_norm = min(speed / 120.0, 1.0)
    look_ahead_start = 3.0 + (5.0 * speed_norm)
    look_ahead_end   = 15.0 + (20.0 * speed_norm)
    lane_width_roi   = 7.0

    world_roi = [
        [-lane_width_roi, look_ahead_start], # BL
        [ lane_width_roi, look_ahead_start], # BR
        [ lane_width_roi, look_ahead_end],   # TR
        [-lane_width_roi, look_ahead_end]    # TL
    ]
    
    R = get_camera_extrinsics(cam_config['height'], cam_config['pitch'])

    src_pixels = project_world_to_image(world_roi, mtx, R, cam_config['height'])
    
    return src_pixels


def get_src_points(img_shape, speed=0, previous_steering=0, vehicle_model='q8_andronisk', calibration_data=None):
    """
    Backward-compatible wrapper to get source points for perspective transform.
    
    This function supports the old API while using the new physical-based approach.
    
    Args:
        img_shape: Image shape (height, width, channels)
        speed: Vehicle speed for dynamic adjustment
        previous_steering: Previous steering angle (for future enhancements)
        vehicle_model: Vehicle model ('q8_andronisk' or 'etk800')
        calibration_data: Calibration data dictionary with 'mtx' key (optional; uses vehicle config by default)
    
    Returns:
        Source points for perspective transform (4 points)
    """
    if vehicle_model not in CAMERA_CONFIGS:
        print(f"Warning: Unknown vehicle model '{vehicle_model}', using default")
        vehicle_model = 'q8_andronisk'
    
    cam_config = CAMERA_CONFIGS[vehicle_model]
    
    if calibration_data is None:
        calibration_data = {'mtx': cam_config['mtx'], 'dist': cam_config['dist']}
    
    if 'mtx' in calibration_data:
        try:
            src_points = get_dynamic_src_points(calibration_data, speed=speed, cam_config=cam_config)
            return src_points
        except Exception as e:
            print(f"Warning: Physical projection failed ({e}), falling back to hardcoded points")
    
    h, w = img_shape[0], img_shape[1]
    
    if vehicle_model == 'etk800':
        # ETK800 specific points
        src = np.float32([
            [w * 0.15, h],      # Bottom-left
            [w * 0.85, h],      # Bottom-right
            [w * 0.95, h * 0.6],  # Top-right
            [w * 0.05, h * 0.6]   # Top-left
        ])
    else:  # q8_andronisk (default)
        # Q8 specific points
        src = np.float32([
            [w * 0.15, h],      # Bottom-left
            [w * 0.85, h],      # Bottom-right
            [w * 0.95, h * 0.6],  # Top-right
            [w * 0.05, h * 0.6]   # Top-left
        ])
    
    return src

def perspective_warp(img, speed=0, calibration_data=None, vehicle_model='q8_andronisk', cam_height=None, cam_pitch=None):
    """
    Apply perspective transform to get bird's-eye view.
    
    Args:
        img: Input image
        speed: Vehicle speed for dynamic adjustment
        calibration_data: Calibration data dictionary
        vehicle_model: Vehicle model ('q8_andronisk' or 'etk800')
        cam_height: Optional camera height override (if not using vehicle_model config)
        cam_pitch: Optional camera pitch override (if not using vehicle_model config)
    
    Returns:
        binary_warped: Warped image in bird's-eye view
        Minv: Inverse perspective transform matrix
    """
    
    if calibration_data is not None:
        img = undistort_image(img, calibration_data)
        mtx = calibration_data.get('mtx')
    else:
        mtx = None

    img_size = (img.shape[1], img.shape[0])
    w, h = img_size
    
    if vehicle_model in CAMERA_CONFIGS:
        cam_config = CAMERA_CONFIGS[vehicle_model]
    else:
        if cam_height is None:
            cam_height = 1.4
        if cam_pitch is None:
            cam_pitch = 10
        cam_config = {'height': cam_height, 'pitch': cam_pitch}
    
    src = get_src_points(img.shape, speed=speed, vehicle_model=vehicle_model, calibration_data=calibration_data)

    scale = 20 # pixels per meter
    
    dst = np.float32([
        [w*0.2, h],       # BL
        [w*0.8, h],       # BR
        [w*0.8, 0],       # TR
        [w*0.2, 0]        # TL
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    binary_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return binary_warped, Minv


def debug_perspective_live(img, speed_kph, previous_steering=0, vehicle_model='q8_andronisk', calibration_data=None):
    """
    Visualize the perspective transform source points on the original image for debugging.
    """
    debug_img = img.copy()
    
    src_points = get_src_points(img.shape, speed_kph, previous_steering, vehicle_model=vehicle_model, calibration_data=calibration_data)
    
    src_int = src_points.astype(np.int32)
    
    cv2.polylines(debug_img, [src_int], isClosed=True, color=(0, 255, 0), thickness=2)
    
    labels = ['Bottom Left', 'Bottom Right', 'Top Right', 'Top Left']
    colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
    
    for i, (point, label, color) in enumerate(zip(src_int, labels, colors)):
        cv2.circle(debug_img, tuple(point), 5, color, -1)
        cv2.putText(debug_img, label, (point[0] + 10, point[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.putText(debug_img, f"Speed: {speed_kph:.1f} km/h", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Steering: {previous_steering:.1f} deg", (10, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Vehicle: {vehicle_model}", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(debug_img, "Perspective Transform Region", (10, 105), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(debug_img, "Green: Transform Area", (10, 125), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    debug_img_resized = cv2.resize(debug_img, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Perspective Transform Debug', debug_img_resized)
    cv2.waitKey(1)
