import sys
import os
import yaml
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, Lidar, Radar, GPS, AdvancedIMU
from foxglove.schemas import Color

from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model

import torch
import numpy as np
import time
import math
import cv2
from scipy.spatial.transform import Rotation as R

from src.control_planning.mpc_controller import MPCController

#from simulation.perception_client import PerceptionClient

# bypassed docker aggregator setup
from ultralytics import YOLO

print("[Main] Loading local models...")
# loaded models locally to bypass docker
local_models = {}
local_models['vehicle'] = YOLO('models/object_detection/object_detection.pt')
local_models['traffic_light'] = YOLO('models/traffic_light/traffic_light_detection.pt')
local_models['sign_detect'] = YOLO('models/traffic_sign/traffic_sign_detection.pt')
local_models['sign_classify'] = load_model('models/traffic_sign/traffic_sign_classification.h5')

# initialize yolop model locally
try:
    import torch
    import torchvision.transforms as transforms
    
    # Add YOLOP repo to path
    yolop_repo_path = os.path.join(os.path.dirname(__file__), '..', 'yolopx')
    sys.path.insert(0, yolop_repo_path)
    
    from lib.config import cfg
    from lib.models import get_net
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolop_model = get_net(cfg)
    checkpoint = torch.load('models/yolop/yolopx.pth', map_location=device)
    yolop_model.load_state_dict(checkpoint['state_dict'])
    yolop_model = yolop_model.to(device)
    yolop_model.eval()
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    yolop_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    local_models['yolop_model'] = yolop_model
    local_models['device'] = device
    local_models['yolop_transforms'] = yolop_transforms
    print("[YOLOP] YOLOP model loaded successfully.")
except Exception as e:
    print(f"[YOLOP] YOLOP utilities not found: {e}")

import sys
sys.modules['__main__'].MODELS = local_models

from src.perception.yolop.main import process_frame as yolop_process
from src.perception.object_detection.main import process_frame as object_process
from src.perception.traffic_light_detection.main import process_frame as tl_process
from src.perception.sign_detection.main import process_frame as sign_process
from src.perception.lane_detection.main import process_frame_cv as cv_lane_process
from src.perception.lane_detection.visualization import create_mask_overlay, draw_multiple_lanes_overlay
from src.perception.lane_detection.cv.perspective import perspective_warp, get_src_points

from src.sensor_fusion.lidar.main import process_frame as lidar_process_frame
from src.sensor_fusion.radar.main import process_frame as radar_process_frame
from src.sensor_fusion.radar.main import process_bsd_frame as radar_bsd_process

from simulation.foxglove_integration.bridge_instance import bridge

logger = logging.getLogger(__name__)

# removed redundant MODELS = {} 

def yaw_to_quat(yaw_deg):
    """
    Convert yaw angle in degrees to a quaternion representation for vehicle orientation.
    Args:
        yaw_deg (float): Yaw angle in degrees
    Returns:
        tuple: Quaternion (x, y, z, w)
    """
    yaw = math.radians(yaw_deg)
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (0.0, 0.0, z, w)

def yaw_rad_to_quaternion(yaw_rad):
    """
    Convert yaw angle in radians to a quaternion representation for vehicle orientation.
    Args:
        yaw_rad (float): Yaw angle in radians
    Returns:
        tuple: Quaternion (x, y, z, w)
    """
    w = math.cos(yaw_rad / 2)
    z = math.sin(yaw_rad / 2)
    return (0.0, 0.0, z, w)

def get_timestamp_ns():
    """
    Get current timestamp in nanoseconds since epoch.
    Returns:
        int: Timestamp in nanoseconds
    """
    return int(time.time_ns())

def load_config():
    """Load all configuration files."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config')
    
    with open(os.path.join(config_path, 'beamng.yaml'), 'r') as f:
        beamng_config = yaml.safe_load(f)
    with open(os.path.join(config_path, 'scenarios.yaml'), 'r') as f:
        scenarios_config = yaml.safe_load(f)
    with open(os.path.join(config_path, 'sensors.yaml'), 'r') as f:
        sensors_config = yaml.safe_load(f)
    with open(os.path.join(config_path, 'control.yaml'), 'r') as f:
        control = yaml.safe_load(f)
    with open(os.path.join(config_path, 'perception.yaml'), 'r') as f:
        perception_config = yaml.safe_load(f)

    return beamng_config, scenarios_config, sensors_config, control, perception_config


def sim_setup(map_name='west_coast_usa', scenario_type='highway', vehicle_name='q8_andronisk'):
    """
    Setup BeamNG simulation, scenario, vehicle, spawn point and sensors.
    Args:
        map_name (str): Name of the map ('west_coast_usa' or 'italy')
        scenario_type (str): Scenario type ('highway' or 'city')
        vehicle_name (str): Name of the vehicle to use ('etk800' or 'q8_andronisk')
    """
    beamng_config, scenarios_config, sensors_config, _, _ = load_config()
    
    if map_name not in scenarios_config['maps']:
        raise ValueError(f"Map '{map_name}' not found in config. Available maps: {list(scenarios_config['maps'].keys())}")
    
    map_cfg = scenarios_config['maps'][map_name]
    
    if scenario_type not in map_cfg or scenario_type not in ['highway', 'city']:
        raise ValueError(f"Scenario type '{scenario_type}' not found for map '{map_name}'. Available: {list(map_cfg.keys())}")
    
    scenario_cfg = map_cfg[scenario_type]
    
    if vehicle_name not in beamng_config['vehicles']:
        raise ValueError(f"Vehicle '{vehicle_name}' not found in config. Available vehicles: {list(beamng_config['vehicles'].keys())}")
    
    vehicle_cfg = beamng_config['vehicles'][vehicle_name]
    
    sim_cfg = beamng_config['simulation']
    beamng = BeamNGpy(sim_cfg['host'], sim_cfg['port'], home=sim_cfg['home'])
    beamng.open()

    scenario = Scenario(map_cfg['map_path'], scenario_cfg['scene'])

    vehicle = Vehicle(
        vehicle_cfg['name'],
        model=vehicle_cfg['model'],
        licence=vehicle_cfg['license'],
        part_config=vehicle_cfg.get('part_config', None)
    )

    # Spawn vehicle
    rot = yaw_to_quat(scenario_cfg['spawn_yaw'])
    scenario.add_vehicle(vehicle, pos=tuple(scenario_cfg['spawn_pos']), rot_quat=rot)

    scenario.make(beamng)
    beamng.settings.set_deterministic(60)
    beamng.scenario.load(scenario)
    beamng.scenario.start()

    # Setup sensors - select config based on vehicle model
    vehicle_model = vehicle_cfg['model']
    if vehicle_model not in sensors_config:
        raise ValueError(f"Sensor configuration for vehicle model '{vehicle_model}' not found in config")
    
    sensors = sensors_config[vehicle_model]
    cameras = {}
    lidar = None
    radars = {}
    gps_sensors = {}
    imus = {}

    # Initialize cameras - support multiple cameras (camera_front, camera_left, camera_right, etc.)
    for sensor_key, sensor_cfg in sensors.items():
        if sensor_key.startswith('camera_') and sensor_cfg.get('enabled', False):
            try:
                camera = Camera(
                    sensor_cfg['name'],
                    beamng,
                    vehicle,
                    requested_update_time=sensor_cfg['requested_update_time'],
                    is_using_shared_memory=sensor_cfg.get('is_using_shared_memory', False),
                    pos=tuple(sensor_cfg['pos']),
                    dir=tuple(sensor_cfg['dir']),
                    field_of_view_y=sensor_cfg['field_of_view_y'],
                    near_far_planes=tuple(sensor_cfg['near_far_planes']),
                    resolution=tuple(sensor_cfg['resolution']),
                    is_streaming=sensor_cfg.get('is_streaming', False),
                    is_render_colours=sensor_cfg.get('is_render_colours', True),
                    is_render_depth=sensor_cfg.get('is_render_depth', False),
                    is_visualised=sensor_cfg.get('is_visualised', False),
                )
                cameras[sensor_key] = camera
                print(f"Camera '{sensor_key}' initialized")
            except Exception as e:
                print(f"Camera '{sensor_key}' initialization error: {e}")
                cameras[sensor_key] = None

    # Initialize LiDAR - support multiple LiDARs (lidar_top, lidar_rear, etc.)
    for sensor_key, sensor_cfg in sensors.items():
        if sensor_key.startswith('lidar_') and sensor_cfg.get('enabled', False):
            try:
                lidar = Lidar(
                    sensor_cfg['name'],
                    beamng,
                    vehicle,
                    requested_update_time=sensor_cfg['requested_update_time'],
                    is_using_shared_memory=sensor_cfg.get('is_using_shared_memory', False),
                    is_rotate_mode=sensor_cfg.get('is_rotate_mode', False),
                    horizontal_angle=sensor_cfg.get('horizontal_angle', 360),
                    vertical_angle=sensor_cfg.get('vertical_angle', 26.9),
                    vertical_resolution=sensor_cfg.get('vertical_resolution', 64),
                    density=sensor_cfg.get('density', 1),
                    frequency=sensor_cfg.get('frequency', 20),
                    max_distance=sensor_cfg.get('max_distance', 120),
                    pos=tuple(sensor_cfg['pos']),
                    dir=tuple(sensor_cfg.get('dir', [0, -1, 0])),
                    is_visualised=sensor_cfg.get('is_visualised', False),
                )
                print(f"[LiDAR] LiDAR '{sensor_key}' initialized")
                break  # Use first enabled LiDAR as primary
            except Exception as e:
                print(f"[LiDAR] LiDAR '{sensor_key}' initialization error: {e}")

    # Initialize Radars - support multiple radars (radar_front, radar_rear_left, radar_rear_right, etc.)
    for sensor_key, sensor_cfg in sensors.items():
        if sensor_key.startswith('radar_') and sensor_cfg.get('enabled', False):
            try:
                print(f"[Radar] Attempting {sensor_key} initialization...")
                radar = Radar(
                    sensor_cfg['name'],
                    beamng,
                    vehicle,
                    requested_update_time=sensor_cfg.get('requested_update_time', 0.05),
                    pos=tuple(sensor_cfg['pos']),
                    dir=tuple(sensor_cfg.get('dir', [0, -1, 0])),
                    up=tuple(sensor_cfg.get('up', [0, 0, 1])),
                    size=tuple(sensor_cfg.get('size', [200, 200])),
                    near_far_planes=tuple(sensor_cfg.get('near_far_planes', [0.1, 200])),
                    field_of_view_y=sensor_cfg.get('field_of_view_y', 18),
                    range_min=sensor_cfg.get('range_min', 0.5),
                    range_max=sensor_cfg.get('range_max', 150.0),
                    vel_min=sensor_cfg.get('vel_min', -40),
                    vel_max=sensor_cfg.get('vel_max', 40),
                    range_bins=sensor_cfg.get('range_bins', 128),
                    azimuth_bins=sensor_cfg.get('azimuth_bins', 64),
                    vel_bins=sensor_cfg.get('vel_bins', 32),
                    half_angle_deg=sensor_cfg.get('half_angle_deg', 9),
                    is_visualised=sensor_cfg.get('is_visualised', False),
                )
                radars[sensor_key] = radar
                print(f"[Radar] {sensor_key.replace('_', ' ').title()} initialized")
            except Exception as e:
                print(f"[Radar] {sensor_key.replace('_', ' ').title()} initialization error: {e}")
                radars[sensor_key] = None

    # Initialize GPS sensors - support multiple GPS (gps_front, gps_rear, etc.)
    for sensor_key, sensor_cfg in sensors.items():
        if sensor_key.startswith('gps_') and sensor_cfg.get('enabled', False):
            try:
                print(f"[GPS] Attempting {sensor_key} initialization...")
                gps = GPS(
                    sensor_cfg['name'],
                    beamng,
                    vehicle,
                    gfx_update_time=sensor_cfg.get('gfx_update_time', 0.0),
                    physics_update_time=sensor_cfg.get('physics_update_time', 0.05),
                    pos=tuple(sensor_cfg['pos']),
                    ref_lon=sensor_cfg.get('ref_lon', 13.1856),
                    ref_lat=sensor_cfg.get('ref_lat', 51.5074),
                    is_send_immediately=sensor_cfg.get('is_send_immediately', False),
                    is_visualised=sensor_cfg.get('is_visualised', False),
                    is_snapping_desired=sensor_cfg.get('is_snapping_desired', False),
                    is_force_inside_triangle=sensor_cfg.get('is_force_inside_triangle', False),
                    is_dir_world_space=sensor_cfg.get('is_dir_world_space', False),
                )
                gps_sensors[sensor_key] = gps
                print(f"[GPS] {sensor_key.replace('_', ' ').title()} initialized")
            except Exception as e:
                print(f"[GPS] {sensor_key} initialization error: {e}")
                gps_sensors[sensor_key] = None

    # Initialize IMU sensors
    for sensor_key, sensor_cfg in sensors.items():
        if sensor_key.startswith('imu_') and sensor_cfg.get('enabled', False):
            try:
                print(f"[IMU] Attempting {sensor_key} initialization...")
                imu = AdvancedIMU(
                    sensor_cfg['name'],
                    beamng,
                    vehicle,
                    gfx_update_time=sensor_cfg.get('gfx_update_time', 0.0),
                    physics_update_time=sensor_cfg.get('physics_update_time', 0.01),
                    pos=tuple(sensor_cfg['pos']),
                    dir=tuple(sensor_cfg.get('dir', [0, -1, 0])),
                    up=tuple(sensor_cfg.get('up', [0, 0, 1])),
                    smoother_strength=sensor_cfg.get('smoother_strength', 1.0),
                    is_send_immediately=sensor_cfg.get('is_send_immediately', False),
                    is_using_gravity=sensor_cfg.get('is_using_gravity', False),
                    is_allow_wheel_nodes=sensor_cfg.get('is_allow_wheel_nodes', False),
                    is_visualised=sensor_cfg.get('is_visualised', False),
                    is_snapping_desired=sensor_cfg.get('is_snapping_desired', False),
                    is_force_inside_triangle=sensor_cfg.get('is_force_inside_triangle', False),
                    is_dir_world_space=sensor_cfg.get('is_dir_world_space', False),
                )
                imus[sensor_key] = imu
                print(f"[IMU] {sensor_key.replace('_', ' ').title()} initialized")
            except Exception as e:
                print(f"[IMU] {sensor_key} initialization error: {e}")
                imus[sensor_key] = None

    # Return primary camera (camera_front) and primary GPS for backwards compatibility
    primary_camera = cameras.get('camera_front', next(iter(cameras.values())) if cameras else None)
    primary_gps = gps_sensors.get('gps_front', next(iter(gps_sensors.values())) if gps_sensors else None)
    primary_imu = imus.get('imu_1', next(iter(imus.values())) if imus else None)

    return beamng, scenario, vehicle, primary_camera, lidar, radars, primary_gps, primary_imu, vehicle_model

def get_vehicle_speed(vehicle):
    """
    Get the vehicle speed in m/s and kph, and also return position.
    Args:
        vehicle (Vehicle): BeamNG vehicle object
    Returns:
        tuple: (speed_mps, speed_kph, position)
    """

    vehicle.poll_sensors()
    if 'vel' in vehicle.state:
        speed_mps = vehicle.state['vel'][0]
        speed_kph = speed_mps * 3.6
    else:
        speed_mps = 0.0
        speed_kph = 0.0

    if 'pos' in vehicle.state:
        position = vehicle.state['pos']
    else:
        print("Vehicle position not available")
        position = None

    if 'dir' in vehicle.state:
        direction = vehicle.state['dir']
    else:
        print("Vehicle direction not available")
        direction = None

    return speed_mps, speed_kph, position, direction


def radar_aeb_acc(radar_front, perception_cfg, speed_kph):
    radar_cfg = perception_cfg['radar']
    radar_result = radar_process_frame(radar_front, radar_cfg, speed_kph)
    return radar_result

def radar_bsd(radar_left, radar_right, perception_cfg):
    """
    Check left and right blind spots using rear radars.
    Returns dictionary with warning statuses for each side.
    """
    radar_cfg = perception_cfg['radar']
    # If the radar sensor exists process its frame to determine if theres a vehicle in blind spot
    left_warning = radar_bsd_process(radar_left, radar_cfg) if radar_left else False
    right_warning = radar_bsd_process(radar_right, radar_cfg) if radar_right else False
    
    return {
        'left_warning': left_warning,
        'right_warning': right_warning
    }


def draw_combined_detections(img, sign_detections, vehicle_detections, tl_detections):
    result_img = img.copy()
    
    # Draw Signs in blue
    for det in sign_detections:
        x1, y1, x2, y2 = det['bbox']
        classification = det.get('classification', 'Sign')
        conf = det.get('classification_confidence', 0.0)
        label = f"{classification} {conf:.2f}"
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(result_img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw Vehicles in green
    for det in vehicle_detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(result_img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw Traffic Lights in orange
    for det in tl_detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2) 
        cv2.putText(result_img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
    return result_img

def cruise_control(target_speed_kph, current_speed_kph, speed_pid, dt):
    """
    Simple cruise control to maintain target speed using PID controller.
    Args:
        target_speed_kph (float): Desired speed in kph
        current_speed_kph (float): Current speed in kph
        speed_pid (PIDController): PID controller instance for speed
        dt (float): Time delta in seconds
    Returns:
        float: Throttle value between 0.0 and 1.0
    """
    speed_error = target_speed_kph - current_speed_kph
    throttle = speed_pid.update(speed_error, dt)
    throttle = np.clip(throttle, 0.0, 1.0)
    return throttle

def main():
    """
    Main function to run the simulation.
    """
    # bypass aggregator setup for now

    # print("Initializing aggregator client")
    # perception_client = PerceptionClient(
    #     host='localhost',
    #     service_ports={
    #         'cv_lane_detection': 4777,
    #         'object_detection': 5777,
    #         'traffic_light_detection': 6777,
    #         'sign_detection': 7777,
    #         'sign_classification': 8777,
    #         'yolop': 9777
    #     },
    #     timeout=2.0,
    #     auto_health_check=True
    # )
    # print("Aggregator ready\n")


    # Change map/scenario here: use map_name='west_coast_usa' or 'italy', scenario_type='highway' or 'city'
    # vehicle_name can be 'etk800' or 'q8_andronisk'
    beamng, scenario, vehicle, camera, lidar, radars, gps, imu, vehicle_model = sim_setup(
        map_name='italy', 
        scenario_type='highway', 
        vehicle_name='etk800'
    )
    print("[Main] Simulation setup complete")

    print("[Main] Waiting for sensors to initialize")
    time.sleep(3)
    
    try:
        print("[Camera] Testing camera...")
        camera_test = camera.poll()
        print(f"[Camera] Camera working: {type(camera_test)}")
    except Exception as e:
        print(f"[Camera] Camera error: {e}")
        
    try:
        print("[LiDAR] Testing lidar...")
        lidar_test = lidar.poll()
        print(f"[LiDAR] LiDAR working: {type(lidar_test)}")
    except Exception as e:
        print(f"[LiDAR] LiDAR error: {e}")

    # Test all radars
    try:
        for radar_name, radar in radars.items():
            try:
                radar_test = radar_name.poll()
                print(f"[Radar] {radar_name} working: {type(radar_test)}")
            except Exception as e:
                print(f"[Radar] {radar_name} error: {e}")
    except Exception as e:
        print(f"[Radar] {radar_name} error: {e}")

    try:
        print("[GPS] Testing GPS...")
        gps_test = gps.poll()
        print(f"[GPS] GPS working: {type(gps_test)}")
    except Exception as e:
        print(f"[GPS] GPS error: {e}")

    try:
        print("[IMU] Testing IMU...")
        imu_test = imu.poll()
        print(f"[IMU] IMU working: {type(imu_test)}")
    except Exception as e:
        print(f"[IMU] IMU error: {e}")

    print("[Main] Setting up traffic")
    try:
        beamng.traffic.spawn(max_amount=3, police_ratio=0.0, extra_amount=0, parked_amount=0)
        print("Traffic spawned: 3 vehicles")
    except Exception as e:
        print(f"[Main] Traffic setup error: {e}")

    # Load control parameters from config
    beamng_cfg, _, _, control, perception_config = load_config()
    control_cfg = control['control']
    perception_cfg = perception_config['perception']
    
    # load perception module flags from beamng config
    perception_flags = beamng_cfg.get('perception', {})
    enable_cv_lane = perception_flags.get('enable_cv_lane_detection', True)
    enable_obj_det = perception_flags.get('enable_object_detection', True)
    enable_tl_det = perception_flags.get('enable_traffic_light_detection', True)
    enable_sign_det = perception_flags.get('enable_sign_detection', True)
    enable_yolop_flag = perception_flags.get('enable_yolop', True)
    
    # Load debug display flags
    debug_cv_lane = perception_flags.get('debug_cv_lane_detection', False)
    debug_perspective = perception_flags.get('debug_perspective', False)
    debug_obj_det = perception_flags.get('debug_object_detection', False)
    debug_yolop = perception_flags.get('debug_yolop', False)
    
    print(f"[Main] perception flags - lane:{enable_cv_lane} obj:{enable_obj_det} tl:{enable_tl_det} sign:{enable_sign_det} yolop:{enable_yolop_flag}")
    print(f"[Main] debug windows - lane:{debug_cv_lane} perspective:{debug_perspective} obj:{debug_obj_det} yolop:{debug_yolop}")

    
    mpc_controller = MPCController(T=0.1, N=10)
    print(f"[Main] MPC Controller initialized")

    previous_steering = 0.0  # For lane detection smoothing
    frame_count = 0

    last_time = time.time()
    try:
        step_i = 0
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            try:
                beamng.control.step(10)
            except Exception as e:
                print(f"[Main] Simulation step error: {e}")

            images = camera.poll()
            if images is None or 'colour' not in images:
                print(f"[Camera] Invalid camera poll response")
                continue
            
            img = np.array(images['colour'], dtype=np.uint8)

            # Send camera image to Foxglove
            try:
                timestamp_ns = get_timestamp_ns()
                bridge.send_camera_image(img, timestamp_ns, frame_id="camera")
            except Exception as camera_send_e:
                print(f"[Foxglove] Error sending camera image to Foxglove: {camera_send_e}")

            # Speed
            try:
                speed_mps, speed_kph, car_pos, direction = get_vehicle_speed(vehicle)
                speed_mps = abs(speed_mps)
                speed_kph = abs(speed_kph)
            except Exception as e:
                print(f"[Simulation] Speed retrieval error: {e}")
                continue

            # Lane Detection
            # bypassed docker aggregation, checks flags for each module
            try:
                start_proc = time.time()
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # conditional execution based on perception flags
                if enable_obj_det:
                    object_detections, _ = object_process(img_bgr, confidence_threshold=0.4, draw_detections=False, model=local_models.get('vehicle'))
                else:
                    object_detections = []
                    
                if enable_tl_det:
                    traffic_light_detections, _ = tl_process(img_bgr, confidence_threshold=0.2, draw_detections=False, model=local_models.get('traffic_light'))
                else:
                    traffic_light_detections = []
                    
                if enable_sign_det:
                    sign_detections, _ = sign_process(img_bgr, confidence_threshold=0.45, draw_detections=False, detection_model=local_models.get('sign_detect'), classification_model=local_models.get('sign_classify'))
                else:
                    sign_detections = []
                
                # added yolop process frame locally, checks flag
                if enable_yolop_flag:
                    try:
                        yolop_detections, drivable_area, yolop_lane_mask = yolop_process(
                            img_bgr, 
                            confidence_threshold=0.3, 
                            model=local_models.get('yolop_model'), 
                            device=local_models.get('device'), 
                            transforms=local_models.get('yolop_transforms'),
                            speed=speed_kph,
                            calibration_data=None,
                            vehicle_model='etk800'
                        )
                    except Exception as e:
                        print(f"[YOLOP] Error processing local yolop: {e}")
                        drivable_area = None
                        yolop_lane_mask = None
                else:
                    drivable_area = None
                    yolop_lane_mask = None
                
                if enable_cv_lane:
                    cv_result_image, metrics, cv_confidence = cv_lane_process(
                        img, 
                        speed=speed_kph,
                        previous_steering=previous_steering,
                        debug_display=debug_cv_lane,
                        perspective_debug_display=debug_perspective,
                        vehicle_model='etk800',
                        num_lanes=3,
                        yolop_lane_mask=yolop_lane_mask
                    )
                else:
                    cv_result_image = None
                    metrics = {'deviation': 0.0, 'lane_center': 0.0, 'vehicle_center': 0.0, 'confidence': 0.0}
                    cv_confidence = 0.0
                
                logger.info(f"Local processing latency: {(time.time()-start_proc)*1000:.1f}ms")
                
                # assign directly instead of extracting from aggregator result
                lane_metrics = metrics
                deviation = lane_metrics.get('deviation', 0.0)
                # ensure deviation is not None
                if deviation is None:
                    deviation = 0.0
                smoothed_deviation = lane_metrics.get('smoothed_deviation', deviation)
                if smoothed_deviation is None:
                    smoothed_deviation = deviation
                effective_deviation = lane_metrics.get('effective_deviation', deviation)
                if effective_deviation is None:
                    effective_deviation = deviation
                lane_center = lane_metrics.get('lane_center', 0.0)
                if lane_center is None:
                    lane_center = 0.0
                vehicle_center = lane_metrics.get('vehicle_center', 0.0)
                if vehicle_center is None:
                    vehicle_center = 0.0
                fused_confidence = lane_metrics.get('confidence', 0.0)
                if fused_confidence is None:
                    fused_confidence = 0.0
                
            except Exception as agg_e:
                print(f"[Main] Local perception error: {agg_e}")
                continue

            # Calculate yaw:
            # - car_yaw: for Foxglove 3D models (often rotated 180 deg)
            # - math_yaw: for actual MPC local physics math so it points forward
            car_yaw = np.arctan2(-direction[1], -direction[0])
            math_yaw = np.arctan2(direction[1], direction[0])

            # MPC Control logic
            try:
                # build current state vector to pass to the mpc controller
                current_state = np.array([
                    car_pos[0],           # x position
                    car_pos[1],           # y position
                    speed_mps,            # vx (longitudinal velocity)
                    0.0,                  # vy (lateral velocity)
                    math_yaw,             # theta (heading angle) - MUST use math_yaw
                    0.0                   # vtheta (angular velocity)
                ])
                
                # Build waypoints from lane center
                # Use detected lane deviation (in meters) to stay in lane
                waypoints = []
                
                # deviation > 0 means car is to the right of lane center, so we need to move left
                target_lateral_offset = effective_deviation 
                
                for i in range(10):  # N=10 steps ahead
                    t_ahead = (i + 1) * 0.1  # 0.1s per step
                    
                    # Local forward distance (cap minimum to ensure waypoints project if stopped)
                    forward_dist = max(speed_mps, 2.0) * t_ahead
                    
                    # Local lateral distance: gradually move towards the lane center
                    # Over 5 steps (0.5s), we fully apply the lateral offset to center the car
                    blend = min(1.0, (i + 1) / 5.0)
                    lateral_dist = target_lateral_offset * blend
                    
                    # Transform local coordinates (forward_dist, lateral_dist) to global (x, y)
                    # Using math_yaw assumes standard +X forward, +Y left relative math
                    x_pred = car_pos[0] + forward_dist * np.cos(math_yaw) - lateral_dist * np.sin(math_yaw)
                    y_pred = car_pos[1] + forward_dist * np.sin(math_yaw) + lateral_dist * np.cos(math_yaw)
                    
                    vx_ref = 15.0  # Target speed 15 m/s (~54 km/h)
                    waypoints.append((x_pred, y_pred, vx_ref))
                
                # build obstacles from obstacle detection (NOT YET IMPLEMENTED as we only have 2d detections but the mpc needs 3d positions)
                # This will allow MPC to proactively plan around obstacles instead of relying just on AEB
                obstacles = []
                if lidar_lane_boundaries is not None:
                    #placeholder
                    pass
                
                # Compute optimal control with MPC
                throttle, steering = mpc_controller.compute_control(
                    current_state=current_state,
                    waypoints=waypoints,
                    obstacles=obstacles,
                    target_speed=10.0
                )
                
                # AEB as fallback
                # MPC handles proactive avoidance, but AEB acts as safety net for imminent collisions
                aeb_triggered = False
                try:
                    radar_front = radars.get('radar_front', None)
                    if radar_front:
                        radar_result = radar_aeb_acc(radar_front, perception_cfg, speed_kph)
                        ttc = radar_result.get('ttc', float('inf'))
                        closest_distance = radar_result.get('closest_distance', float('inf'))
                        
                        if ttc <= 1.0:
                            # Emergency braking override MPC throttle
                            print(f"[AEB] EMERGENCY BRAKING TRIGGERED: TTC {ttc:.2f}s, Distance {closest_distance:.2f}m")
                            throttle = 0.0
                            aeb_triggered = True
                        elif ttc <= 2.5:
                            # Warning reduce throttle but let MPC handle steering
                            print(f"[AEB] WARNING: TTC {ttc:.2f}s, Distance {closest_distance:.2f}m - Reducing throttle")
                            throttle = max(0.0, throttle * 0.5)  # Reduce to half
                            aeb_triggered = True
                except Exception as aeb_e:
                    print(f"[AEB] Radar processing error: {aeb_e}")
                
                # Apply control to vehicle
                vehicle.control(throttle=float(throttle), steering=float(steering))
                previous_steering = steering
                
            except Exception as mpc_e:
                print(f"[MPC] Control computation error: {mpc_e}")
                throttle = 0.0
                steering = 0.0
                vehicle.control(throttle=0.0, steering=0.0)

            # Display CV lane detection window
            if debug_cv_lane and cv_result_image is not None:
                cv_disp = cv2.cvtColor(cv_result_image, cv2.COLOR_RGB2BGR) if len(cv_result_image.shape) == 3 else cv_result_image
                cv_disp = cv2.resize(cv_disp, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow('CV Lane Detection', cv_disp)

            # display drivable area and lanes with proper overlay visualization
            img_bgr_for_display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            yolop_displayed = False
            
            img_height = img_bgr_for_display.shape[0]
            
            if drivable_area is not None:
                try:
                    if drivable_area.size > 0:
                        drivable_area_roi = drivable_area.copy()
                        img_bgr_for_display = create_mask_overlay(img_bgr_for_display, drivable_area_roi, alpha=0.1, color=(0, 255, 0))
                        yolop_displayed = True
                except Exception as e:
                    print(f"[Visualization] Error displaying drivable area: {e}")
            
            if yolop_lane_mask is not None and np.sum(yolop_lane_mask) > 0:
                try:
                    img_bgr_for_display = create_mask_overlay(img_bgr_for_display, yolop_lane_mask, alpha=0.3, color=(255, 0, 0))
                    yolop_displayed = True
                except Exception as e:
                    print(f"[YOLOP] Error drawing YOLOP lane mask: {e}")
                    
            if debug_yolop and yolop_displayed:
                yolop_disp = cv2.resize(img_bgr_for_display, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow('YOLOP - Drivable Area & Lanes', yolop_disp)

            # Draw and show Combined object detections window
            if debug_obj_det:
                try:
                    img_bgr_for_obj = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    combined_img = draw_combined_detections(img_bgr_for_obj, sign_detections, object_detections, traffic_light_detections)
                    combined_disp = cv2.resize(combined_img, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow('Object Detections', combined_disp)
                except Exception as draw_e:
                    print(f"[Object Detection] Error drawing detections: {draw_e}")
                
            fused_confidence = lane_metrics.get('confidence', 0.0)
            
            # Lidar setup using car_yaw already calculated above
            lidar_offset = np.array([0.0, -0.35, 1.425])
            car_quat = yaw_rad_to_quaternion(car_yaw)
            rotation = R.from_quat([car_quat[0], car_quat[1], car_quat[2], car_quat[3]])
            lidar_pos_in_map = rotation.apply(lidar_offset) + car_pos
            lidar_yaw = car_yaw  # LiDAR has same yaw as vehicle

            try:
                lidar_lane_boundaries, filtered_points = lidar_process_frame(lidar, beamng=beamng, speed=speed_kph, debug_window=None, vehicle=vehicle, car_position=car_pos, car_direction=direction)
            except Exception as lidar_e:
                print(f"[LiDAR] Lidar process error: {lidar_e}")
                lidar_lane_boundaries = None
                filtered_points = None


            # Lidar Object Detection
            # lidar_detections, lidar_obj_img = lidar_object_detections(lidar, camera_detections=vehicle_detections)

            # Blind Spot Monitoring bsd
            try:
                radar_left = radars.get('radar_rear_left', None)
                radar_right = radars.get('radar_rear_right', None)
                if radar_left or radar_right:
                    bsd_status = radar_bsd(radar_left, radar_right, perception_cfg)
                    if bsd_status['left_warning']:
                        print(f"[BSD] Vehicle in left blind spot")
                    if bsd_status['right_warning']:
                        print(f"[BSD] Vehicle in right blind spot")
            except Exception as bsd_e:
                print(f"[BSD] Processing error: {bsd_e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1
            step_i += 1

            try:
                # Send vehicle control state (steering, throttle, brake)
                timestamp_ns = get_timestamp_ns()
                bridge.send_vehicle_control(
                    timestamp_ns=timestamp_ns,
                    speed_kph=speed_kph,
                    steering=steering,
                    throttle=throttle,
                    brake=0.0
                )
            except Exception as control_send_e:
                print(f"[Foxglove] Error sending vehicle control to Foxglove: {control_send_e}")

            try:
                # Send vehicle pose (PosesInFrame)
                quat_x, quat_y, quat_z, quat_w = yaw_rad_to_quaternion(car_yaw)
                timestamp_ns = get_timestamp_ns()
                bridge.send_vehicle_pose(
                    timestamp_ns=timestamp_ns,
                    x=car_pos[0],
                    y=car_pos[1],
                    z=car_pos[2],
                    quat_x=quat_x,
                    quat_y=quat_y,
                    quat_z=quat_z,
                    quat_w=quat_w,
                    frame_id="map"
                )
            except Exception as pose_send_e:
                print(f"[Foxglove] Error sending vehicle pose to Foxglove: {pose_send_e}")

            try:
                # Publish complete TF tree (map - base_link - lidar_top)
                quat_x, quat_y, quat_z, quat_w = yaw_rad_to_quaternion(car_yaw)
                timestamp_ns = get_timestamp_ns()
                bridge.send_tf_tree(
                    timestamp_ns=timestamp_ns,
                    x=car_pos[0],
                    y=car_pos[1],
                    z=car_pos[2],
                    quat_x=quat_x,
                    quat_y=quat_y,
                    quat_z=quat_z,
                    quat_w=quat_w
                )
            except Exception as tf_send_e:
                print(f"[Foxglove] Error publishing TF tree to Foxglove: {tf_send_e}")

            try:
                # Send LiDAR point cloud
                if filtered_points is not None and len(filtered_points) > 0:
                    timestamp_ns = get_timestamp_ns()
                    
                    bridge.send_lidar(
                        filtered_points,
                        timestamp_ns=timestamp_ns,
                        frame_id="map"
                    )
            except Exception as lidar_send_e:
                print(f"[Foxglove] Error sending LiDAR to Foxglove: {lidar_send_e}")

            try:
                timestamp_ns = get_timestamp_ns()
                all_detections = []
                
                for detection in object_detections:
                    all_detections.append({
                        'bbox': detection['bbox'],
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'type': 'vehicle'
                    })
                
                for sign_det in sign_detections:
                    all_detections.append({
                        'bbox': sign_det['bbox'],
                        'class': sign_det.get('classification', 'Sign'),
                        'confidence': sign_det.get('classification_confidence', 0.0),
                        'type': 'sign'
                    })
                
                for tl_det in traffic_light_detections:
                    all_detections.append({
                        'bbox': tl_det['bbox'],
                        'class': tl_det.get('class', 'Traffic Light'),
                        'confidence': tl_det.get('confidence', 0.0),
                        'type': 'traffic_light'
                    })
                
                if all_detections:
                    bridge.send_2d_detections(all_detections, timestamp_ns, image_width=1280, image_height=720)
                    
                    bridge.send_2d_detections_as_3d(
                        all_detections,
                        timestamp_ns,
                        camera_pos=car_pos + np.array([0, -1.3, 1.4]),
                        camera_dir=direction,
                        frame_id="map"
                    )
            except Exception as det_send_e:
                print(f"[Foxglove] Error sending detections: {det_send_e}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"[Main] Error: {e}")
    finally:
        cv2.destroyAllWindows()
        # if 'perception_client' in locals():
        #     perception_client.shutdown()
        beamng.close()

if __name__ == "__main__":
    main()