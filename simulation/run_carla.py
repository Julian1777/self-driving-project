import sys
import os
import yaml
import logging
import threading
from queue import Queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add CARLA PythonAPI to path
carla_path = r'C:\Users\user\Documents\CARLA_0.9.16\PythonAPI\carla'
sys.path.insert(0, carla_path)

from utils.pid_controller import PIDController

import carla

from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model

import torch
import numpy as np
import time
import math
import cv2
from scipy.spatial.transform import Rotation as R

from simulation.perception_client import PerceptionClient


from src.sensor_fusion.lidar.main import process_frame as lidar_process_frame
from src.sensor_fusion.radar.main import process_frame as radar_process_frame

from simulation.foxglove_integration.bridge_instance import bridge

logger = logging.getLogger(__name__)

MODELS = {}


class CARLASensorWrapper:
    """Wrapper to adapt CARLA sensor data to BeamNG sensor interface for sensor fusion functions."""
    def __init__(self, sensor_key, sensor_container, sensor_type='radar'):
        self.sensor_key = sensor_key
        self.sensor_container = sensor_container
        self.sensor_type = sensor_type
    
    def poll(self):
        """Return latest sensor data (matches BeamNG sensor.poll() interface)."""
        if self.sensor_type == 'radar':
            data = self.sensor_container.get_radar_detections()
            # Convert CARLA format to format expected by radar_aeb_acc
            # CARLA returns: [{'depth': ..., 'velocity': ..., 'azimuth': ..., 'altitude': ..., }, ...]
            # Need: list of tuples (range_dist, doppler_vel, azimuth, elevation, rcs, snr)
            if data is not None and len(data) > 0:
                converted = []
                for det in data:
                    point = (
                        det.get('depth', 0.0),           # range_dist
                        det.get('velocity', 0.0),        # doppler_vel
                        det.get('azimuth', 0.0),         # azimuth_angle
                        det.get('altitude', 0.0),        # elevation_angle
                        0.0,                             # rcs (not provided)
                        0.0                              # snr (not provided)
                    )
                    converted.append(point)
                return converted
            return []
        elif self.sensor_type == 'lidar':
            return self.sensor_container.get_lidar_points()
        else:
            return None


class SensorDataContainer:
    """
    Thread-safe container for storing latest sensor data from CARLA listeners.
    """
    def __init__(self):
        self.lock = threading.Lock()
        
        # Camera data
        self.camera_data = {}  # {camera_key: rgb_array}
        
        # LiDAR data
        self.lidar_data = None  # Nx3 array of points
        self.lidar_point_count = 0
        
        # Radar data
        self.radar_data = None  # Raw radar detection data
        
        # GPS/GNSS data
        self.gnss_data = {}  # {gnss_key: (latitude, longitude, altitude)}
        
        # IMU data
        self.imu_data = {}  # {imu_key: (accel, gyro, compass)}
        
    def set_camera_data(self, camera_key, image_array):
        """Store camera RGB data."""
        with self.lock:
            self.camera_data[camera_key] = image_array
    
    def get_camera_data(self, camera_key):
        """Retrieve camera RGB data."""
        with self.lock:
            return self.camera_data.get(camera_key)
    
    def set_lidar_data(self, points):
        """Store LiDAR point cloud data."""
        with self.lock:
            self.lidar_data = points
            self.lidar_point_count = len(points) if points is not None else 0
    
    def get_lidar_data(self):
        """Retrieve LiDAR point cloud data."""
        with self.lock:
            return self.lidar_data.copy() if self.lidar_data is not None else None

    get_lidar_points = get_lidar_data

    
    def set_radar_data(self, radar_data):
        """Store radar detection data."""
        with self.lock:
            self.radar_data = radar_data
    
    def get_radar_data(self):
        """Retrieve radar detection data."""
        with self.lock:
            return self.radar_data
    
    def set_gnss_data(self, gnss_key, lat, lon, alt):
        """Store GNSS/GPS data."""
        with self.lock:
            self.gnss_data[gnss_key] = (lat, lon, alt)
    
    def get_gnss_data(self, gnss_key):
        """Retrieve GNSS/GPS data."""
        with self.lock:
            return self.gnss_data.get(gnss_key)
    
    def set_imu_data(self, imu_key, accel, gyro, compass):
        """Store IMU data."""
        with self.lock:
            self.imu_data[imu_key] = (accel, gyro, compass)
    
    def get_imu_data(self, imu_key):
        """Retrieve IMU data."""
        with self.lock:
            return self.imu_data.get(imu_key)


# Global sensor data container (shared across callbacks)
sensor_container = SensorDataContainer()

def on_camera_image(image_data, camera_key):
    """
    Callback for RGB camera image. Converts CARLA image to numpy array.
    Args:
        image_data: CARLA image object
        camera_key: Identifier for the camera (e.g., 'camera_front')
    """
    try:
        # Convert CARLA image to numpy array
        array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
        array = array.reshape((image_data.height, image_data.width, 4))
        rgb_array = array[:, :, :3]  # Extract RGB channels only
        sensor_container.set_camera_data(camera_key, rgb_array)
    except Exception as e:
        logger.error(f"Error processing camera {camera_key}: {e}")


def on_lidar_point_cloud(point_cloud_data):
    """
    Callback for LiDAR point cloud. Converts CARLA point cloud to numpy array.
    Args:
        point_cloud_data: CARLA point cloud object
    """
    try:
        # Convert CARLA point cloud to numpy array (Nx4: x, y, z, intensity)
        points = np.frombuffer(point_cloud_data.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4))
        # Extract only XYZ coordinates (ignore intensity for now)
        points_xyz = points[:, :3]
        sensor_container.set_lidar_data(points_xyz)
    except Exception as e:
        logger.error(f"Error processing LiDAR: {e}")


def on_radar_detection(radar_data):
    """
    Callback for Radar detection data. Converts CARLA radar data to list format.
    Args:
        radar_data: CARLA radar object with detections
    """
    try:
        # Convert CARLA radar detections to list of dicts
        detections = []
        for detection in radar_data:
            detections.append({
                'velocity': detection.velocity,
                'azimuth': detection.azimuth,
                'altitude': detection.altitude,
                'depth': detection.depth
            })
        sensor_container.set_radar_data(detections)
    except Exception as e:
        logger.error(f"Error processing Radar: {e}")


def on_gnss_location(gnss_data, gnss_key):
    """
    Callback for GNSS/GPS location data.
    Args:
        gnss_data: CARLA GNSS data object
        gnss_key: Identifier for the GNSS sensor (e.g., 'gps_front')
    """
    try:
        latitude = gnss_data.latitude
        longitude = gnss_data.longitude
        altitude = gnss_data.altitude
        sensor_container.set_gnss_data(gnss_key, latitude, longitude, altitude)
    except Exception as e:
        logger.error(f"Error processing GNSS {gnss_key}: {e}")


def on_imu_measurement(imu_data, imu_key):
    """
    Callback for IMU measurement data. Combines accelerometer, gyroscope, compass.
    Args:
        imu_data: CARLA IMU data object
        imu_key: Identifier for the IMU (e.g., 'imu_1')
    """
    try:
        accel = np.array([imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z])
        gyro = np.array([imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z])
        compass = imu_data.compass
        sensor_container.set_imu_data(imu_key, accel, gyro, compass)
    except Exception as e:
        logger.error(f"Error processing IMU {imu_key}: {e}")


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
    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config'))
    
    # Load CARLA simulation config if available, otherwise use generic config
    carla_config_path = os.path.join(config_path, 'carla_sim.yaml')
    if os.path.exists(carla_config_path):
        with open(carla_config_path, 'r') as f:
            sim_config = yaml.safe_load(f)
    else:
        # Fallback: use basic CARLA defaults
        sim_config = {
            'simulation': {
                'host': 'localhost',
                'port': 2000,
                'timeout': 30.0,
                'synchronous_mode': True,
                'fixed_timestep': 0.05,
                'substeps': 1
            }
        }
    
    with open(os.path.join(config_path, 'scenarios.yaml'), 'r') as f:
        scenarios_config = yaml.safe_load(f)
    with open(os.path.join(config_path, 'sensors.yaml'), 'r') as f:
        sensors_config = yaml.safe_load(f)
    with open(os.path.join(config_path, 'control.yaml'), 'r') as f:
        control = yaml.safe_load(f)
    with open(os.path.join(config_path, 'perception.yaml'), 'r') as f:
        perception_config = yaml.safe_load(f)

    return sim_config, scenarios_config, sensors_config, control, perception_config


def sim_setup(map_name='Town04', scenario_type='highway', vehicle_blueprint='vehicle.tesla.model3'):
    """
    Setup CARLA simulation, world, vehicle, and sensors.
    
    Args:
        map_name (str): CARLA town name ('Town01' through 'Town13')
        scenario_type (str): Scenario type ('highway' or 'city') - for future expansion
        vehicle_blueprint (str): CARLA vehicle blueprint ID (e.g., 'vehicle.tesla.model3')
    
    Returns:
        tuple: (client, world, vehicle, sensor_container, vehicle_transform)
    """
    global sensor_container
    
    # Load configuration
    sim_config, scenarios_config, sensors_config, _, _ = load_config()
    sim_cfg = sim_config['simulation']
    
    # Connect to CARLA
    print(f"Connecting to CARLA server at {sim_cfg['host']}:{sim_cfg['port']}...")
    client = carla.Client(sim_cfg['host'], sim_cfg['port'])
    client.set_timeout(sim_cfg['timeout'])
    
    # Load world and map
    print(f"Loading map: {map_name}")
    world = client.load_world(map_name)
    
    # Set synchronous mode for deterministic simulation
    if sim_cfg['synchronous_mode']:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_timestep = sim_cfg['fixed_timestep']
        settings.substeps = sim_cfg['substeps']
        world.apply_settings(settings)
        print("Synchronous mode enabled")
    
    # Get blueprint library and spawn vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(vehicle_blueprint)
    vehicle_bp.set_attribute('role_name', 'ego_vehicle')
    

    # highway spawn point for Town04
    spawn_point = carla.Transform(carla.Location(x=83.75, y=9.92, z=11.40), carla.Rotation(pitch=0, yaw=0, roll=0))
    print(f"Spawning vehicle at spawn point 0")
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Vehicle spawned: {vehicle.id}")
    
    # Move spectator
    spectator = world.get_spectator()
    spectator_transform = carla.Transform(spawn_point.location + carla.Location(z=5, x=-8), carla.Rotation(pitch=-15, yaw=spawn_point.rotation.yaw))
    spectator.set_transform(spectator_transform)
    
    # Reset sensor container for fresh data
    sensor_container = SensorDataContainer()
    
    # Setup sensors from configuration
    sensors_config_data = sensors_config.get(vehicle_blueprint, sensors_config.get('default', {}))
    
    cameras = {}
    lidar_sensor = None
    radars = {}
    gnss_sensors = {}
    imu_sensors = {}
    
    # camera sensor
    for sensor_key, sensor_cfg in sensors_config_data.items():
        if sensor_key.startswith('camera_') and sensor_cfg.get('enabled', False):
            try:
                camera_bp = blueprint_library.find('sensor.camera.rgb')
                
                # Set camera attributes from config
                camera_bp.set_attribute('image_size_x', str(sensor_cfg['resolution'][0]))
                camera_bp.set_attribute('image_size_y', str(sensor_cfg['resolution'][1]))
                camera_bp.set_attribute('fov', str(sensor_cfg['field_of_view_y']))
                camera_bp.set_attribute('sensor_tick', str(sensor_cfg.get('requested_update_time', 0.0)))
                
                # Optional: post-processing effects
                if sensor_cfg.get('enable_postprocess_effects', False):
                    camera_bp.set_attribute('enable_postprocess_effects', 'true')
                
                # Create transform (position and rotation relative to vehicle)
                pos = sensor_cfg['pos']
                rot = sensor_cfg['dir']  # In CARLA, dir might be used as rotation indicator
                transform = carla.Transform(
                    carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                    carla.Rotation(pitch=0, yaw=0, roll=0)  # Adjust based on 'dir' if needed
                )
                
                camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
                
                # Register listener with callback (using lambda to capture camera_key)
                camera.listen(lambda image_data, key=sensor_key: on_camera_image(image_data, key))
                
                cameras[sensor_key] = camera
                print(f"Camera '{sensor_key}' initialized (resolution: {sensor_cfg['resolution']})")
                
            except Exception as e:
                print(f"Camera '{sensor_key}' initialization error: {e}")
                cameras[sensor_key] = None
    
    # lidar sensor
    for sensor_key, sensor_cfg in sensors_config_data.items():
        if sensor_key.startswith('lidar_') and sensor_cfg.get('enabled', False):
            try:
                lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
                
                # Set LiDAR attributes from config
                lidar_bp.set_attribute('channels', str(sensor_cfg.get('vertical_resolution', 64)))
                lidar_bp.set_attribute('range', str(sensor_cfg.get('max_distance', 120)))
                lidar_bp.set_attribute('points_per_second', str(sensor_cfg.get('points_per_second', 1000000)))
                lidar_bp.set_attribute('rotation_frequency', str(sensor_cfg.get('frequency', 20)))
                lidar_bp.set_attribute('upper_fov', str(sensor_cfg.get('upper_fov', 10)))
                lidar_bp.set_attribute('lower_fov', str(sensor_cfg.get('lower_fov', -30)))
                lidar_bp.set_attribute('horizontal_fov', str(sensor_cfg.get('horizontal_angle', 360)))
                lidar_bp.set_attribute('sensor_tick', str(sensor_cfg.get('requested_update_time', 0.0)))
                
                # Create transform
                pos = sensor_cfg['pos']
                transform = carla.Transform(
                    carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                    carla.Rotation(pitch=0, yaw=0, roll=0)
                )
                
                lidar_sensor = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
                lidar_sensor.listen(on_lidar_point_cloud)
                
                print(f"LiDAR '{sensor_key}' initialized (channels: {sensor_cfg.get('vertical_resolution', 64)}, range: {sensor_cfg.get('max_distance', 120)}m)")
                break  # Use first enabled LiDAR as primary
                
            except Exception as e:
                print(f"LiDAR '{sensor_key}' initialization error: {e}")
    
    # radar sensor
    for sensor_key, sensor_cfg in sensors_config_data.items():
        if sensor_key.startswith('radar_') and sensor_cfg.get('enabled', False):
            try:
                radar_bp = blueprint_library.find('sensor.other.radar')
                
                # Set Radar attributes from config
                radar_bp.set_attribute('horizontal_fov', str(sensor_cfg.get('field_of_view_y', 30)))
                radar_bp.set_attribute('vertical_fov', str(sensor_cfg.get('field_of_view_y', 30)))
                radar_bp.set_attribute('range', str(sensor_cfg.get('range_max', 150)))
                radar_bp.set_attribute('points_per_second', str(sensor_cfg.get('points_per_second', 1600)))
                radar_bp.set_attribute('sensor_tick', str(sensor_cfg.get('requested_update_time', 0.05)))
                
                # Create transform
                pos = sensor_cfg['pos']
                transform = carla.Transform(
                    carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                    carla.Rotation(pitch=0, yaw=0, roll=0)
                )
                
                radar = world.spawn_actor(radar_bp, transform, attach_to=vehicle)
                radar.listen(on_radar_detection)
                
                radars[sensor_key] = radar
                print(f"Radar '{sensor_key}' initialized (range: {sensor_cfg.get('range_max', 150)}m)")
                
            except Exception as e:
                print(f"Radar '{sensor_key}' initialization error: {e}")
                radars[sensor_key] = None
    
    # gnss sensor
    for sensor_key, sensor_cfg in sensors_config_data.items():
        if sensor_key.startswith('gps_') and sensor_cfg.get('enabled', False):
            try:
                gnss_bp = blueprint_library.find('sensor.other.gnss')
                
                # GNSS noise parameters (optional)
                gnss_bp.set_attribute('noise_alt_stddev', str(sensor_cfg.get('noise_alt_stddev', 0.0)))
                gnss_bp.set_attribute('noise_lat_stddev', str(sensor_cfg.get('noise_lat_stddev', 0.0)))
                gnss_bp.set_attribute('noise_lon_stddev', str(sensor_cfg.get('noise_lon_stddev', 0.0)))
                gnss_bp.set_attribute('sensor_tick', str(sensor_cfg.get('physics_update_time', 0.05)))
                
                # Create transform
                pos = sensor_cfg['pos']
                transform = carla.Transform(
                    carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                    carla.Rotation(pitch=0, yaw=0, roll=0)
                )
                
                gnss = world.spawn_actor(gnss_bp, transform, attach_to=vehicle)
                gnss.listen(lambda data, key=sensor_key: on_gnss_location(data, key))
                
                gnss_sensors[sensor_key] = gnss
                print(f"GNSS '{sensor_key}' initialized")
                
            except Exception as e:
                print(f"GNSS '{sensor_key}' initialization error: {e}")
                gnss_sensors[sensor_key] = None
    
    # imu sensor
    for sensor_key, sensor_cfg in sensors_config_data.items():
        if sensor_key.startswith('imu_') and sensor_cfg.get('enabled', False):
            try:
                imu_bp = blueprint_library.find('sensor.other.imu')
                
                # IMU noise parameters (optional)
                imu_bp.set_attribute('noise_accel_stddev_x', str(sensor_cfg.get('noise_accel_stddev_x', 0.0)))
                imu_bp.set_attribute('noise_accel_stddev_y', str(sensor_cfg.get('noise_accel_stddev_y', 0.0)))
                imu_bp.set_attribute('noise_accel_stddev_z', str(sensor_cfg.get('noise_accel_stddev_z', 0.0)))
                imu_bp.set_attribute('noise_gyro_stddev_x', str(sensor_cfg.get('noise_gyro_stddev_x', 0.0)))
                imu_bp.set_attribute('noise_gyro_stddev_y', str(sensor_cfg.get('noise_gyro_stddev_y', 0.0)))
                imu_bp.set_attribute('noise_gyro_stddev_z', str(sensor_cfg.get('noise_gyro_stddev_z', 0.0)))
                imu_bp.set_attribute('sensor_tick', str(sensor_cfg.get('physics_update_time', 0.01)))
                
                # Create transform
                pos = sensor_cfg['pos']
                transform = carla.Transform(
                    carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                    carla.Rotation(pitch=0, yaw=0, roll=0)
                )
                
                imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
                imu.listen(lambda data, key=sensor_key: on_imu_measurement(data, key))
                
                imu_sensors[sensor_key] = imu
                print(f"IMU '{sensor_key}' initialized")
                
            except Exception as e:
                print(f"IMU '{sensor_key}' initialization error: {e}")
                imu_sensors[sensor_key] = None
    
    # get primaries
    primary_camera = cameras.get('camera_front', next((v for v in cameras.values() if v is not None), None))
    primary_gnss = gnss_sensors.get('gps_front', next((v for v in gnss_sensors.values() if v is not None), None))
    primary_imu = imu_sensors.get('imu_1', next((v for v in imu_sensors.values() if v is not None), None))
    
    # wrap sensors to provide .poll() interface and keep data gathering the same
    wrapped_radar = {key: CARLASensorWrapper('radar', sensor_container, 'radar') for key in radars.keys()}
    wrapped_lidar = CARLASensorWrapper('lidar', sensor_container, 'lidar')
    
    print("Simulation setup complete")
    
    return client, world, vehicle, sensor_container, cameras, wrapped_lidar, wrapped_radar, gnss_sensors, imu_sensors

def get_vehicle_speed(vehicle):
    """
    Get the vehicle speed in m/s and kph, position, and direction vector.
    
    Args:
        vehicle: CARLA vehicle object
    
    Returns:
        tuple: (speed_mps, speed_kph, position, direction)
            - speed_mps: Speed in meters per second
            - speed_kph: Speed in kilometers per hour
            - position: np.array [x, y, z] in world coordinates
            - direction: np.array [x, y, z] forward direction vector (normalized)
    """
    try:
        # get velocity using api
        velocity = vehicle.get_velocity()
        speed_mps = velocity.length()
        speed_kph = speed_mps * 3.6  # Convert m/s to km/h
        
        # Get position
        location = vehicle.get_location()
        position = np.array([location.x, location.y, location.z])
        
        # Get direction (forward vector from rotation)
        transform = vehicle.get_transform()
        forward = transform.get_forward_vector()
        direction = np.array([forward.x, forward.y, forward.z])
        
        return speed_mps, speed_kph, position, direction
    
    except Exception as e:
        logger.error(f"Error getting vehicle speed: {e}")
        return 0.0, 0.0, np.array([0, 0, 0]), np.array([1, 0, 0])


def get_camera_image(camera_key='camera_front'):
    """
    Get latest camera image from sensor container.
    
    Args:
        camera_key (str): Camera identifier (e.g., 'camera_front')
    
    Returns:
        np.ndarray: RGB image or None if not yet received
    """
    return sensor_container.get_camera_data(camera_key)


def get_lidar_points():
    """
    Get latest LiDAR point cloud from sensor container.
    
    Returns:
        np.ndarray: Nx3 array of points or None if not yet received
    """
    return sensor_container.get_lidar_data()


def get_radar_detections():
    """
    Get latest Radar detections from sensor container.
    
    Returns:
        list: List of detection dicts or None if not yet received
    """
    return sensor_container.get_radar_data()


def get_gnss_location(gnss_key='gps_front'):
    """
    Get latest GNSS/GPS location from sensor container.
    
    Args:
        gnss_key (str): GNSS identifier (e.g., 'gps_front')
    
    Returns:
        tuple: (latitude, longitude, altitude) or None
    """
    return sensor_container.get_gnss_data(gnss_key)


def get_imu_measurement(imu_key='imu_1'):
    """
    Get latest IMU measurement from sensor container.
    
    Args:
        imu_key (str): IMU identifier (e.g., 'imu_1')
    
    Returns:
        tuple: (accel, gyro, compass) or None
    """
    return sensor_container.get_imu_data(imu_key)


def spawn_npc_traffic(world, blueprint_library, spawn_points, num_vehicles=3):
    """
    Spawn NPC vehicles with autopilot in CARLA world.
    
    Args:
        world: CARLA world object
        blueprint_library: CARLA blueprint library
        spawn_points: List of spawn points
        num_vehicles (int): Number of NPC vehicles to spawn
    
    Returns:
        list: List of spawned vehicle actors
    """
    npc_vehicles = []
    
    # skip first spawn since its for the ego vehicle
    available_spawns = spawn_points[1:min(1 + num_vehicles, len(spawn_points))]
    
    for spawn_point in available_spawns:
        try:
            # get random vehicle blueprints
            import random
            vehicle_list = list(blueprint_library.filter('vehicle.*'))
            vehicle_bp = random.choice(vehicle_list)
            
            # Don't use the same vehicle as player
            if vehicle_bp.id == 'vehicle.tesla.model3':
                vehicle_bp = [bp for bp in blueprint_library.filter('vehicle.*') 
                             if bp.id != 'vehicle.tesla.model3'][0]
            
            npc_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            npc_vehicle.set_autopilot(True)
            npc_vehicles.append(npc_vehicle)
            
            print(f"Spawned NPC vehicle: {vehicle_bp.id}")
        
        except Exception as e:
            logger.warning(f"Failed to spawn NPC vehicle: {e}")
    
    return npc_vehicles


def cleanup_carla(client, world, vehicle, sensors_dict, npc_vehicles=None):
    """
    Properly cleanup CARLA simulation by destroying actors and disabling synchronous mode.
    
    Args:
        client: CARLA client
        world: CARLA world
        vehicle: Player vehicle actor
        sensors_dict: Dict containing camera, lidar, radar, gnss, imu actors
        npc_vehicles: List of NPC vehicle actors (optional)
    """
    try:
        # Destroy sensors
        print("Cleaning up sensors...")
        for sensor_list in sensors_dict.values():
            if isinstance(sensor_list, dict):
                for sensor in sensor_list.values():
                    if sensor is not None:
                        sensor.destroy()
            elif sensor_list is not None:
                sensor_list.destroy()
        
        # Destroy NPC vehicles
        if npc_vehicles:
            print(f"Destroying {len(npc_vehicles)} NPC vehicles...")
            for npc in npc_vehicles:
                npc.destroy()
        
        # Destroy player vehicle
        print("Destroying player vehicle...")
        vehicle.destroy()
        
        # Disable synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        print("✓ CARLA cleanup complete")
    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    
    finally:
        client = None


def radar_aeb_acc(radar_front, perception_cfg, speed_kph):
    radar_cfg = perception_cfg['radar']
    radar_result = radar_process_frame(radar_front, radar_cfg, speed_kph)
    return radar_result


def draw_combined_detections(img, sign_detections, vehicle_detections, tl_detections):
    result_img = img.copy()
    
    # Draw Signs (Blue)
    for det in sign_detections:
        x1, y1, x2, y2 = det['bbox']
        classification = det.get('classification', 'Sign')
        conf = det.get('classification_confidence', 0.0)
        label = f"{classification} {conf:.2f}"
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(result_img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw Vehicles (Green)
    for det in vehicle_detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(result_img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw Traffic Lights (Orange)
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
    Main function to run the CARLA simulation.
    """

    print("Initializing aggregator client")
    perception_client = PerceptionClient(
        host='localhost',
        service_ports={
            # 'cv_lane_detection': 4777,
            'object_detection': 5777,
            'traffic_light_detection': 6777,
            'sign_detection': 7777,
            'sign_classification': 8777,
            'yolop': 9777
        },
        timeout=2.0,
        auto_health_check=True
    )
    print("Aggregator ready")

    # initialize carla with town4
    client, world, vehicle, sensor_container, cameras, lidar, radars, gps, imus = sim_setup(
        map_name='Town04',
        scenario_type='highway',
        vehicle_blueprint='vehicle.tesla.model3'
    )
    print("Simulation setup complete")

    # Get blueprint library and spawn points for NPC traffic
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    print("Waiting for sensors to initialize and gather data...")
    # Let sensors gather data by ticking world a few times
    npc_vehicles = []
    for i in range(50):
        world.tick()
        if i % 10 == 0:
            print(f"  Warmup tick {i}/50...")

    print("Testing sensor data availability...")
    try:
        camera_img = get_camera_image('camera_front')
        if camera_img is not None:
            print(f"Camera working: shape {camera_img.shape}")
        else:
            print("Camera: No data yet")
    except Exception as e:
        print(f"Camera error: {e}")

    try:
        lidar_points = get_lidar_points()
        if lidar_points is not None:
            print(f"LiDAR working: {len(lidar_points)} points")
        else:
            print("LiDAR: No data yet")
    except Exception as e:
        print(f"LiDAR error: {e}")

    try:
        radar_data = get_radar_detections()
        if radar_data is not None:
            print(f"Radar working: {len(radar_data)} detections")
        else:
            print("Radar: No data yet")
    except Exception as e:
        print(f"Radar error: {e}")

    try:
        gps_data = get_gnss_location('gps_front')
        if gps_data is not None:
            print(f"GPS working: lat={gps_data[0]:.4f}, lon={gps_data[1]:.4f}")
        else:
            print("GPS: No data yet")
    except Exception as e:
        print(f"GPS error: {e}")

    try:
        imu_data = get_imu_measurement('imu_1')
        if imu_data is not None:
            print(f"IMU working")
        else:
            print("IMU: No data yet")
    except Exception as e:
        print(f"IMU error: {e}")

    print("\nSpawning NPC traffic...")
    try:
        npc_vehicles = spawn_npc_traffic(world, blueprint_library, spawn_points, num_vehicles=3)
        print(f"Traffic spawned: {len(npc_vehicles)} NPC vehicles")
    except Exception as e:
        print(f"Traffic setup error: {e}")

    # Load control parameters from config
    _, _, _, control, perception_config = load_config()
    control_cfg = control['control']
    perception_cfg = perception_config['perception']

    steering_pid = PIDController(**control_cfg['steering_pid'])
    max_steering_change = control_cfg['max_steering_change']
    previous_steering = 0.0

    min_gap = control_cfg['min_gap']
    target_speed_kph = control_cfg['target_speed_kph']
    speed_pid = PIDController(
        Kp=control_cfg['speed_pid']['Kp'],
        Ki=control_cfg['speed_pid']['Ki'],
        Kd=control_cfg['speed_pid']['Kd']
    )

    # Speed control mode: 'cruise' (normal), 'adaptive' (ACC), or 'none' (manual)
    speed_control_mode = control_cfg['speed_control_mode']
    print(f"Speed control mode: {speed_control_mode}")
    
    # Start Foxglove Bridge
    try:
        bridge.start_server()
        bridge.initialize_channels()
        print("Foxglove bridge started")
    except Exception as e:
        print(f"Failed to start Foxglove bridge: {e}")

    frame_count = 0

    last_time = time.time()
    try:
        step_i = 0
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Step CARLA simulation (also triggers sensor listeners)
            try:
                world.tick()
            except Exception as e:
                print(f"Simulation step error: {e}")

            # Get latest camera image from sensor container
            img = get_camera_image('camera_front')
            if img is None:
                print("Waiting for camera data...")
                continue

            # Send camera image to Foxglove
            try:
                timestamp_ns = get_timestamp_ns()
                bridge.send_camera_image(img, timestamp_ns, frame_id="camera")
            except Exception as camera_send_e:
                print(f"Error sending camera image to Foxglove: {camera_send_e}")

            # Speed
            try:
                speed_mps, speed_kph, car_pos, direction = get_vehicle_speed(vehicle)
                speed_mps = abs(speed_mps)
                speed_kph = abs(speed_kph)
            except Exception as e:
                print(f"Speed retrieval error: {e}")
                continue

            # Lane Detection
            try:
                agg_result = perception_client.process_frame(
                    frame=img,
                    speed_kph=speed_kph,
                    timestamp_ns=get_timestamp_ns(),
                    vehicle_pos=car_pos,
                    vehicle_direction=direction
                )
                
                processing_time_ms = agg_result.processing_time_ms
                logger.info(f"Aggregation latency: {processing_time_ms:.1f}ms")
                
            except Exception as agg_e:
                print(f"Aggregation error: {agg_e}")
                import traceback
                traceback.print_exc()
                continue

            # Lane Metric extraction
            lane_metrics = perception_client.extract_lane_detection(agg_result)
            deviation = lane_metrics['deviation']
            smoothed_deviation = lane_metrics.get('smoothed_deviation', deviation)
            effective_deviation = lane_metrics.get('effective_deviation', deviation)
            lane_center = lane_metrics['lane_center']
            vehicle_center = lane_metrics['vehicle_center']
            fused_confidence = lane_metrics['confidence']

            # Extract CV lane detection results
            cv_lane_results = perception_client.extract_cv_lane_detection(agg_result)
            cv_confidence = cv_lane_results['confidence']
            cv_result_image = cv_lane_results['result_image']
            
            # Display CV lane detection window
            if cv_result_image is not None:
                cv2.imshow('CV Lane Detection', cv_result_image)

            # Extract other detections
            object_detections = perception_client.extract_object_detection(agg_result)
            traffic_light_detections = perception_client.extract_traffic_light_detection(agg_result)
            sign_detections = perception_client.extract_sign_detection(agg_result)

            # Extract YOLOP results
            yolop_results = perception_client.extract_yolop(agg_result)
            drivable_area = yolop_results['drivable_area']
            lane_lines = yolop_results['lane_lines']

            # Display drivable area window
            if drivable_area is not None and drivable_area.size > 0:
                drivable_area_img = cv2.resize(drivable_area.astype(np.uint8) * 255, (img.shape[1], img.shape[0]))
                cv2.imshow('YOLOP - Drivable Area', drivable_area_img)
            
            # Display lane lines window
            if lane_lines is not None and lane_lines.size > 0:
                lane_lines_img = cv2.resize(lane_lines.astype(np.uint8) * 255, (img.shape[1], img.shape[0]))
                cv2.imshow('YOLOP - Lane Lines', lane_lines_img)

            steering = steering_pid.update(-effective_deviation, dt)
            steering = np.clip(steering, -1.0, 1.0)
            steering_change = steering - previous_steering
            if abs(steering_change) > max_steering_change:
                steering = previous_steering + np.sign(steering_change) * max_steering_change

            throttle = cruise_control(target_speed_kph, speed_kph, speed_pid, dt)
            throttle = throttle * (1.0 - 0.3 * abs(steering))
            throttle = np.clip(throttle, 0.05, 0.3)

            fused_confidence = lane_metrics.get('confidence', 0.0)
            
            # Calculate vehicle yaw from direction
            car_yaw = np.arctan2(-direction[1], -direction[0])
            
            # LiDAR pose (offset from base_link + vehicle rotation/position)
            lidar_offset = np.array([0.0, -0.35, 1.425])
            car_quat = yaw_rad_to_quaternion(car_yaw)
            rotation = R.from_quat([car_quat[0], car_quat[1], car_quat[2], car_quat[3]])
            lidar_pos_in_map = rotation.apply(lidar_offset) + car_pos
            lidar_yaw = car_yaw  # LiDAR has same yaw as vehicle


            if step_i % 80 == 0:
                try:
                    combined_img = draw_combined_detections(img, sign_detections, object_detections, traffic_light_detections)
                except Exception as draw_e:
                    print(f"Error drawing detections: {draw_e}")

            try:
                # Use wrapped lidar sensor that provides .poll() interface for lidar_process_frame
                lidar_lane_boundaries, filtered_points = lidar_process_frame(lidar, beamng=None, speed=speed_kph, debug_window=None, vehicle=vehicle, car_position=car_pos, car_direction=direction)
            except Exception as lidar_e:
                print(f"Lidar process error: {lidar_e}")
                lidar_lane_boundaries = None
                filtered_points = None


            # Lidar Object Detection
            # lidar_detections, lidar_obj_img = lidar_object_detections(lidar, camera_detections=vehicle_detections)

            throttle = 0.0
            brake = 0.0

            if speed_control_mode == 'adaptive':
                try:
                    # Pass wrapped radar sensor that provides .poll() interface
                    radar_front = radars.get('radar_front', next(iter(radars.values())) if radars else None)
                    if radar_front is not None:
                        radar_result = radar_aeb_acc(radar_front, perception_cfg, speed_kph)
                    else:
                        radar_result = {'ttc': float('inf'), 'closest_distance': None, 'closest_velocity': None}

                    ttc = radar_result.get('ttc', float('inf'))
                    closest_distance = radar_result.get('closest_distance', float('inf'))
                    closest_velocity = radar_result.get('closest_velocity', float('inf'))

                    if ttc <= 1.0:
                        # full breaking
                        print(f"EMERGENCY BRAKING: TTC {ttc:.2f}s, Distance {closest_distance:.2f}m")
                        throttle = 0.0
                        brake = 1.0
                    elif ttc <= 3.0:
                        # medium breaking
                        print(f"MEDIUM BRAKING: TTC {ttc:.2f}s, Distance {closest_distance:.2f}m")
                        throttle = 0.0
                        brake = 0.3
                    elif ttc < float('inf'):
                        # Reduce throttle
                        print(f"WARNING: TTC {ttc:.2f}s, Distance {closest_distance:.2f}m")
                        throttle = cruise_control(target_speed_kph, speed_kph, speed_pid, dt) * 0.5
                        brake = 0.0
                    else:
                        # No object detected normal cruise control
                        throttle = cruise_control(target_speed_kph, speed_kph, speed_pid, dt)
                        brake = 0.0
                    
                except Exception as radar_e:
                    print(f"Radar processing error: {radar_e}")
                    throttle = cruise_control(target_speed_kph, speed_kph, speed_pid, dt)
                    brake = 0.0

            elif speed_control_mode == 'cruise':
                # Normal cruise control (no adaptive features)
                throttle = cruise_control(target_speed_kph, speed_kph, speed_pid, dt)
                brake = 0.0

            elif speed_control_mode == 'none':
                # No automatic speed control manual throttle
                throttle = 0.0
                brake = 0.0
            
            # Limit throttle based on steering angle to prevent spinning out
            throttle = throttle * (1.0 - 0.3 * abs(steering))
            throttle = np.clip(throttle, 0.05, 0.3)

            # Apply control to CARLA vehicle
            control = carla.VehicleControl()
            control.steer = steering
            control.throttle = throttle
            control.brake = brake
            vehicle.apply_control(control)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1
            step_i += 1

            try:
                timestamp_ns = get_timestamp_ns()
                lane_message = {
                    "timestamp": timestamp_ns,
                    "lane_center": float(lane_center) if lane_center is not None else 0.0,
                    "vehicle_center": float(vehicle_center) if vehicle_center is not None else 0.0,
                    "deviation": float(deviation) if deviation is not None else 0.0,
                    "confidence": float(fused_confidence)
                }
                if lidar_lane_boundaries and 'left_lane_points' in lidar_lane_boundaries:
                    lane_message["left_lane_points"] = [
                        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0}
                        for p in lidar_lane_boundaries['left_lane_points']
                    ]
                if lidar_lane_boundaries and 'right_lane_points' in lidar_lane_boundaries:
                    lane_message["right_lane_points"] = [
                        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0}
                        for p in lidar_lane_boundaries['right_lane_points']
                    ]
                bridge.lane_channel.log(lane_message)
            except Exception as lane_send_e:
                print(f"Error sending lane data to Foxglove: {lane_send_e}")
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
                print(f"Error sending vehicle control to Foxglove: {control_send_e}")

            try:
                # Send vehicle pose (PosesInFrame)
                car_yaw = np.arctan2(-direction[1], -direction[0])
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
                print(f"Error sending vehicle pose to Foxglove: {pose_send_e}")

            try:
                # Publish complete TF tree (map - base_link - lidar_top)
                car_yaw = np.arctan2(-direction[1], -direction[0])
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
                print(f"Error publishing TF tree to Foxglove: {tf_send_e}")

            try:
                car_yaw = np.arctan2(-direction[1], -direction[0])
                quat_x, quat_y, quat_z, quat_w = yaw_rad_to_quaternion(car_yaw)
                timestamp_ns = get_timestamp_ns()
                bridge.send_vehicle_3d(
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
            except Exception as vehicle_3d_send_e:
                print(f"Error sending vehicle 3D model to Foxglove: {vehicle_3d_send_e}")

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
                print(f"Error sending LiDAR to Foxglove: {lidar_send_e}")

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
                print(f"Error sending detections: {det_send_e}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        if 'perception_client' in locals():
            perception_client.shutdown()
        # Cleanup CARLA simulation
        cleanup_carla(client, world, vehicle, {
            'cameras': cameras,
            'lidar': lidar,
            'radars': radars,
            'gps': gps,
            'imu': imus
        }, npc_vehicles=npc_vehicles)

if __name__ == "__main__":
    main()