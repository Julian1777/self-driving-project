from math import cos, sin
import numpy as np

def get_raw_points(radar_data):
    if radar_data is None:
        return []
    elif isinstance(radar_data, np.ndarray):
        return radar_data.tolist() if len(radar_data) > 0 else []
    elif isinstance(radar_data, dict):
        if 'point_cloud' in radar_data:
            return radar_data['point_cloud']
        else:
            return radar_data.get('points', radar_data.get('readings', []))
    else:
        return []

def process_frame(radar_front_sensor, radar_cfg, speed_kph):
    """
    Process radar data for AEB and ACC.
    Returns raw radar data for decision logic in main loop.
    """
    radar_data = radar_front_sensor.poll()
    raw_points = get_raw_points(radar_data)
    filtered_points = filter_radar(raw_points, radar_cfg)

    converted_points = convert_to_xyz(filtered_points)

    # Calculate metrics (logic to be implemented)
    aeb_result = calculate_aeb(converted_points, speed_kph, radar_cfg)
    acc_result = calculate_acc(converted_points, speed_kph, radar_cfg)

    # Return combined result with raw data for beamng.py to make decisions
    return {
        'ttc': aeb_result.get('ttc', float('inf')),
        'closest_distance': aeb_result.get('closest_distance', None),
        'closest_velocity': aeb_result.get('closest_velocity', None),
        'converted_points': converted_points,
        'acc_adjustment': acc_result.get('throttle_adjustment', None)
    }


def calculate_aeb(converted_points, speed_kph, radar_cfg):
    """
    Calculate AEB metrics (TTC, distance, velocity).
    Logic to be implemented.

    TTC = Relative Distance / Relative Velocity
    Relative distance is the distance to the target
    Relative velocity is the difference between ego vehicle speed and target speed
    Relative velocity (Doppler velocity) is positive when the target is approaching, negative when receding.

    """
    min_dist = radar_cfg['aeb']['min_distance']
    closest_point = None

    for point in converted_points:
        x, y, z, doppler_vel, _, _ = point
        distance = np.sqrt(x**2 + y**2 + z**2) # Euclidean distance
        if distance < min_dist:
            min_dist = distance # Distance from ego vehicle to target
            closest_point = point
    if closest_point is not None:
        _, _, _, doppler_vel, _, _ = closest_point

        ego_speed_mps = speed_kph / 3.6
        relative_velocity = ego_speed_mps - doppler_vel

        ttc = min_dist / relative_velocity
        return {
            'ttc': ttc,
            'closest_distance': min_dist,
            'closest_velocity': doppler_vel
        }
    else:
        return {
            'ttc': float('inf'),
            'closest_distance': None,
            'closest_velocity': None
        }

def calculate_acc(converted_points, speed_kph, radar_cfg):
    """
    Calculate ACC metrics (throttle adjustment).
    Logic to be implemented.
    """
    return {
        'throttle_adjustment': None
    }


def convert_to_xyz(points):
    converted_points = []
    for point in points:
        try:
            range_dist = float(point[0])
            doppler_vel = float(point[1])
            azimuth_angle = float(point[2])
            elevation_angle = float(point[3])
            rcs = float(point[4])
            
            if len(point) > 6:
                metric = float(point[6])
            else:
                metric = float(point[5])

            azimuth_rad = np.deg2rad(azimuth_angle)
            elevation_rad = np.deg2rad(elevation_angle)

            x = range_dist * cos(elevation_rad) * cos(azimuth_rad)
            y = range_dist * cos(elevation_rad) * sin(azimuth_rad)
            z = range_dist * sin(elevation_rad)

            converted_points.append((x, y, z, doppler_vel, rcs, metric))
        except (ValueError, TypeError, IndexError):
            continue

    return converted_points


def filter_radar(raw_points, radar_cfg):

    filtered_points = []
    filtering_cfg = radar_cfg.get('radar_filtering', {})
    # parameters for filtering
    max_dist = filtering_cfg.get('max_range', 100.0)
    min_dist = filtering_cfg.get('min_range', 0.5)
    min_snr  = filtering_cfg.get('min_snr', 5.0)
    max_el   = filtering_cfg.get('max_elevation', 5.0)
    min_el   = filtering_cfg.get('min_elevation', -5.0)
    max_az   = filtering_cfg.get('max_azumith', 45.0)
    min_az   = filtering_cfg.get('min_azumith', -45.0)

    for point in raw_points:

        try:
            # extract point data
            range_dist = float(point[0])
            doppler_vel = float(point[1])
            azimuth_angle = float(point[2])
            elevation_angle = float(point[3])
            rcs = float(point[4])
            
            if len(point) > 6:
                quality = float(point[6])
                strong_signal = (quality >= 0.5)
            else:
                snr = float(point[5])
                strong_signal = (snr >= min_snr)

            within_range = (min_dist <= range_dist <= max_dist)
            elevation = (min_el <= elevation_angle <= max_el)
            azimuth = (min_az <= azimuth_angle <= max_az)

            if within_range and strong_signal and elevation and azimuth:
                # keep original point format
                filtered_points.append(point)
        except (ValueError, TypeError, IndexError):
            continue

    return filtered_points

def process_bsd_frame(radar_sensor, radar_cfg):
    """
    Process radar data for Blind Spot Detection (BSD).
    Returns True if an object is detected in the blind spot zone.
    """
    if radar_sensor is None:
        return False
        
    try:
        radar_data = radar_sensor.poll()
        raw_points = get_raw_points(radar_data)
    except Exception as e:
        print(f"[BSD] Error polling BSD radar: {e}")
        return False
        
    filtered_points = filter_radar(raw_points, radar_cfg)
    converted_points = convert_to_xyz(filtered_points)

    return calculate_bsd(converted_points, radar_cfg)

def calculate_bsd(converted_points, radar_cfg):
    """
    Returns True if an object is present in the blindspot.
    """
    # min and max distance for bsd zone
    bsd_cfg = radar_cfg.get('bsd', {})
    bsd_min_dist = bsd_cfg.get('min_distance', 0.5)
    bsd_max_dist = bsd_cfg.get('max_distance', 8.0)
    
    for point in converted_points:
        x, y, z, doppler_vel, _, _ = point
        distance = np.sqrt(x**2 + y**2)
        if bsd_min_dist < distance < bsd_max_dist:
            return True
            
    return False