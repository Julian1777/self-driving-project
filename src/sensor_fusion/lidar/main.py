from src.sensor_fusion.lidar.lidar import collect_lidar_data
import numpy as np

from .preprocessing import LidarPreprocessor

_preprocessor = LidarPreprocessor()

def process_frame(lidar_sensor, beamng, speed, debug_window=None, vehicle=None, car_position=None, car_direction=None):
    try:
        lidar_data = lidar_sensor.poll()
        if lidar_data is None:
            print("[Lidar] Warning: LiDAR sensor returned None")
            return {}, []

        point_cloud = collect_lidar_data(beamng, lidar_data)
        if len(point_cloud) == 0:
            print("[Lidar] Warning: Empty LiDAR point cloud")
            return {}, []

        filtered_points = point_cloud

        if len(filtered_points) == 0:
            print("[Lidar] Warning: No points after filtering")
            print(f"  Point cloud range: X=[{point_cloud[:, 0].min():.1f}, {point_cloud[:, 0].max():.1f}]")
            print(f"  Point cloud range: Y=[{point_cloud[:, 1].min():.1f}, {point_cloud[:, 1].max():.1f}]")
            print(f"  Point cloud range: Z=[{point_cloud[:, 2].min():.1f}, {point_cloud[:, 2].max():.1f}]")
            return {}, []

        # TEMPORARY: Bypass boundary detection and just return raw points for faster streaming
        return {}, filtered_points
    except Exception as e:
        print(f"[Lidar] LiDAR processing error: {e}")
        return {}, []