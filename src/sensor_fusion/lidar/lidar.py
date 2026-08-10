import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

"""
Utility function for collecting lidar point cloud data, to later be used for processing
"""

def collect_lidar_data(lidar_data):
    
    if lidar_data is None:
        print("LiDAR data is None")
        return []
    
    readings_data = lidar_data
    
    point_cloud = readings_data.get("pointCloud", [])

    return point_cloud