"""
Foxglove Bridge for BeamNG Simulation
Handles all communication with Foxglove Studio via WebSocket
"""

import json
import time
import struct
import numpy as np
import cv2
from pathlib import Path
import foxglove
from foxglove import start_server, Channel
from foxglove.channels import (
    PosesInFrameChannel,
    SceneUpdateChannel,
    PointCloudChannel,
    FrameTransformsChannel,
    CompressedImageChannel,
    LinePrimitiveChannel,
    ImageAnnotationsChannel,
)
from foxglove.schemas import (
    Timestamp,
    PointCloud,
    PackedElementField,
    PackedElementFieldNumericType,
    PosesInFrame,
    Pose,
    Quaternion,
    Vector3,
    SceneUpdate,
    SceneEntity,
    ModelPrimitive,
    CubePrimitive,
    CompressedImage,
    Color,
    FrameTransform,
    FrameTransforms,
    LinePrimitive,
    LinePrimitiveLineType,
    ImageAnnotations,
    PointsAnnotation,
    PointsAnnotationType,
    Point2,
    TextAnnotation,
    Duration,
)

class FoxgloveBridge:
    """Bridge class for sending data to Foxglove Studio"""
    
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.server = None
        self.channels = {}
        
    def start_server(self):
        """Start the Foxglove WebSocket server in a background thread"""
        try:
            self.server = start_server(
                name="BeamNG Simulation",
                host=self.host,
                port=self.port
            )
            print(f"Foxglove server started on {self.host}:{self.port}")
        except Exception as e:
            print(f"Error starting Foxglove server: {e}")
            raise
    
    def initialize_channels(self):
        """Initialize all channels for different data types"""
        
        # Vehicle control channel (JSON)
        self.channels['vehicle_control'] = Channel(
            topic="/vehicle_control",
            message_encoding="json",
            schema={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "integer"},
                    "speed_kph": {"type": "number"},
                    "steering": {"type": "number"},
                    "throttle": {"type": "number"},
                    "brake": {"type": "number"}
                }
            }
        )
        
        # Vehicle pose channel (PosesInFrame)
        self.channels['vehicle_pose'] = PosesInFrameChannel(topic="/vehicle_pose")
        
        # TF tree channel (FrameTransforms)
        self.channels['tf'] = FrameTransformsChannel(topic="/tf")
        
        # Scene update channel (for 3D model)
        self.channels['scene'] = SceneUpdateChannel(topic="/scene")
        
        # LiDAR point cloud channel (PointCloud)
        self.channels['lidar'] = PointCloudChannel(topic="/lidar")
        
        # Camera image channel (CompressedImage)
        self.channels['camera'] = CompressedImageChannel(topic="/camera/image/compressed")
        
        # Image annotations for all 2D bounding boxes
        self.channels['image_annotations'] = ImageAnnotationsChannel(topic="/camera/annotations")
        
        # 3D detections scene (for 3D bounding boxes using cubes)
        self.channels['detections_3d'] = SceneUpdateChannel(topic="/detections_3d")
        
        print("All Foxglove channels initialized")
    
    def _timestamp_to_time(self, timestamp_ns):
        """Convert nanoseconds timestamp to Timestamp message"""
        sec = timestamp_ns // 1_000_000_000
        nsec = timestamp_ns % 1_000_000_000
        return Timestamp(sec=sec, nsec=nsec)
    
    def send_vehicle_control(self, timestamp_ns, speed_kph, steering, throttle, brake):
        """Send vehicle control state"""
        try:
            if 'vehicle_control' not in self.channels:
                return
            message = {
                "timestamp": timestamp_ns,
                "speed_kph": float(speed_kph),
                "steering": float(steering),
                "throttle": float(throttle),
                "brake": float(brake)
            }
            self.channels['vehicle_control'].log(message)
        except Exception as e:
            pass
    
    def send_vehicle_pose(self, timestamp_ns, x, y, z, quat_x, quat_y, quat_z, quat_w, frame_id="map"):
        """Send vehicle pose as PosesInFrame in the map frame"""
        try:
            if 'vehicle_pose' not in self.channels:
                return
            timestamp = self._timestamp_to_time(timestamp_ns)
            
            # Send the pose of the vehicle (base_link) in the map frame
            # This visualizes where base_link is positioned in the world
            pose = PosesInFrame(
                timestamp=timestamp,
                frame_id=frame_id,
                poses=[
                    Pose(
                        position=Vector3(x=float(x), y=float(y), z=float(z)),
                        orientation=Quaternion(x=float(quat_x), y=float(quat_y), z=float(quat_z), w=float(quat_w))
                    )
                ]
            )
            
            self.channels['vehicle_pose'].log(pose)
        except Exception as e:
            pass
    
    def send_tf_tree(self, timestamp_ns, x, y, z, quat_x, quat_y, quat_z, quat_w):
        """Send TF tree with map -> base_link -> lidar_top and map -> base_link -> camera_front transforms"""
        try:
            if 'tf' not in self.channels:
                return
            timestamp = self._timestamp_to_time(timestamp_ns)
            
            # Build transforms for complete hierarchy:
            # map (root/world origin) - base_link (vehicle body at position)
            # lidar_top (LiDAR sensor mount)
            # camera_front (front camera mount)
            transforms = [
            # Vehicle position in world (map - base_link establishes where vehicle is)
            FrameTransform(
                timestamp=timestamp,
                parent_frame_id="map",
                child_frame_id="base_link",
                translation=Vector3(x=float(x), y=float(y), z=float(z)),
                rotation=Quaternion(x=float(quat_x), y=float(quat_y), z=float(quat_z), w=float(quat_w))
            ),
            # LiDAR sensor mount (base_link - lidar_top is fixed offset)
            FrameTransform(
                timestamp=timestamp,
                parent_frame_id="base_link",
                child_frame_id="lidar_top",
                translation=Vector3(x=0.0, y=-0.35, z=1.425),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ),
            # Front camera mount (base_link - camera_front is fixed offset)
            FrameTransform(
                timestamp=timestamp,
                parent_frame_id="base_link",
                child_frame_id="camera_front",
                translation=Vector3(x=0.0, y=-1.3, z=1.4),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            )
        ]
        
            # Send as FrameTransforms message
            tf_message = FrameTransforms(transforms=transforms)
            self.channels['tf'].log(tf_message)
        except Exception as e:
            pass
    
    def send_lidar(self, points, timestamp_ns, frame_id="lidar_top"):
        """
        Send LiDAR point cloud
        Args:
            points: numpy array of shape (N, 3) or (N, 4) with x, y, z, [intensity]
            timestamp_ns: timestamp in nanoseconds
            frame_id: frame ID for the point cloud (should be "lidar_top" for sensor-relative, or "map" for world-relative)
        """
        try:
            if 'lidar' not in self.channels or points is None or len(points) == 0:
                return
        
            timestamp = self._timestamp_to_time(timestamp_ns)
            
            if not isinstance(points, np.ndarray):
                points = np.array(points)
            
            if points.shape[1] == 3:
                # x, y, z only
                num_points = points.shape[0]
                point_stride = 12
                fields = [
                    PackedElementField(name="x", offset=0, type=PackedElementFieldNumericType.Float32),
                    PackedElementField(name="y", offset=4, type=PackedElementFieldNumericType.Float32),
                    PackedElementField(name="z", offset=8, type=PackedElementFieldNumericType.Float32)
                ]
                data = points.astype(np.float32).tobytes()
            elif points.shape[1] == 4:
                num_points = points.shape[0]
                point_stride = 16
                fields = [
                    PackedElementField(name="x", offset=0, type=PackedElementFieldNumericType.Float32),
                    PackedElementField(name="y", offset=4, type=PackedElementFieldNumericType.Float32),
                    PackedElementField(name="z", offset=8, type=PackedElementFieldNumericType.Float32),
                    PackedElementField(name="intensity", offset=12, type=PackedElementFieldNumericType.Float32)
                ]
                data = points.astype(np.float32).tobytes()
            else:
                return
            
            point_cloud = PointCloud(
                timestamp=timestamp,
                frame_id=frame_id,
                pose=Pose(
                    position=Vector3(x=0.0, y=0.0, z=0.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                ),
                point_stride=point_stride,
                fields=fields,
                data=data
            )
            
            self.channels['lidar'].log(point_cloud)
        except Exception as e:
            pass
    
    def send_camera_image(self, image, timestamp_ns, frame_id="camera"):
        """
        Send camera image as CompressedImage
        Args:
            image: numpy array (BGR format from OpenCV)
            timestamp_ns: timestamp in nanoseconds
            frame_id: frame ID for the image
        """
        try:
            if 'camera' not in self.channels or image is None:
                return
            
            timestamp = self._timestamp_to_time(timestamp_ns)
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            result, encoded_image = cv2.imencode('.jpg', image_rgb, encode_param)
            
            if not result:
                return
            
            compressed_image = CompressedImage(
                timestamp=timestamp,
                frame_id=frame_id,
                data=encoded_image.tobytes(),
                format="jpeg"
            )
            
            self.channels['camera'].log(compressed_image)
        except Exception as e:
            pass
    
    def send_2d_detections(self, detections, timestamp_ns, image_width=1280, image_height=720):
        """
        Send 2D bounding boxes as ImageAnnotations overlay on camera image
        Args:
            detections: list of detection dicts with 'bbox', 'class', 'confidence', 'type' (vehicle/sign/traffic_light)
            timestamp_ns: timestamp in nanoseconds
            image_width: width of the image in pixels
            image_height: height of the image in pixels
        """
        if not detections:
            return
        
        timestamp = self._timestamp_to_time(timestamp_ns)
        points_annotations = []
        text_annotations = []
        
        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            det_type = det.get('type', 'unknown')
            det_class = det.get('class', 'object')
            confidence = det.get('confidence', 0.0)
            
            if det_type == 'vehicle':
                outline_color = Color(r=0.0, g=1.0, b=0.0, a=1.0)
            elif det_type == 'sign':
                outline_color = Color(r=0.0, g=0.0, b=1.0, a=1.0)
            elif det_type == 'traffic_light':
                outline_color = Color(r=1.0, g=0.65, b=0.0, a=1.0)
            else:
                outline_color = Color(r=1.0, g=1.0, b=1.0, a=1.0)
            
            box_points = [
                Point2(x=float(x1), y=float(y1)),
                Point2(x=float(x2), y=float(y1)),
                Point2(x=float(x2), y=float(y2)),
                Point2(x=float(x1), y=float(y2)),
            ]
            
            points_annotation = PointsAnnotation(
                timestamp=timestamp,
                type=PointsAnnotationType.LINELIST,
                points=box_points,
                outline_color=outline_color,
                thickness=2.0
            )
            points_annotations.append(points_annotation)
            
            label = f"{det_class} {confidence:.2f}"
            text_annotation = TextAnnotation(
                timestamp=timestamp,
                position=Point2(x=float(x1), y=float(y1 - 5)),
                text=label,
                font_size=12.0,
                text_color=outline_color,
                background_color=Color(r=0.0, g=0.0, b=0.0, a=0.7)
            )
            text_annotations.append(text_annotation)
        
        image_annotations = ImageAnnotations(
            points=points_annotations,
            texts=text_annotations
        )
        
        self.channels['image_annotations'].log(image_annotations)
    
    def send_3d_detections(self, detections_3d, timestamp_ns, frame_id="map"):
        """
        Send 3D bounding boxes as CubePrimitives (for LiDAR object detection)
        Args:
            detections_3d: list of dicts with 'position' (x,y,z), 'size' (w,h,d), 'orientation' (quat), 'class', 'confidence', 'type'
            timestamp_ns: timestamp in nanoseconds
            frame_id: frame ID (default: "map")
        """
        if not detections_3d:
            return
        
        timestamp = self._timestamp_to_time(timestamp_ns)
        entities = []
        
        for i, det in enumerate(detections_3d):
            position = det.get('position', [0, 0, 0])
            size = det.get('size', [1, 1, 1])
            orientation = det.get('orientation', [0, 0, 0, 1])
            det_type = det.get('type', 'unknown')
            det_class = det.get('class', 'object')
            
            if det_type == 'vehicle':
                cube_color = Color(r=0.0, g=1.0, b=0.0, a=0.5)
            elif det_type == 'sign':
                cube_color = Color(r=0.0, g=0.0, b=1.0, a=0.5)
            elif det_type == 'traffic_light':
                cube_color = Color(r=1.0, g=0.65, b=0.0, a=0.5)
            else:
                cube_color = Color(r=1.0, g=1.0, b=1.0, a=0.5)
            
            cube = CubePrimitive(
                pose=Pose(
                    position=Vector3(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                    orientation=Quaternion(x=float(orientation[0]), y=float(orientation[1]), z=float(orientation[2]), w=float(orientation[3]))
                ),
                size=Vector3(x=float(size[0]), y=float(size[1]), z=float(size[2])),
                color=cube_color
            )
            
            entity = SceneEntity(
                timestamp=timestamp,
                frame_id=frame_id,
                id=f"detection_3d_{i}",
                cubes=[cube]
            )
            entities.append(entity)
        
        if entities:
            scene_update = SceneUpdate(entities=entities)
            self.channels['detections_3d'].log(scene_update)
    
    def send_2d_detections_as_3d(self, detections, timestamp_ns, camera_pos, camera_dir, frame_id="map"):
        """
        Convert 2D detections to approximate 3D positions for visualization
        Uses estimated depth based on detection type and size
        Args:
            detections: list of detection dicts with 'bbox', 'class', 'confidence', 'type'
            timestamp_ns: timestamp in nanoseconds
            camera_pos: camera position [x, y, z]
            camera_dir: camera direction vector [x, y, z]
            frame_id: frame ID (default: "map")
        """
        if not detections or camera_pos is None or camera_dir is None:
            return
        
        timestamp = self._timestamp_to_time(timestamp_ns)
        entities = []
        
        for i, det in enumerate(detections):
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            det_type = det.get('type', 'unknown')
            det_class = det.get('class', 'object')
            
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            if det_type == 'vehicle':
                cube_color = Color(r=0.0, g=1.0, b=0.0, a=0.3)
                estimated_depth = max(10.0, 50.0 * (100.0 / bbox_width))
                size = Vector3(x=4.5, y=1.8, z=1.5)
            elif det_type == 'sign':
                cube_color = Color(r=0.0, g=0.0, b=1.0, a=0.5)
                estimated_depth = max(5.0, 30.0 * (50.0 / bbox_width))
                size = Vector3(x=0.8, y=0.2, z=0.8)
            elif det_type == 'traffic_light':
                cube_color = Color(r=1.0, g=0.65, b=0.0, a=0.5)
                estimated_depth = max(10.0, 40.0 * (80.0 / bbox_height))
                size = Vector3(x=0.3, y=0.2, z=0.8)
            else:
                continue
            
            center_x_norm = ((x1 + x2) / 2 - 640) / 640
            center_y_norm = ((y1 + y2) / 2 - 360) / 360
            
            cam_forward = np.array(camera_dir)
            cam_forward = cam_forward / np.linalg.norm(cam_forward)
            
            cam_right = np.cross(cam_forward, np.array([0, 0, 1]))
            if np.linalg.norm(cam_right) > 0.001:
                cam_right = cam_right / np.linalg.norm(cam_right)
            else:
                cam_right = np.array([1, 0, 0])
            
            cam_up = np.cross(cam_right, cam_forward)
            cam_up = cam_up / np.linalg.norm(cam_up)
            
            offset = (
                estimated_depth * cam_forward +
                center_x_norm * 30.0 * cam_right +
                center_y_norm * -20.0 * cam_up
            )
            
            position = np.array(camera_pos) + offset
            
            cube = CubePrimitive(
                pose=Pose(
                    position=Vector3(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                ),
                size=size,
                color=cube_color
            )
            
            entity = SceneEntity(
                timestamp=timestamp,
                frame_id=frame_id,
                id=f"{det_type}_{i}",
                lifetime=Duration(sec=0, nsec=500_000_000),
                cubes=[cube]
            )
            entities.append(entity)
        
        if entities:
            scene_update = SceneUpdate(entities=entities)
            self.channels['detections_3d'].log(scene_update)

