"""Optimized configuration for retail object detection on Intel CPU."""
import json
import os
from pathlib import Path
import numpy as np


class CameraConfigManager:
    """Manages camera configurations from JSON file."""
    
    @staticmethod
    def load_config(config_path="cameras_config.json"):
        """Load camera configuration from JSON file."""
        # Look for config file in project root
        if not os.path.isabs(config_path):
            project_root = Path(__file__).parent.parent
            config_path = project_root / config_path
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Camera config file not found: {config_path}")
            return {"cameras": {}, "mqtt": {}, "detection": {}}
        except json.JSONDecodeError as e:
            print(f"⚠️  Invalid JSON in camera config: {e}")
            return {"cameras": {}, "mqtt": {}, "detection": {}}
    
    @staticmethod
    def get_enabled_cameras(config=None):
        """Get list of enabled camera IDs."""
        if config is None:
            config = CameraConfigManager.load_config()
        
        enabled_cameras = []
        for cam_id, cam_config in config.get("cameras", {}).items():
            if cam_config.get("enabled", False):
                enabled_cameras.append(cam_id)
        return enabled_cameras
    
    @staticmethod
    def get_camera_config(camera_id, config=None):
        """Get configuration for specific camera."""
        if config is None:
            config = CameraConfigManager.load_config()
        
        return config.get("cameras", {}).get(camera_id, {})


class PerformanceMode:
    """Performance mode configurations optimized for Intel CPU."""
    
    # Custom YOLOv8 Model Configuration
    # confidence: 0.5, image_size: 640, iou_threshold: 0.5
    CUSTOM = {
        "name": "Custom YOLOv8",
        "model": "best_openvino_model/",
        "resolution": (640, 480),  # Match camera's actual capability
        "frame_skip": 1,  # Process every frame
        "conf_threshold": 0.5,  # Your config: confidence=0.5
        "iou_threshold": 0.5,  # Your config: iou_threshold=0.5
        "annotation_thickness": 2,
        "text_scale": 0.5
    }
    
    @classmethod
    def get_mode(cls, mode_name):
        """Get performance mode configuration."""
        return cls.CUSTOM  # Only one mode supported now
    
    @classmethod
    def list_modes(cls):
        """Print available performance mode."""
        print("📋 Available Performance Mode:")
        print("=" * 60)
        print(f"\n🔧 custom:")
        print(f"   📝 {cls.CUSTOM['name']}")
        print(f"   📐 Resolution: {cls.CUSTOM['resolution'][0]}x{cls.CUSTOM['resolution'][1]}")
        print(f"   ⏭️  Frame Skip: {cls.CUSTOM['frame_skip']}")
        print(f"   🎚️  Confidence: {cls.CUSTOM['conf_threshold']}")
        print(f"   🔧 Model: {cls.CUSTOM['model']}")


class CameraConfig:
    """Optimized camera configuration for maximum FPS."""
    
    # OPTIMIZED: Auto exposure for better brightness
    AUTO_EXPOSURE_MODE = 0.75  # Auto exposure (3 in OpenCV = auto, 0.75 enables it)
    EXPOSURE = -5  # Slightly faster than default, but not too dark
    BRIGHTNESS = 150  # Increased brightness
    CONTRAST = 128  # Standard contrast
    BUFFER_SIZE = 1
    FPS_TARGET = 30  # Request 30 FPS from camera
    
    # OPTIMIZED: Reduced warmup time
    WARMUP_TIME = 0.5  # seconds
    WARMUP_FRAMES = 5
    
    # Reliability settings
    MAX_FAILED_READS = 10


class ZoneConfig:
    """Detection zone configuration."""
    
    BOX_WIDTH_RATIO = 0.3  # 30% of frame width
    BOX_HEIGHT_RATIO = 0.4  # 40% of frame height
    
    @staticmethod
    def create_box_polygon(frame_width, frame_height):
        """Create centered box polygon for zone detection."""
        box_width = int(frame_width * ZoneConfig.BOX_WIDTH_RATIO)
        box_height = int(frame_height * ZoneConfig.BOX_HEIGHT_RATIO)
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        return np.array([
            [center_x - box_width // 2, center_y - box_height // 2],
            [center_x + box_width // 2, center_y - box_height // 2],
            [center_x + box_width // 2, center_y + box_height // 2],
            [center_x - box_width // 2, center_y + box_height // 2],
        ])


class MQTTConfig:
    """MQTT broker configuration with Tapway format support."""
    
    DEFAULT_BROKER = "localhost"
    DEFAULT_PORT = 1883
    TOPIC = "cv/zone_events"  # Legacy topic
    
    @staticmethod
    def load_mqtt_config(config_path="cameras_config.json"):
        """Load MQTT configuration from JSON file."""
        config = CameraConfigManager.load_config(config_path)
        mqtt_config = config.get("mqtt", {})
        
        return {
            "broker": mqtt_config.get("broker", MQTTConfig.DEFAULT_BROKER),
            "port": mqtt_config.get("port", MQTTConfig.DEFAULT_PORT),
            "username": mqtt_config.get("username"),
            "password": mqtt_config.get("password"),
            "topic_prefix": mqtt_config.get("topic_prefix", "tapway/raw_event/metadata")
        }
    
    @staticmethod
    def get_client_id(mode, camera_id=None):
        """Generate MQTT client ID."""
        if camera_id:
            return f"tapway_publisher_{camera_id}_{mode}"
        return f"cv_publisher_{mode}"
    
    @staticmethod
    def get_topic(camera_id, config_path="cameras_config.json"):
        """Get MQTT topic for specific camera."""
        mqtt_config = MQTTConfig.load_mqtt_config(config_path)
        topic_prefix = mqtt_config["topic_prefix"]
        return f"{topic_prefix}/{camera_id}"
    
    @staticmethod
    def get_detection_config(config_path="cameras_config.json"):
        """Load detection configuration from JSON file."""
        config = CameraConfigManager.load_config(config_path)
        detection_config = config.get("detection", {})
        
        return {
            "site_id": detection_config.get("site_id", "TAPWAY"),
            "subgroup_id": detection_config.get("subgroup_id", "Live Cam"),
            "model_name": detection_config.get("model_name", "PeopleNet.2025.9.1"),
            "model_version": detection_config.get("model_version", "0.0.1")
        }


class DisplayConfig:
    """Display window configuration."""
    
    SCALE_FACTOR = 1.0  # OPTIMIZED: No scaling for performance
    WINDOW_NAME_TEMPLATE = "YOLO Zone Detection - {mode_name}"