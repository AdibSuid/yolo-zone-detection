"""Optimized configuration for retail object detection on Intel CPU."""
import numpy as np


class PerformanceMode:
    """Performance mode configurations optimized for Intel CPU and GPU."""
    
    # Custom YOLOv8 Model Configuration
    # confidence: 0.5, image_size: 640, iou_threshold: 0.5
    CUSTOM = {
        "name": "Custom YOLOv8",
        "model": "best_openvino_model/",
        "resolution": (640, 640),  # Your training size
        "frame_skip": 1,  # Process every frame
        "conf_threshold": 0.5,  # Your config: confidence=0.5
        "iou_threshold": 0.5,  # Your config: iou_threshold=0.5
        "annotation_thickness": 2,
        "text_scale": 0.5,
        "device": "auto"  # Let OpenVINO choose best device (GPU if available)
    }
    
    # GPU-optimized mode for Intel Iris Xe
    CUSTOM_GPU = {
        "name": "Custom YOLOv8 (GPU)",
        "model": "best_openvino_model/",
        "resolution": (640, 640),  # Your training size
        "frame_skip": 1,  # Process every frame
        "conf_threshold": 0.5,  # Your config: confidence=0.5
        "iou_threshold": 0.5,  # Your config: iou_threshold=0.5
        "annotation_thickness": 2,
        "text_scale": 0.5,
        "device": "GPU"  # Force GPU usage
    }
    
    @classmethod
    def get_mode(cls, mode_name):
        """Get performance mode configuration."""
        modes = {
            "custom": cls.CUSTOM,  # Your custom YOLOv8 model
            "custom_gpu": cls.CUSTOM_GPU,  # GPU-optimized version
        }
        return modes.get(mode_name, cls.CUSTOM)  # Default to custom
    
    @classmethod
    def list_modes(cls):
        """Print all available performance modes."""
        print("📋 Available Performance Modes:")
        print("=" * 60)
        modes = [
            ("custom", cls.CUSTOM),  # Show custom first
            ("custom_gpu", cls.CUSTOM_GPU),  # GPU mode
        ]
        for key, config in modes:
            print(f"\n🔧 {key}:")
            print(f"   📝 {config['name']}")
            print(f"   📐 Resolution: {config['resolution'][0]}x{config['resolution'][1]}")
            print(f"   ⏭️  Frame Skip: {config['frame_skip']}")
            print(f"   🎚️  Confidence: {config['conf_threshold']}")
            print(f"   🔧 Model: {config['model']}")
            print(f"   🖥️  Device: {config.get('device', 'CPU')}")


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
    """MQTT broker configuration."""
    
    DEFAULT_BROKER = "localhost"
    DEFAULT_PORT = 1883
    TOPIC = "cv/zone_events"
    
    @staticmethod
    def get_client_id(mode):
        """Generate MQTT client ID."""
        return f"cv_publisher_{mode}"


class DisplayConfig:
    """Display window configuration."""
    
    SCALE_FACTOR = 1.0  # OPTIMIZED: No scaling for performance
    WINDOW_NAME_TEMPLATE = "YOLO Zone Detection - {mode_name}"