#!/usr/bin/env python3
"""Simple test to check camera configuration."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Direct import from the file
from src.config import CameraConfigManager, MQTTConfig

def test_camera_config():
    """Test camera configuration loading."""
    print("📋 Testing camera configuration...")
    
    # Load configuration
    config = CameraConfigManager.load_config()
    print(f"Config loaded: {config}")
    
    # Get enabled cameras
    enabled_cameras = CameraConfigManager.get_enabled_cameras(config)
    print(f"Enabled cameras: {enabled_cameras}")
    
    # Test MQTT config
    mqtt_config = MQTTConfig.load_mqtt_config()
    print(f"MQTT config: {mqtt_config}")
    
    return enabled_cameras

if __name__ == "__main__":
    test_camera_config()