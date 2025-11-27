"""MQTT client for publishing detection events in Tapway format."""
import json
import time
import uuid
from datetime import datetime, timezone
import paho.mqtt.client as mqtt
from .config import MQTTConfig, CameraConfigManager


class TapwayMQTTPublisher:
    """MQTT client for publishing zone detection events in Tapway format."""
    
    def __init__(self, camera_id, mode="custom", config_path="cameras_config.json"):
        """Initialize MQTT publisher with Tapway format support."""
        self.camera_id = camera_id
        self.mode = mode
        self.config_path = config_path
        
        # Load MQTT configuration
        self.mqtt_config = MQTTConfig.load_mqtt_config(config_path)
        self.detection_config = MQTTConfig.get_detection_config(config_path)
        
        # MQTT settings
        self.broker = self.mqtt_config["broker"]
        self.port = self.mqtt_config["port"]
        self.username = self.mqtt_config["username"]
        self.password = self.mqtt_config["password"]
        self.topic = MQTTConfig.get_topic(camera_id, config_path)
        
        # Camera configuration
        self.camera_config = CameraConfigManager.get_camera_config(camera_id, 
                                                                   CameraConfigManager.load_config(config_path))
        
        # MQTT client
        self.client = None
        self.connected = False
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            self.connected = True
            print(f"✅ MQTT connection established")
        else:
            self.connected = False
            print(f"⚠️  MQTT connection failed with code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Callback when disconnected from MQTT broker."""
        self.connected = False
        if rc != 0:
            print(f"⚠️  MQTT unexpected disconnection (code: {rc}). Reconnecting...")
    
    def connect(self):
        """Connect to MQTT broker with authentication."""
        try:
            client_id = MQTTConfig.get_client_id(self.mode, self.camera_id)
            self.client = mqtt.Client(
                client_id=client_id,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
            
            # Set callbacks for connection management
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            # Set authentication if provided
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
                print(f"🔐 MQTT authentication set for user: {self.username}")
            
            # Enable automatic reconnection
            self.client.reconnect_delay_set(min_delay=1, max_delay=120)
            
            self.client.connect(self.broker, self.port, keepalive=60)
            
            # Start network loop in background thread
            self.client.loop_start()
            
            self.connected = True
            print(f"✅ MQTT connected: {self.broker}:{self.port}")
            print(f"📡 Publishing to topic: {self.topic}")
            return True
        except Exception as e:
            print(f"⚠️  MQTT connection failed: {e}")
            self.connected = False
            return False
    
    def create_tapway_event(self, detections_data, zone_name="zone_1"):
        """Create Tapway-formatted event message."""
        timestamp = datetime.now(timezone.utc).isoformat()
        unique_event_id = str(uuid.uuid4()).replace('-', '').upper()[:10]
        
        # Get camera resolution from config
        resolution = self.camera_config.get("stream_resolution", [640, 480])
        cam_resolution = resolution + [3]  # Add color channels
        
        # Build inout section (minimum required format)
        inout = {}
        if detections_data:
            zone_detections = {
                "class_name": [],
                "confidence": []
            }
            
            for detection in detections_data:
                zone_detections["class_name"].append(detection.get("class_name", "person"))
                zone_detections["confidence"].append(detection.get("confidence", 0.5))
            
            inout[zone_name] = zone_detections
        
        # Create the event payload
        event_payload = {
            "timestamp": timestamp,
            "device_id": self.camera_id,
            "camResolution": cam_resolution,
            "cameraID": self.camera_id,
            "cameraStage": 1,
            "eventTimeStamp": timestamp,
            "siteID": self.detection_config["site_id"],
            "subgroupID": self.detection_config["subgroup_id"],
            "uniqueEventID": unique_event_id,
            "inout": inout
        }
        
        return event_payload
    
    def publish_zone_event(self, tracker_id, class_id, class_name, confidence, fps, zone_name="zone_1"):
        """Publish detection event when object is in zone using Tapway format."""
        if self.client is None:
            print("⚠️  MQTT client not initialized")
            return False
        
        # Check if connected, if not, try to reconnect
        if not self.connected:
            print("⚠️  MQTT not connected, attempting reconnection...")
            try:
                self.connect()
            except Exception as e:
                print(f"⚠️  Reconnection failed: {e}")
                return False
        
        try:
            # Prepare detection data
            detections_data = [{
                "tracker_id": int(tracker_id),
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": float(confidence)
            }]
            
            # Create Tapway-formatted event
            event_payload = self.create_tapway_event(detections_data, zone_name)
            
            # Publish to camera-specific topic
            message = json.dumps(event_payload, indent=2)
            result = self.client.publish(self.topic, message, qos=1)
            
            # Wait for message to be published (with timeout)
            result.wait_for_publish(timeout=1.0)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"📡 {class_name} (ID:{tracker_id}) | Conf: {confidence:.2f} | Topic: {self.topic}")
                return True
            else:
                print(f"⚠️  MQTT publish failed with return code: {result.rc}")
                self.connected = False
                return False
                
        except Exception as e:
            print(f"⚠️  MQTT publish error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client is not None:
            # Stop the background network loop
            self.client.loop_stop()
            if self.connected:
                self.client.disconnect()
            self.connected = False
            print(f"📡 MQTT disconnected from {self.broker}:{self.port}")


# Backward compatibility alias
MQTTPublisher = TapwayMQTTPublisher