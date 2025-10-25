"""MQTT client for publishing detection events."""
import json
import time
import paho.mqtt.client as mqtt
from .config import MQTTConfig


class MQTTPublisher:
    """MQTT client for publishing zone detection events."""
    
    def __init__(self, broker, port, mode):
        """Initialize MQTT publisher.
        
        Args:
            broker: MQTT broker address
            port: MQTT broker port
            mode: Performance mode name (for client ID)
        """
        self.broker = broker
        self.port = port
        self.mode = mode
        self.client = None
        self.connected = False
    
    def connect(self):
        """Connect to MQTT broker.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            client_id = MQTTConfig.get_client_id(self.mode)
            self.client = mqtt.Client(
                client_id=client_id,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
            self.client.connect(self.broker, self.port)
            self.connected = True
            print(f"✅ MQTT connected: {self.broker}:{self.port}")
            return True
        except Exception as e:
            print(f"⚠️  MQTT connection failed: {e}")
            self.connected = False
            return False
    
    def publish_zone_event(self, tracker_id, class_id, class_name, confidence, fps):
        """Publish detection event when object is in zone.
        
        Args:
            tracker_id: Object tracker ID
            class_id: Object class ID
            class_name: Object class name
            confidence: Detection confidence
            fps: Current FPS
        """
        if not self.connected or self.client is None:
            return
        
        try:
            payload = {
                "event": "inside_zone",
                "tracker_id": int(tracker_id),
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": float(confidence),
                "timestamp": time.time(),
                "fps": fps,
                "mode": self.mode
            }
            self.client.publish(MQTTConfig.TOPIC, json.dumps(payload))
            print(f"📡 {class_name} (ID:{tracker_id}) inside zone | Conf: {confidence:.2f}")
        except Exception as e:
            print(f"⚠️  MQTT publish error: {e}")
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client is not None and self.connected:
            self.client.disconnect()
            self.connected = False
