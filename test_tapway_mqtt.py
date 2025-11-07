#!/usr/bin/env python3
"""
Test script for Tapway MQTT system verification.
Tests MQTT publishing to camera-specific topics and JSON format validation.
"""

import json
import time
import threading
import argparse
from datetime import datetime
import paho.mqtt.client as mqtt
from src.config import CameraConfigManager, MQTTConfig
from src.mqtt_client import TapwayMQTTPublisher


class MQTTSubscriber:
    """MQTT subscriber to verify published messages."""
    
    def __init__(self, broker, port, username=None, password=None):
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.client = None
        self.messages_received = []
        self.connected = False
    
    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.connected = True
            print(f"✅ Subscriber connected to {self.broker}:{self.port}")
        else:
            print(f"❌ Subscriber connection failed with code {rc}")
            if rc == 4:
                print("   Error: Bad username or password")
            elif rc == 5:
                print("   Error: Not authorized")
    
    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n📨 [RECEIVED] {timestamp}")
            print(f"📡 Topic: {topic}")
            print(f"📄 Payload:")
            print(json.dumps(payload, indent=2))
            print("-" * 60)
            
            self.messages_received.append({
                "topic": topic,
                "payload": payload,
                "timestamp": timestamp
            })
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON received: {e}")
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
    def connect_and_subscribe(self, topic_pattern):
        """Connect to MQTT broker and subscribe to topic pattern."""
        try:
            self.client = mqtt.Client(
                client_id="tapway_test_subscriber",
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
            
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            
            self.client.connect(self.broker, self.port, 60)
            self.client.subscribe(topic_pattern)
            print(f"📡 Subscribed to: {topic_pattern}")
            
            # Start the loop in a separate thread
            self.client.loop_start()
            return True
            
        except Exception as e:
            print(f"❌ Subscriber connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print("📡 Subscriber disconnected")


def test_camera_mqtt_publishing(camera_id, duration=10):
    """Test MQTT publishing for a specific camera."""
    print(f"\n🧪 Testing MQTT publishing for camera: {camera_id}")
    print("=" * 60)
    
    # Load MQTT configuration
    mqtt_config = MQTTConfig.load_mqtt_config()
    topic = MQTTConfig.get_topic(camera_id)
    
    print(f"📡 MQTT Broker: {mqtt_config['broker']}:{mqtt_config['port']}")
    print(f"🔐 Username: {mqtt_config['username']}")
    print(f"📄 Topic: {topic}")
    
    # Start subscriber
    subscriber = MQTTSubscriber(
        mqtt_config['broker'],
        mqtt_config['port'],
        mqtt_config['username'],
        mqtt_config['password']
    )
    
    topic_pattern = f"{mqtt_config['topic_prefix']}/+"  # Subscribe to all camera topics
    if not subscriber.connect_and_subscribe(topic_pattern):
        return False
    
    # Wait for subscriber to connect
    time.sleep(2)
    
    # Initialize publisher
    try:
        publisher = TapwayMQTTPublisher(camera_id, "test")
        if not publisher.connect():
            print(f"❌ Failed to connect publisher for {camera_id}")
            subscriber.disconnect()
            return False
        
        print(f"✅ Publisher connected for {camera_id}")
        
    except Exception as e:
        print(f"❌ Publisher initialization failed: {e}")
        subscriber.disconnect()
        return False
    
    # Publish test messages
    print(f"\n📡 Publishing test messages for {duration} seconds...")
    start_time = time.time()
    message_count = 0
    
    while time.time() - start_time < duration:
        try:
            # Publish a test detection event
            success = publisher.publish_zone_event(
                tracker_id=message_count + 1,
                class_id=0,
                class_name="person",
                confidence=0.85 + (message_count % 10) * 0.01,
                fps=25.0,
                zone_name="zone_1"
            )
            
            if success:
                message_count += 1
                print(f"📤 Published message {message_count}")
            else:
                print(f"❌ Failed to publish message {message_count + 1}")
            
            time.sleep(2)  # Publish every 2 seconds
            
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error during publishing: {e}")
            break
    
    # Cleanup
    publisher.disconnect()
    time.sleep(1)  # Wait for last messages
    subscriber.disconnect()
    
    # Summary
    print(f"\n📊 Test Summary for {camera_id}:")
    print(f"   Messages published: {message_count}")
    print(f"   Messages received: {len(subscriber.messages_received)}")
    
    if subscriber.messages_received:
        print(f"\n✅ Successfully verified MQTT publishing to: {topic}")
        return True
    else:
        print(f"\n❌ No messages received - check MQTT configuration")
        return False


def validate_json_format(payload):
    """Validate if payload matches Tapway requirements."""
    required_fields = ["timestamp", "device_id", "inout"]
    missing_fields = []
    
    for field in required_fields:
        if field not in payload:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    
    # Validate inout structure
    inout = payload.get("inout", {})
    if not inout:
        print("❌ 'inout' section is empty")
        return False
    
    for zone_name, zone_data in inout.items():
        if "class_name" not in zone_data or "confidence" not in zone_data:
            print(f"❌ Zone '{zone_name}' missing required fields")
            return False
    
    print("✅ JSON format validation passed")
    return True


def list_camera_topics():
    """List all camera topics that would be used."""
    print("\n📋 Camera Topics Configuration:")
    print("=" * 50)
    
    config = CameraConfigManager.load_config()
    mqtt_config = MQTTConfig.load_mqtt_config()
    
    print(f"📡 MQTT Broker: {mqtt_config['broker']}:{mqtt_config['port']}")
    print(f"🔐 Username: {mqtt_config['username']}")
    print(f"📄 Topic Prefix: {mqtt_config['topic_prefix']}")
    print()
    
    cameras = config.get("cameras", {})
    if not cameras:
        print("⚠️  No cameras found in configuration")
        return
    
    for camera_id, camera_config in cameras.items():
        topic = MQTTConfig.get_topic(camera_id)
        status = "🟢 ENABLED" if camera_config.get("enabled", False) else "🔴 DISABLED"
        
        print(f"🎥 {camera_id}: {status}")
        print(f"   Topic: {topic}")
        print(f"   Source: {camera_config.get('path', 'N/A')}")
        print(f"   Type: {camera_config.get('source_type', 'N/A')}")
        print()


def main():
    """Main function for MQTT testing."""
    parser = argparse.ArgumentParser(
        description="Test Tapway MQTT system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test specific camera
  python test_tapway_mqtt.py --camera-id cam-1

  # Test with custom duration
  python test_tapway_mqtt.py --camera-id cam-2 --duration 30

  # List all camera topics
  python test_tapway_mqtt.py --list-topics

  # Test all enabled cameras
  python test_tapway_mqtt.py --test-all
        """
    )
    
    parser.add_argument("--camera-id", type=str,
                       help="Test specific camera ID (e.g., cam-1)")
    
    parser.add_argument("--duration", type=int, default=10,
                       help="Test duration in seconds (default: 10)")
    
    parser.add_argument("--list-topics", action="store_true",
                       help="List all camera topics and exit")
    
    parser.add_argument("--test-all", action="store_true",
                       help="Test all enabled cameras")
    
    args = parser.parse_args()
    
    if args.list_topics:
        list_camera_topics()
        return
    
    if args.test_all:
        enabled_cameras = CameraConfigManager.get_enabled_cameras()
        if not enabled_cameras:
            print("⚠️  No enabled cameras found in configuration")
            return
        
        print(f"🧪 Testing all enabled cameras: {enabled_cameras}")
        for camera_id in enabled_cameras:
            success = test_camera_mqtt_publishing(camera_id, args.duration)
            if not success:
                print(f"❌ Test failed for {camera_id}")
                break
        return
    
    if not args.camera_id:
        print("⚠️  Must specify --camera-id, --test-all, or --list-topics")
        parser.print_help()
        return
    
    # Test specific camera
    test_camera_mqtt_publishing(args.camera_id, args.duration)


if __name__ == "__main__":
    main()