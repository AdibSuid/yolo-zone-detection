"""MQTT subscriber to monitor zone detection events."""
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime


MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "cv/zone_events"


def on_connect(client, userdata, flags, rc, properties=None):
    """Callback when connected to MQTT broker."""
    if rc == 0:
        print(f"✅ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        print(f"📡 Subscribing to topic: {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
        print("🎧 Listening for zone events... (Press Ctrl+C to stop)")
        print("-" * 60)
    else:
        print(f"❌ Failed to connect to MQTT broker. Return code: {rc}")


def on_message(client, userdata, msg):
    """Callback when message received."""
    try:
        payload = json.loads(msg.payload.decode())
        
        timestamp = datetime.fromtimestamp(payload.get('timestamp', time.time()))
        
        print(f"📅 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Event: {payload.get('event', 'unknown')}")
        print(f"🏷️  Object: {payload.get('class_name', 'N/A')}")
        print(f"🆔 Tracker ID: {payload.get('tracker_id', 'N/A')}")
        print(f"📊 Confidence: {payload.get('confidence', 0):.2f}")
        print(f"⚡ FPS: {payload.get('fps', 0):.1f}")
        print(f"🔧 Mode: {payload.get('mode', 'N/A')}")
        print("-" * 60)
        
    except json.JSONDecodeError:
        print(f"⚠️  Received non-JSON message: {msg.payload.decode()}")
    except Exception as e:
        print(f"❌ Error processing message: {e}")


def on_disconnect(client, userdata, rc, properties=None):
    """Callback when disconnected from broker."""
    print(f"🔌 Disconnected from MQTT broker")


def main():
    """Main entry point for MQTT subscriber."""
    client = mqtt.Client(
        client_id="zone_event_subscriber",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    
    try:
        print(f"🔄 Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
        
    except KeyboardInterrupt:
        print("\n👋 Stopping subscriber...")
        client.disconnect()
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()