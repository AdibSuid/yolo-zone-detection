"""MQTT subscriber to monitor zone detection events."""
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime


MQTT_BROKER = "localhost"
MQTT_PORT = 1883
# Subscribe to all Tapway zone events for cam1 (or use "+" for all cameras)
MQTT_TOPIC = "tapway/raw_event/metadata/cam1"


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

        # Get current time for display
        current_time = datetime.now()
        
        # Try to parse event timestamp for comparison
        ts_raw = payload.get('timestamp', None)
        if ts_raw:
            try:
                event_timestamp = datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
                # If event has timezone info, convert to local
                if event_timestamp.tzinfo:
                    event_timestamp = event_timestamp.astimezone()
            except Exception:
                event_timestamp = current_time
        else:
            event_timestamp = current_time

        print(f"📅 {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Print Tapway event details
        print(f"📡 Device ID: {payload.get('device_id', 'N/A')}")
        print(f"🆔 Unique Event ID: {payload.get('uniqueEventID', 'N/A')}")
        print(f"🏷️  InOut: {payload.get('inout', {})}")
        print(f"🔧 Site: {payload.get('siteID', 'N/A')} | Subgroup: {payload.get('subgroupID', 'N/A')}")
        print("-" * 60)

    except json.JSONDecodeError:
        print(f"⚠️  Received non-JSON message: {msg.payload.decode()}")
    except Exception as e:
        print(f"❌ Error processing message: {e}")


def on_disconnect(client, userdata, rc, *args, **kwargs):
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