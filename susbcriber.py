import paho.mqtt.client as mqtt
import json

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected: {rc}")
    client.subscribe("tapway/raw_event/metadata/+")

def on_message(client, userdata, msg):
    event = json.loads(msg.payload.decode())
    print(f"📱 Received from {event['device_id']}: {event['inout']}")
    
    # YOUR BUSINESS LOGIC HERE
    process_detection_event(event)

def process_detection_event(event):
    # Example business logic
    camera_id = event['device_id']
    detections = event['inout']
    
    if camera_id == 'cam1':  # Checkout camera
        handle_checkout_event(detections)
    elif camera_id == 'cam2':  # Security camera
        handle_security_event(detections)

def handle_checkout_event(detections):
    """Handle events from checkout cameras (e.g., self-checkout monitoring)"""
    print("🛒 CHECKOUT EVENT DETECTED!")
    
    for zone_name, zone_data in detections.items():
        class_names = zone_data.get('class_name', [])
        confidences = zone_data.get('confidence', [])
        
        for class_name, confidence in zip(class_names, confidences):
            print(f"   🔍 Zone: {zone_name}")
            print(f"   📦 Object: {class_name}")
            print(f"   📊 Confidence: {confidence:.2f}")
            
            # Business logic for checkout
            if class_name == 'person' and confidence > 0.8:
                print("   ✅ Customer detected at checkout")
                # Could trigger: scan reminder, staff alert, etc.
            elif class_name in ['bottle', 'cup', 'book'] and confidence > 0.7:
                print(f"   🚨 Unscanned item detected: {class_name}")
                # Could trigger: alert, lock checkout, staff notification
    
    print("   " + "-" * 40)

def handle_security_event(detections):
    """Handle events from security cameras (e.g., restricted area monitoring)"""
    print("🔒 SECURITY EVENT DETECTED!")
    
    for zone_name, zone_data in detections.items():
        class_names = zone_data.get('class_name', [])
        confidences = zone_data.get('confidence', [])
        
        for class_name, confidence in zip(class_names, confidences):
            print(f"   🔍 Zone: {zone_name}")
            print(f"   👤 Object: {class_name}")
            print(f"   📊 Confidence: {confidence:.2f}")
            
            # Business logic for security
            if class_name == 'person' and confidence > 0.85:
                print("   🚨 SECURITY ALERT: Unauthorized person detected!")
                # Could trigger: record video, send alert, notify security
            elif class_name == 'car' and confidence > 0.8:
                print("   🚗 Vehicle detected in restricted area")
                # Could trigger: barrier activation, license plate capture
    
    print("   " + "-" * 40)

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set("tapway-admin", "T@pw4yAdm1n")
client.on_connect = on_connect
client.on_message = on_message

try:
    print("🔌 Connecting to MQTT broker...")
    client.connect("localhost", 1883, 60)  # Change to your MQTT broker IP if needed
    print("👂 Listening for detection events...")
    print("🔍 Subscribed to: tapway/raw_event/metadata/+")
    print("=" * 60)
    client.loop_forever()
except KeyboardInterrupt:
    print("\n👋 Subscriber stopped by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("💡 Make sure MQTT broker is running and accessible")
finally:
    client.disconnect()