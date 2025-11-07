#!/usr/bin/env python3
"""Simple MQTT test."""

import paho.mqtt.client as mqtt
import time

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ Connected successfully!")
    else:
        print(f"❌ Connection failed with code {rc}")
        if rc == 4:
            print("   Error: Bad username or password")

def test_simple_connection():
    client = mqtt.Client(
        client_id="simple_test",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    
    client.username_pw_set("tapway-admin", "T@pw4yAdm1n")
    client.on_connect = on_connect
    
    try:
        client.connect("localhost", 1883, 60)
        client.loop_start()
        time.sleep(3)
        client.loop_stop()
        client.disconnect()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_simple_connection()