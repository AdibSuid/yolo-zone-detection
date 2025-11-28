from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import paho.mqtt.client as mqtt
import json
import logging
from typing import Dict, Optional
from threading import Lock
from contextlib import asynccontextmanager # <--- Added for lifespan

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state management
class KioskState:
    def __init__(self):
        self.inference_counter = 0
        self.is_subscribing = False
        self.mqtt_client = None
        self.lock = Lock()

    def reset_counter(self):
        with self.lock:
            self.inference_counter = 0

    def increment_counter(self, inout_data):
        with self.lock:
            self.inference_counter += len(inout_data.get("class_name", []))

    def get_counter(self):
        with self.lock:
            return self.inference_counter

# Global state instance
kiosk_state = KioskState()

# MQTT Configuration
MQTT_BROKER = "localhost" 
MQTT_PORT = 1883
MQTT_TOPIC = "tapway/raw_event/metadata/#"

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info(f"Connected to MQTT broker successfully | Topic {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
    else:
        logger.error(f"Failed to connect to MQTT broker with result code {rc}")

def on_mqtt_message(client, userdata, msg):
    try:
        message = json.loads(msg.payload.decode())
        logger.info(f"Received inference event: {message}")

        if not kiosk_state.is_subscribing:
            logger.info("Sells did not start this event will be ignored")
            return

        in_out_data = message.get("inout", {})
        for _, data in in_out_data.items():
            kiosk_state.increment_counter(data)

        logger.info(f"Inference counter incremented to: {kiosk_state.get_counter()}")

    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

def setup_mqtt():
    """Setup MQTT client"""
    client = mqtt.Client()
    username = "tapway-admin"
    password = "T@pw4yAdm1n"
    client.username_pw_set(username, password)
    client.on_connect = on_mqtt_connect
    client.on_message = on_mqtt_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start() 
        return client
    except Exception as e:
        logger.error(f"Failed to setup MQTT: {e}")
        return None

# --- NEW LIFESPAN HANDLER (Replaces @app.on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Application starting up...")
    kiosk_state.mqtt_client = setup_mqtt()
    
    yield # The app runs here
    
    # Shutdown logic
    logger.info("Application shutting down...")
    if kiosk_state.mqtt_client:
        kiosk_state.mqtt_client.loop_stop()
        kiosk_state.mqtt_client.disconnect()

# Initialize App with lifespan
app = FastAPI(title="Self-Checkout Kiosk API", version="1.0.0", lifespan=lifespan)

# Request/Response Models
class KioskRequest(BaseModel):
    itemcode: Optional[int] = None
    processid: int
    storeno: int
    kioskid: str
    Runningno: int

class DetectionResponse(BaseModel):
    DetectType: int

class HealthResponse(BaseModel):
    status: str = "OK"

# API Endpoints
@app.get("/")
def health():
    return {"status": "OK"}

@app.post("/api/handshake")
async def handle_kiosk_status(request: KioskRequest):
    process_id = request.processid

    try:
        if process_id == 0:
            logger.info("Health check requested")
            return HealthResponse()

        elif process_id == 1:
            logger.info(f"Sales start for kiosk {request.kioskid}, store {request.storeno}")
            kiosk_state.is_subscribing = True
            kiosk_state.reset_counter()
            return HealthResponse()

        elif process_id == 2:
            logger.info(f"Product scanning for item {request.itemcode}")
            current_counter = kiosk_state.get_counter()
            logger.info(f"Current inference counter: {current_counter}")

            detect_type = 0
            if current_counter == 1:
                detect_type = 0 # Good scan
                kiosk_state.reset_counter()
                logger.info("Good scan detected")
            elif current_counter >= 2:
                detect_type = 1 # Missed scan
                kiosk_state.reset_counter()
                logger.info("Missed scan detected")
            else: 
                detect_type = 0 # No inference, assume good scan
                logger.info("No inference detected, returning good scan")

            return DetectionResponse(DetectType=detect_type)

        elif process_id == 3:
            logger.info(f"Payment started for kiosk {request.kioskid}")
            kiosk_state.is_subscribing = False
            current_counter = kiosk_state.get_counter()

            if current_counter > 1:
                detect_type = 1
                kiosk_state.reset_counter()
                return DetectionResponse(DetectType=detect_type)

            return HealthResponse()

        elif process_id == 4:
            logger.info(f"Purchase completed for kiosk {request.kioskid}")
            kiosk_state.is_subscribing = False
            return HealthResponse()

        elif process_id == 5:
            logger.info(f"Sales closed for kiosk {request.kioskid}")
            kiosk_state.is_subscribing = False
            return HealthResponse()

        else:
            raise HTTPException(status_code=400, detail=f"Invalid processid: {process_id}")

    except Exception as e:
        logger.error(f"Error handling request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/status")
async def get_status():
    return {
        "is_subscribing": kiosk_state.is_subscribing,
        "inference_counter": kiosk_state.get_counter(),
        "mqtt_connected": kiosk_state.mqtt_client is not None
    }

@app.post("/reset")
async def reset_system():
    kiosk_state.reset_counter()
    kiosk_state.is_subscribing = False
    return {"status": "reset_complete"}

# --- FIXED MAIN BLOCK ---
if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 allows connection from other machines
    uvicorn.run(app, host="0.0.0.0", port=8000)