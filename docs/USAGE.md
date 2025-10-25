# Usage Guide

Complete guide for using the YOLO Zone Detection system.

## Basic Usage

### Running the Detection System

```bash
# Default mode (balanced, camera 0)
python -m src.main

# Specify camera and mode
python -m src.main --camera 1 --mode ultra_fast

# Custom MQTT broker
python -m src.main --mqtt-broker 192.168.1.100 --mqtt-port 1883
```

### Command-Line Options

```
--mode {ultra_fast,maximum_fps,balanced,high_accuracy}
    Performance mode (default: balanced)

--camera INDEX
    Camera index (0=built-in, 1=USB, etc.)

--mqtt-broker ADDRESS
    MQTT broker address (default: localhost)

--mqtt-port PORT
    MQTT broker port (default: 1883)

--list-modes
    List all available performance modes
```

## Performance Modes

### Ultra Fast Mode
```bash
python -m src.main --mode ultra_fast --camera 1
```
- **Model**: YOLOv8n
- **Resolution**: 320×320
- **Best for**: Maximum FPS, quick prototyping
- **Expected FPS**: 20-30 FPS (depends on CPU)

### Maximum FPS Mode
```bash
python -m src.main --mode maximum_fps --camera 1
```
- **Model**: YOLOv8n
- **Resolution**: 416×416
- **Best for**: High-speed detection with good accuracy
- **Expected FPS**: 15-25 FPS

### Balanced Mode (Recommended)
```bash
python -m src.main --mode balanced --camera 1
```
- **Model**: YOLOv8s
- **Resolution**: 640×480
- **Best for**: General use, good accuracy/speed tradeoff
- **Expected FPS**: 10-15 FPS

### High Accuracy Mode
```bash
python -m src.main --mode high_accuracy --camera 1
```
- **Model**: YOLOv8s
- **Resolution**: 640×480
- **Best for**: Detection quality over speed
- **Expected FPS**: 8-12 FPS

## Camera Selection

### Finding Cameras

```bash
python -m tools.find_cameras
```

This will:
- List all available cameras
- Show resolution and FPS for each
- Provide interactive preview

**Output Example:**
```
✅ Camera 0 found:
   Resolution: 1280x720
   FPS: 30
   Backend: MSMF

✅ Camera 1 found:
   Resolution: 352x288
   FPS: 15
   Backend: DSHOW
```

### Using Specific Camera

```bash
# Built-in webcam (usually 0)
python -m src.main --camera 0

# USB camera (usually 1)
python -m src.main --camera 1

# Second USB camera (usually 2)
python -m src.main --camera 2
```

## MQTT Integration

### Starting MQTT Broker

```bash
cd mqtt-broker
docker-compose up -d
```

### Monitoring Events

```bash
# In a separate terminal
python -m tools.mqtt_subscriber
```

**Event Output:**
```
📅 2024-01-15 14:30:45
🎯 Event: inside_zone
🏷️  Object: person
🆔 Tracker ID: 123
📊 Confidence: 0.87
⚡ FPS: 15.2
🔧 Mode: balanced
```

### Event Format

Events are published to topic `cv/zone_events`:

```json
{
  "event": "inside_zone",
  "tracker_id": 123,
  "class_id": 0,
  "class_name": "person",
  "confidence": 0.87,
  "timestamp": 1705329045.123,
  "fps": 15.2,
  "mode": "balanced"
}
```

### Using Events in Your Code

```python
import paho.mqtt.client as mqtt
import json

def on_message(client, userdata, msg):
    event = json.loads(msg.payload.decode())
    
    if event['class_name'] == 'person':
        print(f"Person detected! ID: {event['tracker_id']}")
        # Your custom logic here

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message
client.connect("localhost", 1883)
client.subscribe("cv/zone_events")
client.loop_forever()
```

## Detection Zone Configuration

### Default Zone

- **Shape**: Rectangular box
- **Size**: 30% width × 40% height
- **Position**: Centered in frame
- **Color**: Red outline

### Customizing Zone Size

Edit `src/config.py`:

```python
class ZoneConfig:
    BOX_WIDTH_RATIO = 0.5   # 50% of frame width
    BOX_HEIGHT_RATIO = 0.6  # 60% of frame height
```

### Custom Polygon Zone

For non-rectangular zones, modify `create_box_polygon()` in `src/config.py`:

```python
@staticmethod
def create_box_polygon(frame_width, frame_height):
    # Triangle example
    return np.array([
        [frame_width // 2, 0],              # Top center
        [frame_width, frame_height],        # Bottom right
        [0, frame_height],                  # Bottom left
    ])
```

## Keyboard Controls

While the detection window is active:

- **'q'**: Quit the application
- **ESC**: Alternative quit (in some contexts)

## Display Window

### Window Size

Default scaling: **2.5×** for better visibility

Customize in `src/config.py`:

```python
class DisplayConfig:
    SCALE_FACTOR = 3.0  # Make window larger
```

### Display Overlays

The display shows:
- **Bounding boxes**: Around detected objects
- **Labels**: Class name and confidence
- **Zone**: Red box indicating detection area
- **FPS**: Current frames per second
- **Inference time**: Time taken for detection
- **Object count**: Total objects detected
- **Camera index**: Which camera is active

## Detectable Objects (80 COCO Classes)

### Common Objects

**People & Vehicles:**
- person, bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Animals:**
- bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Accessories:**
- backpack, umbrella, handbag, tie, suitcase

**Sports:**
- frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, surfboard, tennis racket

**Kitchen:**
- bottle, wine glass, cup, fork, knife, spoon, bowl

**Furniture:**
- chair, couch, bed, dining table, toilet

**Electronics:**
- tv, laptop, mouse, remote, keyboard, cell phone

**Appliances:**
- microwave, oven, toaster, sink, refrigerator

**Indoor:**
- book, clock, vase, scissors, teddy bear, hair dryer, toothbrush

And more! Total: 80 classes from COCO dataset.

## Performance Tuning

### Maximizing FPS

1. **Use faster mode**:
   ```bash
   python -m src.main --mode ultra_fast
   ```

2. **Reduce resolution** (edit `src/config.py`):
   ```python
   ULTRA_FAST = {
       "resolution": (256, 256),  # Smaller
       ...
   }
   ```

3. **Increase frame skip**:
   ```python
   "frame_skip": 2,  # Process every 2nd frame
   ```

### Improving Accuracy

1. **Use high accuracy mode**:
   ```bash
   python -m src.main --mode high_accuracy
   ```

2. **Lower confidence threshold**:
   ```python
   "conf_threshold": 0.2,  # Detect more objects
   ```

3. **No frame skipping**:
   ```python
   "frame_skip": 1,  # Process every frame
   ```

### Camera Optimization

**For Better Brightness:**
```python
# In src/camera.py
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Increase from 128
cap.set(cv2.CAP_PROP_CONTRAST, 150)
```

**For Higher FPS (may reduce brightness):**
```python
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -5)         # Reduce exposure time
```

## Troubleshooting

### No Objects Detected

- Lower confidence threshold in mode config
- Ensure objects are in the detection zone (red box)
- Check camera is focused and has good lighting
- Try `high_accuracy` mode

### False Detections

- Increase confidence threshold
- Use `balanced` or `high_accuracy` mode
- Improve lighting conditions

### Low FPS

- Use `ultra_fast` mode
- Close other applications
- Reduce resolution
- Increase frame skip

### MQTT Events Not Publishing

```bash
# Check broker is running
docker ps | grep mosquitto

# Check connection
python -m tools.mqtt_subscriber

# Verify objects are in the zone (red box)
```

## Advanced Usage

### Running Multiple Cameras

**Terminal 1:**
```bash
python -m src.main --camera 0 --mqtt-port 1883
```

**Terminal 2:**
```bash
python -m src.main --camera 1 --mqtt-port 1883
```

Both publish to same MQTT broker, different tracker IDs.

### Headless Mode (No Display)

Comment out display code in `src/main.py` or redirect display:

```python
# In process_frame(), comment out:
# cv2.imshow(...)
# cv2.waitKey(1)
```

### Custom Integration

```python
from src import ZoneDetectionApp

app = ZoneDetectionApp(
    mode="balanced",
    camera_index=1,
    mqtt_broker="localhost",
    mqtt_port=1883
)

if app.initialize():
    # Modify app behavior before running
    app.run()
```

---

**Happy detecting! 🎯** For issues, check [Troubleshooting](#troubleshooting) or open a GitHub issue.
