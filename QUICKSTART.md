# 🚀 Quick Start Guide

Get up and running with YOLO Zone Detection in **5 minutes**.

---

## ⚡ Step-by-Step Setup

###  **1. Clone & Navigate**

```bash
git clone https://github.com/AdibSuid/yolo-zone-detection.git
cd yolo-zone-detection
```

### **2. Create Python Environment**

```bash
python3.10 -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
.\venv\Scripts\activate  # Windows PowerShell
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Prepare Your YOLO Model**

Place your trained `best.pt` YOLOv8 model in the project root, then export it:

```bash
python scripts/export.py
```

✅ This creates `best_openvino_model/` with optimized Intel CPU/GPU inference files.

### **5. Configure Camera**

```bash
# Find your camera
python -m tools.find_cameras
```

Create `cameras_config.json` (copy from `cameras_config_examples.json`):

```json
{
  "cameras": {
    "cam1": {
      "enabled": true,
      "source_type": "USB",
      "path": 0,
      "stream_resolution": [640, 480],
      "stream_fps": 30,
      "stream_format": "MJPG"
    }
  },
  "mqtt": {
    "broker": "localhost",
    "port": 1883,
    "username": "tapway",
    "password": "tapway",
    "topic_prefix": "tapway/raw_event/metadata"
  },
  "detection": {
    "site_id": "TAPWAY",
    "subgroup_id": "Live Cam",
    "model_name": "YOLOv8",
    "model_version": "1.0.0"
  }
}
```

### **6. Start MQTT Broker** *(Optional)*

```bash
cd mqtt-broker
docker compose up -d
cd ..
```

### **7. Run Detection System**

```bash
# Basic detection with OpenCV window
python -m src.main --camera-id cam1

# With web dashboard (http://localhost:5000)
python -m src.main --camera-id cam1 --web

# Maximum performance (no display)
python -m src.main --camera-id cam1 --no-display
```

---

## 🎯 What You Should See

### Terminal Output:
```
🚀 OPTIMIZED YOLO Zone Detection for Intel CPU
   Mode: Custom YOLOv8
   Model: best_openvino_model/
   Resolution: 640x480
   Confidence: 0.5
   IOU Threshold: 0.5
   Camera: Index 0
   Display: Enabled
==================================================
📷 Opening camera 0...
✅ Camera ready: (480, 640, 3)
📐 Resolution: 640x480 @ 30 FPS
✅ Model loaded successfully
✅ MQTT connected: localhost:1883
📡 Publishing to topic: tapway/raw_event/metadata/cam1
🎬 Starting inference... Press 'q' to quit
==================================================
```

### OpenCV Window:
- Live camera feed with bounding boxes
- Red detection zone box in center
- Green text showing FPS and inference time
- Object count and tracker IDs

### Web Dashboard (if using `--web`):
- Access: http://localhost:5000
- Live video feed
- Detection event table
- Real-time object counts
- Hourly statistics chart

---

## 🔄 Monitor MQTT Events

Open another terminal:

```bash
source venv/bin/activate
python -m tools.mqtt_subscriber
```

You'll see events like:
```
✅ Connected to MQTT broker at localhost:1883
📡 Subscribing to topic: tapway/raw_event/metadata/+
🎧 Listening for zone events...
--------------------------------------------------
📨 [RECEIVED] 10:30:45
📡 Topic: tapway/raw_event/metadata/cam1
📄 Payload:
{
  "timestamp": "2025-11-13T10:30:45.123Z",
  "device_id": "cam1",
  "inout": {
    "zone_1": {
      "class_name": ["person"],
      "confidence": [0.92]
    }
  }
}
```

---

## ⚙️ Common Commands

### List All Configured Cameras
```bash
python -m src.main --list-cameras
```

### Test Specific Camera by Index
```bash
python -m src.main --camera 0
```

### Run with Different Web Dashboard Port
```bash
python -m src.main --camera-id cam1 --web --web-port 8080
```

### Stop Everything
- **Detection**: Press `q` in OpenCV window or `Ctrl+C` in terminal
- **MQTT Broker**: `cd mqtt-broker && docker compose down`

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `Camera not found` | Run `python -m tools.find_cameras` and update `path` in config |
| `Model not found` | Ensure `best.pt` exists and run `python scripts/export.py` |
| `MQTT connection failed` | Check Docker: `docker ps` or skip MQTT for now |
| `Import error` | Activate venv: `source venv/bin/activate` |
| `Low FPS` | Use smaller resolution or add `--no-display` |

---

## 📚 Next Steps

- Read the full [README.md](README.md) for advanced configuration
- Configure multiple cameras in `cameras_config.json`
- Customize detection zones in `src/config.py`
- Set up remote MQTT broker for network deployment

---

**Need help?** Open an issue on [GitHub](https://github.com/AdibSuid/yolo-zone-detection/issues)

**Ready to deploy?** Check the [README.md](README.md) for production configuration.
