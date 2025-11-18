# self-checkout-inference-intel

**Real-time object detection with zone-based tracking, optimized for Intel CPUs using OpenVINO and MQTT event publishing.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.6-purple)](https://github.com/openvinotoolkit/openvino)

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 🚀 **YOLOv8 + OpenVINO** | Hardware-accelerated inference for Intel CPUs/GPUs |
| 📦 **Zone Detection** | Configurable detection zones with entry/exit tracking |
| 📡 **MQTT Integration** | Real-time event publishing in Tapway format |
| 🌐 **Web Dashboard** | Live video feed with detection visualization |
| 🎯 **Object Tracking** | ByteTrack algorithm for persistent object IDs |
| 📷 **Multi-Camera Support** | USB, RTSP, HTTP, and video file sources |
| 🔧 **Easy Configuration** | JSON-based camera and MQTT settings |

---

## 📋 Prerequisites

- **Python 3.10+** (tested on 3.10.11)
- **Docker Desktop** (for MQTT broker) - [Download](https://www.docker.com/products/docker-desktop/)
- **USB Camera** or RTSP/HTTP stream
- **YOLOv8 Model** (`.pt` file for export to OpenVINO)

---

## 🚀 Deployment Guide

### 1. Clone Repository

```bash
git clone https://github.com/AdibSuid/yolo-zone-detection.git
cd yolo-zone-detection
```

### 2. Setup Python Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Export YOLOv8 Model to OpenVINO

Place your trained `best.pt` model in the project root, then:

```bash
python scripts/export.py
```

This will create `best_openvino_model/` directory with optimized model files.

### 4. Start Docker Desktop

Run the docker desktop app, make sure it's up & running before run the system.

### 5. Start MQTT Broker (Optional)

```bash
cd mqtt-broker
docker compose up -d
cd ..
```

### 6. Run Detection System

```bash
# Using camera ID from config
python -m src.main --camera-id cam1

# With web dashboard (access at http://localhost:5000)
python -m src.main --camera-id cam1 --web

# Using camera index directly
python -m src.main --camera 0
```

---

## 📖 Usage Guide

### Find Available Cameras

```bash
python -m tools.find_cameras
```

This will scan for all available cameras and show their specifications.

### Monitor MQTT Events

In a separate terminal:

```bash
source venv/bin/activate
python -m tools.mqtt_subscriber
```

**Available Options:**

| Option | Description | Example |
|--------|-------------|---------|
| `--camera-id` | Camera ID from config file | `--camera-id cam1` |
| `--camera` | Camera index (0, 1, 2...) | `--camera 0` |
| `--web` | Enable web dashboard | `--web` |
| `--web-port` | Web dashboard port | `--web-port 5000` |
| `--no-display` | Disable OpenCV window | `--no-display` |
| `--list-cameras` | List all configured cameras | `--list-cameras` |

---

## 🏗️ Project Structure

```
yolo-zone-detection/
├── src/                          # Core application code
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Main application entry point
│   ├── config.py                # Configuration management
│   ├── camera.py                # Camera capture and management
│   ├── detector.py              # YOLO detection engine
│   ├── mqtt_client.py           # MQTT event publishing
│   ├── performance.py           # FPS monitoring
│   ├── web_dashboard.py         # Web interface (optional)
│   └── templates/
│       └── dashboard.html       # Dashboard UI template
│
├── tools/                        # Utility scripts
│   ├── find_cameras.py          # Camera discovery tool
│   └── mqtt_subscriber.py       # MQTT event monitor
│
├── scripts/                      # Setup and maintenance
│   ├── export.py                # Export model to OpenVINO
│   └── setup.py                 # Automated setup script
│
├── mqtt-broker/                  # MQTT infrastructure
│   ├── docker-compose.yml       # Docker configuration
│   └── mosquitto/               # Mosquitto MQTT broker
│
├── best_openvino_model/         # Exported OpenVINO model (generated)
├── cameras_config.json          # Your camera configuration (gitignored)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ⚙️ Configuration

### Detection Zone Configuration

Edit `src/config.py` to customize the detection zone:

```python
class ZoneConfig:
    BOX_WIDTH_RATIO = 0.3   # 30% of frame width
    BOX_HEIGHT_RATIO = 0.4  # 40% of frame height
```

### MQTT Event Format

Events are published in this format:

```json
{
  "timestamp": "2025-11-13T10:30:45.123Z",
  "device_id": "cam1",
  "camResolution": [640, 480, 3],
  "siteID": "TAPWAY",
  "subgroupID": "Live Cam",
  "uniqueEventID": "A1B2C3D4E5",
  "inout": {
    "zone_1": {
      "class_name": ["person", "car"],
      "confidence": [0.92, 0.87]
    }
  }
}
```

---

---

## 🐛 Troubleshooting

### Common Issues

#### **"Camera not found" or "Failed to open camera"**

```bash
# Find available cameras
python -m tools.find_cameras

# Make sure no other application is using the camera
# Close apps like Zoom, Skype, Photo Booth, etc.
```

#### **"Model not found" or "Failed to load model"**

```bash
# Ensure you have exported the model
python scripts/export.py

# Check that best_openvino_model/ contains .xml and .bin files
ls -la best_openvino_model/
```

#### **MQTT Connection Failed**

```bash
# Check Docker is running
docker ps

# Restart MQTT broker
cd mqtt-broker
docker compose restart

# Check broker logs
docker compose logs mosquitto
```

#### **"Docker command not found"**

Install Docker Desktop:
- **macOS**: https://docs.docker.com/desktop/install/mac-install/
- **Windows**: https://docs.docker.com/desktop/install/windows-install/
- **Linux**: https://docs.docker.com/engine/install/

#### **Python Virtual Environment Activation Issues (Windows PowerShell)**

```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Then activate venv
.\venv\Scripts\activate
```

#### **Low FPS / Performance Issues**

1. Reduce camera resolution in `cameras_config.json`
2. Use `--no-display` flag to disable OpenCV window
3. Don't use `--web` flag (web dashboard reduces FPS by ~20-30%)
4. Close unnecessary applications

#### **Web Dashboard Not Accessible**

```bash
# Check if web dependencies are installed
pip install flask flask-cors flask-socketio python-socketio

# Access dashboard at http://localhost:5000
# Or use your computer's IP: http://192.168.x.x:5000
```

---

## 🔧 Advanced Configuration

### Custom YOLO Model

To use your own trained YOLOv8 model:

1. Place your `your_model.pt` file in the project root
2. Update `scripts/export.py`:
   ```python
   model = YOLO("your_model.pt")
   ```
3. Export to OpenVINO:
   ```bash
   python scripts/export.py
   ```
4. Update `src/config.py`:
   ```python
   CUSTOM = {
       "model": "your_model_openvino_model/",
       # ... other settings
   }
   ```

### Multiple Cameras Simultaneously

Edit `cameras_config.json` to enable multiple cameras, then run separate instances:

```bash
# Terminal 1
python -m src.main --camera-id cam1

# Terminal 2  
python -m src.main --camera-id cam2

# Terminal 3 - Monitor all events
python -m tools.mqtt_subscriber
```

### Remote MQTT Broker

Update `cameras_config.json` to use remote broker:

```json
{
  "mqtt": {
    "broker": "192.168.1.100",
    "port": 1883,
    "username": "your-username",
    "password": "your-password"
  }
}
```

---

## 📊 Performance Benchmarks

Tested on **Intel Core i5-10400** with built-in webcam:

| Configuration | FPS | CPU Usage | Notes |
|--------------|-----|-----------|-------|
| 640x480, No Display | ~25 FPS | 40% | Best performance |
| 640x480, OpenCV Display | ~20 FPS | 45% | Standard usage |
| 640x480, Web Dashboard | ~15 FPS | 55% | With visualization |
| 1920x1080, No Display | ~12 FPS | 65% | High resolution |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** - YOLO object detection framework
- **[OpenVINO](https://github.com/openvinotoolkit/openvino)** - Intel's deep learning inference toolkit
- **[Supervision](https://github.com/roboflow/supervision)** - Computer vision utilities
- **[Paho MQTT](https://github.com/eclipse/paho.mqtt.python)** - MQTT client library
- **[Eclipse Mosquitto](https://mosquitto.org/)** - MQTT broker

---

## � Support

For issues, questions, or feature requests, please:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [Issues](https://github.com/AdibSuid/yolo-zone-detection/issues)
3. Create a new issue with detailed information

---

**Made with ❤️ for real-time object detection and zone monitoring**
