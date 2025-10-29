# YOLO Zone Detection with OpenVINO & MQTT

Real-time object detection system using YOLOv8 optimized with OpenVINO for Intel CPUs, featuring zone-based detection and MQTT event publishing.

## ✨ Features

- 🚀 **YOLOv8 with OpenVINO**: Optimized inference for Intel CPUs
- 📦 **Zone Detection**: Configurable detection zones (box/polygon)
- 📡 **MQTT Integration**: Real-time event publishing
- � **Web Dashboard**: Real-time browser-based monitoring with live video feed
- �🎯 **Object Tracking**: ByteTrack for persistent object IDs
- ⚡ **Performance Modes**: 4 pre-configured modes (ultra_fast to high_accuracy)
- 📷 **Camera Support**: USB cameras with auto-exposure optimization
- 🔧 **Modular Design**: Clean, organized codebase

## Pre-requisite
- Docker Desktop installed & running/opened
- Python3.10 (I'm using 3.10.11)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd yolo_openvino_mqtt

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Export Models (First Time Only)

```bash
# Export YOLOv8 models to OpenVINO format
python .\scripts\export.py
```

Or run the automated setup script:

```bash
python -m scripts.setup
```

### 3. Find Your Camera

```bash
# Detect available cameras
python -m tools.find_cameras
```

### 4. Start MQTT Broker (Optional)

```bash
# Using Docker
cd mqtt-broker
docker-compose up -d
```

### 5. Run Detection

```bash
# Basic usage with default settings
python -m src.main --camera 0

# Use USB camera with web dashboard
python -m src.main --camera 0 --web

```

### 6. Access Web Dashboard

Once the application starts, the web dashboard will be available at:
- **Local**: http://localhost:5000
- **Network**: http://[your-ip]:5000 (shown in console output)

The dashboard displays:
- 📹 **Live Video Feed**: Real-time camera view with detection overlays
- 📊 **Total Object Count**: Cumulative count of detected objects
- 📸 **Recent Detections**: List of recent objects with cropped images
- ℹ️ **System Info**: Camera ID, name, model info, timestamps
- 🎯 **Detection Details**: Confidence scores, IN/OUT direction, event times

### 7. Monitor MQTT Events (Optional)

In a separate terminal:

```bash
python -m tools.mqtt_subscriber
```

## 🛠️ Configuration

### Camera Settings

The system auto-configures cameras for optimal performance:
- **Auto-exposure**: Enabled (15 FPS on most USB cameras)
- **Brightness**: 128 (boosted for better image quality)
- **Contrast**: 128
- **Buffer size**: 1 (minimize latency)

### Detection Zone

Default zone is a centered box (30% width × 40% height). Customize in `src/config.py`:

```python
class ZoneConfig:
    BOX_WIDTH_RATIO = 0.3   # 30% of frame width
    BOX_HEIGHT_RATIO = 0.4  # 40% of frame height
```

### MQTT Events

Published to topic `cv/zone_events` with format:

```json
{
  "event": "inside_zone",
  "tracker_id": 123,
  "class_id": 0,
  "class_name": "person",
  "confidence": 0.87,
  "timestamp": 1234567890.123,
  "fps": 15.2,
  "mode": "balanced"
}
```

## 🧪 Development

### Running Tests

```bash
# Find available cameras
python -m tools.find_cameras

# Test MQTT connection
python -m tools.mqtt_subscriber

```

### Code Organization

- **src/**: Core application code (modular, reusable)
- **scripts/**: Setup and export utilities
- **tools/**: Developer tools and utilities
- **mqtt-broker/**: MQTT infrastructure

## 📝 Detectable Objects

The system can detect 80 COCO classes including:
- person, bicycle, car, motorcycle, bus, truck
- cat, dog, bird, horse, cow, elephant
- backpack, umbrella, handbag, suitcase
- bottle, cup, fork, knife, spoon, bowl
- laptop, mouse, keyboard, cell phone
- And 60+ more...

Run `python -m src.main --list-modes` for full details.

## 🐛 Troubleshooting

### Creating Venv Issue

```bash
PS C:\Users\Kaizo\Documents\yolo-zone-detection> .\venv\Scripts\activate
.\venv\Scripts\activate : File C:\Users\Kaizo\Documents\yolo-zone-detection\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system. For more information, see about_Execution_Policies at
https:/go.microsoft.com/fwlink/?LinkID=135170.
At line:1 char:1
+ .\venv\Scripts\activate
+ ~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : SecurityError: (:) [], PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
```

### Camera Issues

```bash
# Find available cameras
python -m tools.find_cameras

# Check camera is not in use by another app
# Close other camera applications and try again
```
Open PowerShell as Administrator.

Run:
```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Confirm with Y when prompted.

Try activating again:

```bash
.\venv\Scripts\activate
```

### MQTT Connection Failed

```bash
# Check Docker is running
docker ps

# Restart MQTT broker
cd mqtt-broker
docker-compose restart
```

### Model Not Found

```bash
# Export models to OpenVINO format
python -m scripts.export_custom_models
```

## 📄 License

MIT License - feel free to use and modify!

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino)
- [Supervision](https://github.com/roboflow/supervision)
- [Paho MQTT](https://github.com/eclipse/paho.mqtt.python)

## 📬 Support

For issues and questions, please open a GitHub issue.

---

**Made with ❤️ for real-time object detection**
