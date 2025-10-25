# YOLO Zone Detection System v1.0.0

**Professional Git Repository - Ready for Distribution**

## 📋 Quick Overview

A professional, modular Git repository for real-time object detection using YOLOv8 + OpenVINO, featuring zone-based detection and MQTT event publishing.

## ✅ What's Included

### Core Application (`src/`)
- **main.py** - Entry point with ZoneDetectionApp class
- **config.py** - Performance modes and all settings
- **camera.py** - Camera management with auto-optimization
- **detector.py** - YOLO inference and ByteTrack tracking
- **mqtt_client.py** - MQTT event publisher
- **performance.py** - FPS monitoring

### Utilities
- **scripts/setup.py** - Automated installation
- **scripts/export_models.py** - Model export to OpenVINO
- **tools/find_cameras.py** - Camera detection
- **tools/mqtt_subscriber.py** - Event monitoring

### Documentation
- **README.md** - Main documentation with Quick Start
- **CHANGELOG.md** - Version history
- **docs/INSTALLATION.md** - Complete setup guide
- **docs/USAGE.md** - Usage examples
- **docs/MIGRATION.md** - Migration from old structure
- **docs/ARCHITECTURE.md** - System architecture

### Infrastructure
- **requirements.txt** - Python dependencies (pinned versions)
- **.gitignore** - Git ignore rules
- **run.bat / run.sh** - Quick start scripts
- **mqtt-broker/** - Docker Compose for MQTT

## 📁 Repository Structure

```
yolo_openvino_mqtt/
├── src/                    # Core application (modular)
│   ├── main.py
│   ├── config.py
│   ├── camera.py
│   ├── detector.py
│   ├── mqtt_client.py
│   └── performance.py
├── scripts/               # Setup utilities
│   ├── setup.py
│   └── export_models.py
├── tools/                 # Developer tools
│   ├── find_cameras.py
│   └── mqtt_subscriber.py
├── docs/                  # Documentation
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   ├── MIGRATION.md
│   └── ARCHITECTURE.md
├── mqtt-broker/           # MQTT infrastructure
│   └── docker-compose.yml
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # Main documentation
├── CHANGELOG.md          # Version history
├── run.bat               # Windows launcher
└── run.sh                # Linux/Mac launcher
```

**Note**: Model files (*.pt, *_openvino_model/) are git-ignored and will be downloaded/generated on first setup.

## ✨ Features

**Performance Modes**
- Ultra Fast (YOLOv8n @ 320×320) - Maximum speed
- Maximum FPS (YOLOv8n @ 416×416) - High speed
- Balanced (YOLOv8s @ 640×480) - **Recommended**
- High Accuracy (YOLOv8s @ 640×480) - Best quality

**Detection**
- Zone-based detection (configurable box)
- ByteTrack object tracking
- 80 COCO object classes
- Real-time FPS monitoring

**Camera**
- USB/built-in camera support
- Auto-exposure optimization
- DirectShow backend (Windows)
- 2.5× display scaling

**MQTT**
- Event publishing to `cv/zone_events`
- JSON format with metadata
- Mosquitto broker (Docker)

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd yolo_openvino_mqtt
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Export models
python -m scripts.export_models

# Find camera
python -m tools.find_cameras

# Run detection
python -m src.main --camera 1 --mode balanced
```

**Or use quick start script:**
```bash
run.bat          # Windows
./run.sh         # Linux/Mac
```

## 🔧 Customization

**Add Performance Mode** (`src/config.py`):
```python
MY_MODE = {
    "name": "Custom",
    "model": "yolov8n_openvino_model/",
    "resolution": (512, 512),
    "conf_threshold": 0.4
}
```

**Change Zone Size** (`src/config.py`):
```python
class ZoneConfig:
    BOX_WIDTH_RATIO = 0.5
    BOX_HEIGHT_RATIO = 0.6
```

**Use as Library**:
```python
from src import ZoneDetectionApp

app = ZoneDetectionApp(mode="balanced", camera_index=1)
if app.initialize():
    app.run()
```

## 📊 Technical Details

**Dependencies**
- ultralytics 8.3.221 (YOLOv8)
- openvino 2024.6.0 (Intel CPU optimization)
- opencv-python 4.11.0.86 (Computer vision)
- supervision 0.26.1 (Detection utilities)
- paho-mqtt 2.1.0 (MQTT client)

**Code Quality**
- Modular architecture (7 focused modules)
- Comprehensive docstrings
- Error handling throughout
- Clean separation of concerns

## 📝 Documentation

- **README.md** - Quick start and overview
- **docs/INSTALLATION.md** - Detailed setup guide
- **docs/USAGE.md** - Usage examples and tips
- **docs/ARCHITECTURE.md** - System design
- **docs/MIGRATION.md** - Upgrade from old version
- **CHANGELOG.md** - Version history

## ✅ Status

**Repository is ready for distribution!**

- ✅ Professional structure
- ✅ Comprehensive documentation
- ✅ Easy installation
- ✅ All features working
- ✅ Clean codebase
- ✅ Git-ready (.gitignore configured)

**Next steps:**
1. Initialize git: `git init`
2. Add files: `git add .`
3. Commit: `git commit -m "Initial commit - v1.0.0"`
4. Push to GitHub

---

**Ready to clone, install, and run!** 🎉
