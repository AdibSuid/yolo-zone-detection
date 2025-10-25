# Installation Guide

This guide walks through the complete installation process for the YOLO Zone Detection system.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Docker for MQTT broker
- USB camera or built-in webcam

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd yolo_openvino_mqtt
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `ultralytics==8.3.221` - YOLOv8 framework
- `openvino==2024.6.0` - Intel OpenVINO for CPU optimization
- `opencv-python==4.11.0.86` - Computer vision library
- `supervision==0.26.1` - Detection utilities (ByteTrack, zones, annotators)
- `paho-mqtt==2.1.0` - MQTT client
- `numpy` - Numerical operations
- `tqdm` - Progress bars

### 4. Export YOLO Models

**Option A: Automated Setup**
```bash
python -m scripts.setup
```

**Option B: Manual Export**
```bash
python -m scripts.export_models
```

This will:
- Download YOLOv8n and YOLOv8s models
- Export them to OpenVINO IR format
- Create `yolov8n_openvino_model/` and `yolov8s_openvino_model/`
- Validate exported models

### 5. Setup MQTT Broker (Optional)

**Using Docker (Recommended):**
```bash
cd mqtt-broker
docker-compose up -d
cd ..
```

**Verify MQTT Broker:**
```bash
# Check container is running
docker ps

# You should see mosquitto:2.0 running on port 1883
```

### 6. Find Your Camera

```bash
python -m tools.find_cameras
```

This will:
- Scan for available cameras (indices 0-9)
- Display resolution and FPS for each
- Show test preview for each camera

**Common Camera Indices:**
- `0` - Built-in laptop webcam
- `1` - First USB camera
- `2` - Second USB camera

### 7. Test the System

```bash
# Run with default settings (camera 0, balanced mode)
python -m src.main

# Run with USB camera
python -m src.main --camera 1 --mode balanced
```

## Verification

### Check Installation

```python
# Test imports
python -c "import ultralytics, openvino, cv2, supervision, paho.mqtt.client; print('✅ All imports successful')"
```

### Check Models

```bash
# Should exist after export
ls yolov8n_openvino_model/
ls yolov8s_openvino_model/
```

Expected files:
- `metadata.yaml`
- `yolov8*.xml` (model structure)
- `yolov8*.bin` (model weights)

### Check MQTT

```bash
# Terminal 1: Start subscriber
python -m tools.mqtt_subscriber

# Terminal 2: Run detection
python -m src.main --camera 1
```

You should see events in the subscriber when objects enter the zone.

## Troubleshooting

### Import Errors

```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Model Export Fails

```bash
# Check Python version (must be 3.8+)
python --version

# Try exporting manually
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='openvino')"
```

### Camera Not Found

```bash
# Detailed camera search
python -m tools.find_cameras

# Try with DirectShow backend (Windows)
# Already configured in src/camera.py
```

### MQTT Broker Won't Start

```bash
# Check Docker is running
docker --version

# Check ports are not in use
netstat -an | findstr "1883"  # Windows
netstat -an | grep "1883"     # Linux/Mac

# Restart Docker Desktop and try again
```

### Low FPS

- Start with `ultra_fast` mode: `python -m src.main --mode ultra_fast --camera 1`
- Check camera supports your resolution
- Close other applications using the camera
- Consider reducing detection zone size

## Next Steps

After successful installation:

1. **Customize Configuration**: Edit `src/config.py` for your needs
2. **Adjust Zone Size**: Modify `ZoneConfig.BOX_WIDTH_RATIO` and `BOX_HEIGHT_RATIO`
3. **Performance Tuning**: Try different modes with `--list-modes`
4. **Integration**: Use MQTT events in your own applications

## Platform-Specific Notes

### Windows
- Use PowerShell or CMD (not Git Bash for venv activation)
- DirectShow backend is automatically configured
- Path separators: Use `\` or `/` (Python handles both)

### Linux
- May need to install system dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip
  ```
- Camera permissions: Add user to `video` group
  ```bash
  sudo usermod -a -G video $USER
  ```

### macOS
- Install system dependencies via Homebrew:
  ```bash
  brew install python@3.10
  ```
- Grant camera permissions in System Preferences

## Updating

```bash
# Pull latest changes
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Re-export models if needed
python -m scripts.export_models
```

---

**Installation complete! 🎉** Ready to detect objects with `python -m src.main`
