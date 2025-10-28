# YOLO Zone Detection - Intel CPU Optimized

Real-time retail object detection using YOLOv8 with OpenVINO optimization for Intel CPUs.

## ✨ Features

- 🚀 **15-20 FPS** on Intel Core i5/i7 CPUs
- 📦 **Custom YOLOv8 Model** support
- 📡 **MQTT Integration** for real-time events
- 🌐 **Web Dashboard** (optional, disabled by default)
- 🎯 **Zone-based Detection** with object tracking
- ⚡ **OpenVINO Optimized** for Intel CPUs

## 🚀 Quick Start

### 1. Installation

\`\`\`bash
# Clone/extract repository
cd yolo-zone-detection

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 2. Export Your Model

\`\`\`bash
# Place your best.pt in root directory
# Export to OpenVINO format
python scripts/export_custom_model.py
\`\`\`

### 3. Find Camera

\`\`\`bash
python -m tools.find_cameras
\`\`\`

### 4. Run Detection

\`\`\`bash
# Basic usage (web dashboard OFF by default)
python -m src.main --mode custom --camera 1

# With web dashboard
python -m src.main --mode custom --camera 1 --web

# Maximum performance (no display)
python -m src.main --mode custom --camera 1 --no-display
\`\`\`

## 📊 Performance Modes

| Mode | Resolution | FPS | Use Case |
|------|------------|-----|----------|
| custom | 640x640 | 15-20 | Your YOLOv8 model |
| retail_optimized | 416x416 | 15-20 | Recommended |
| ultra_fast | 320x320 | 20-25 | Maximum speed |

## 🔧 Configuration

Edit \`src/config.py\` to adjust:
- Confidence threshold (default: 0.5)
- Detection zone size
- Camera settings
- Model resolution

## 📡 MQTT Events

Start MQTT broker:
\`\`\`bash
cd mqtt-broker
docker-compose up -d
\`\`\`

Monitor events:
\`\`\`bash
python -m tools.mqtt_subscriber
\`\`\`

## 🌐 Web Dashboard

Access at: http://localhost:5000 (when enabled with --web flag)

**Note:** Web dashboard reduces FPS by 20-30%. For maximum performance, keep it disabled.

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Low FPS | Use \`--no-display\` flag |
| Camera error | Run \`python -m tools.find_cameras\` |
| No detections | Lower confidence in config.py |
| MQTT error | Start broker: \`docker-compose up -d\` |

## 📄 License

MIT License


**Ready to deploy! 🚀**
