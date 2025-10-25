#!/bin/bash
# Quick start script for YOLO Zone Detection System
# Linux/Mac bash script

set -e

echo ""
echo "================================================================"
echo "  YOLO Zone Detection System - Quick Start"
echo "================================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "[SUCCESS] Virtual environment created"
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
echo "[INFO] Checking dependencies..."
if ! python -c "import ultralytics, openvino, cv2, supervision, paho.mqtt.client" 2>/dev/null; then
    echo "[INFO] Installing requirements..."
    pip install -r requirements.txt
    echo "[SUCCESS] Requirements installed"
else
    echo "[SUCCESS] Dependencies already installed"
fi

# Check if models are exported
if [ ! -d "yolov8n_openvino_model" ]; then
    echo "[INFO] Exporting YOLO models..."
    python -m scripts.export_models
    echo "[SUCCESS] Models exported"
else
    echo "[SUCCESS] Models already exported"
fi

echo ""
echo "================================================================"
echo "  Setup Complete!"
echo "================================================================"
echo ""
echo "Choose an option:"
echo "  1. Find available cameras"
echo "  2. Run detection (default camera, balanced mode)"
echo "  3. Run detection (USB camera 1, ultra fast mode)"
echo "  4. Monitor MQTT events"
echo "  5. Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "[INFO] Finding cameras..."
        python -m tools.find_cameras
        ;;
    2)
        echo ""
        echo "[INFO] Starting detection with default settings..."
        python -m src.main
        ;;
    3)
        echo ""
        echo "[INFO] Starting detection with USB camera 1, ultra fast mode..."
        python -m src.main --camera 1 --mode ultra_fast
        ;;
    4)
        echo ""
        echo "[INFO] Starting MQTT subscriber..."
        python -m tools.mqtt_subscriber
        ;;
    5)
        echo ""
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo ""
        echo "[ERROR] Invalid choice"
        ;;
esac

echo ""
