@echo off
REM Quick start script for YOLO Zone Detection System
REM Windows batch file

echo.
echo ================================================================
echo   YOLO Zone Detection System - Quick Start
echo ================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [INFO] Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if requirements are installed
echo [INFO] Checking dependencies...
python -c "import ultralytics, openvino, cv2, supervision, paho.mqtt.client" 2>nul
if errorlevel 1 (
    echo [INFO] Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements
        pause
        exit /b 1
    )
    echo [SUCCESS] Requirements installed
) else (
    echo [SUCCESS] Dependencies already installed
)

REM Check if models are exported
if not exist "yolov8n_openvino_model\" (
    echo [INFO] Exporting YOLO models...
    python -m scripts.export_models
    if errorlevel 1 (
        echo [ERROR] Failed to export models
        pause
        exit /b 1
    )
    echo [SUCCESS] Models exported
) else (
    echo [SUCCESS] Models already exported
)

echo.
echo ================================================================
echo   Setup Complete!
echo ================================================================
echo.
echo Choose an option:
echo   1. Find available cameras
echo   2. Run detection (default camera, balanced mode)
echo   3. Run detection (USB camera 1, ultra fast mode)
echo   4. Monitor MQTT events
echo   5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo [INFO] Finding cameras...
    python -m tools.find_cameras
) else if "%choice%"=="2" (
    echo.
    echo [INFO] Starting detection with default settings...
    python -m src.main
) else if "%choice%"=="3" (
    echo.
    echo [INFO] Starting detection with USB camera 1, ultra fast mode...
    python -m src.main --camera 1 --mode ultra_fast
) else if "%choice%"=="4" (
    echo.
    echo [INFO] Starting MQTT subscriber...
    python -m tools.mqtt_subscriber
) else if "%choice%"=="5" (
    echo.
    echo Goodbye!
    exit /b 0
) else (
    echo.
    echo [ERROR] Invalid choice
)

echo.
pause
