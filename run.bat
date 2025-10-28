@echo off
REM Quick start script for YOLO Zone Detection System

echo.
echo ================================================================
echo   YOLO Zone Detection - Intel CPU Optimized
echo ================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if requirements are installed
echo [INFO] Checking dependencies...
python -c "import ultralytics, openvino, cv2" 2>nul
if errorlevel 1 (
    echo [INFO] Installing requirements...
    pip install -r requirements.txt
)

REM Check if model is exported
if not exist "custom_yolov8_openvino_model\" (
    echo [INFO] Model not found. Please export your model first:
    echo    python scripts/export_custom_model.py
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   Ready to Run!
echo ================================================================
echo.
echo Choose an option:
echo   1. Find available cameras
echo   2. Run detection (custom model, web dashboard OFF)
echo   3. Run detection (custom model, web dashboard ON)
echo   4. Monitor MQTT events
echo   5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    python -m tools.find_cameras
) else if "%choice%"=="2" (
    python -m src.main --mode custom --camera 1
) else if "%choice%"=="3" (
    python -m src.main --mode custom --camera 1 --web
) else if "%choice%"=="4" (
    python -m tools.mqtt_subscriber
) else if "%choice%"=="5" (
    exit /b 0
) else (
    echo [ERROR] Invalid choice
)

pause