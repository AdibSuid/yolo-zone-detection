@echo off

REM Start Docker Desktop
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

REM Wait for Docker to start (optional, adjust as needed)
:waitDocker
docker info >nul 2>&1
if errorlevel 1 (
    echo Waiting for Docker to start...
    timeout /t 2 >nul
    goto waitDocker
)

REM Activate virtual environment
call C:\Users\admin\Documents\yolo-zone-detection\venv\Scripts\activate.bat

REM Run the main Python script
python -m src.main --camera-id cam1 --web

pause