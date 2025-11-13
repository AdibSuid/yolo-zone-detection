"""
YOLO Zone Detection System - Automated Setup Script

This script automates the initial setup process:
1. Checks Python version compatibility (3.8+)
2. Verifies virtual environment usage
3. Installs required Python packages
4. Guides model export if needed
5. Optionally starts MQTT broker with Docker

Usage:
    python scripts/setup.py
    
    Or run directly:
    python -m scripts.setup
"""
import os
import sys
import subprocess


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True


def check_venv():
    """Check if running in virtual environment."""
    print_header("Checking Virtual Environment")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        return True
    else:
        print("⚠️  Not running in virtual environment")
        print("\n💡 Recommended: Create and activate a virtual environment:")
        print("   python -m venv venv")
        print("   .\\venv\\Scripts\\activate  # Windows")
        print("   source venv/bin/activate  # Linux/Mac")
        return False


def install_requirements():
    """Install required packages."""
    print_header("Installing Requirements")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    try:
        print("Installing packages from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def export_models():
    """Export YOLO models to OpenVINO format."""
    print_header("Exporting YOLO Models")
    
    # Check if models already exist
    models_exist = os.path.exists("yolov8n_openvino_model") and os.path.exists("yolov8s_openvino_model")
    
    if models_exist:
        print("✅ Models already exported")
        return True
    
    try:
        print("Exporting models to OpenVINO format...")
        subprocess.check_call([sys.executable, "-m", "scripts.export_models"])
        print("✅ Models exported successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to export models: {e}")
        return False


def start_mqtt_broker():
    """Start MQTT broker using Docker."""
    print_header("Starting MQTT Broker")
    
    # Check if Docker is available
    try:
        subprocess.check_output(["docker", "--version"], stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Docker not found or not running")
        print("\n💡 To use MQTT features:")
        print("   1. Install Docker Desktop")
        print("   2. Run: cd mqtt-broker && docker-compose up -d")
        return False
    
    # Check if docker-compose file exists
    if not os.path.exists("mqtt-broker/docker-compose.yml"):
        print("⚠️  mqtt-broker/docker-compose.yml not found")
        return False
    
    try:
        print("Starting MQTT broker with Docker Compose...")
        subprocess.check_call(["docker-compose", "up", "-d"], cwd="mqtt-broker")
        print("✅ MQTT broker started successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to start MQTT broker: {e}")
        print("\n💡 You can start it manually:")
        print("   cd mqtt-broker")
        print("   docker-compose up -d")
        return False


def main():
    """Main setup script."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║       YOLO Zone Detection System - Setup Script         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    in_venv = check_venv()
    if not in_venv:
        response = input("\nContinue without virtual environment? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed at requirements installation")
        sys.exit(1)
    
    # Export models
    if not export_models():
        print("\n⚠️  Model export failed, you may need to run it manually:")
        print("   python -m scripts.export_models")
    
    # Start MQTT broker
    start_mqtt_broker()
    
    # Final instructions
    print_header("Setup Complete! 🎉")
    print("""
Next steps:

1. Find your camera:
   python -m tools.find_cameras

2. Run the detection system:
   python -m src.main --camera 1 --mode balanced

3. Monitor MQTT events (in another terminal):
   python -m tools.mqtt_subscriber

4. For more options:
   python -m src.main --help
   python -m src.main --list-modes

Enjoy detecting! 🚀
    """)


if __name__ == "__main__":
    main()
