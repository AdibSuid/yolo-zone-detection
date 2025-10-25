# Migration Guide

Guide for transitioning from the old single-file structure to the new modular structure.

## What Changed

### Old Structure (Before v1.0.0)

```
yolo_openvino_mqtt/
├── infer_zone_mqtt_unified.py   # All code in one file
├── find_cameras.py
├── mqtt_subscriber.py
├── model_optimizer.py
├── benchmark.py
├── test_camera_fps.py
└── requirements.txt
```

### New Structure (v1.0.0+)

```
yolo_openvino_mqtt/
├── src/                         # Modular application code
│   ├── main.py
│   ├── config.py
│   ├── camera.py
│   ├── detector.py
│   ├── mqtt_client.py
│   └── performance.py
├── scripts/                     # Setup utilities
│   ├── setup.py
│   └── export_models.py
├── tools/                       # Developer tools
│   ├── find_cameras.py
│   └── mqtt_subscriber.py
├── docs/                        # Documentation
│   ├── INSTALLATION.md
│   └── USAGE.md
├── requirements.txt
├── .gitignore
├── README.md
├── CHANGELOG.md
└── run.bat / run.sh            # Quick start scripts
```

## Command Changes

### Old Commands

```bash
# Old way (still works for backward compatibility)
python infer_zone_mqtt_unified.py --camera 1 --mode balanced

# Old utility scripts
python find_cameras.py
python mqtt_subscriber.py
python model_optimizer.py
```

### New Commands

```bash
# New way (recommended)
python -m src.main --camera 1 --mode balanced

# New utility scripts
python -m tools.find_cameras
python -m tools.mqtt_subscriber
python -m scripts.export_models

# Or use quick start scripts
run.bat          # Windows
./run.sh         # Linux/Mac
```

## Code Migration

### If You Imported the Old Module

**Old:**
```python
# This won't work anymore
from infer_zone_mqtt_unified import PerformanceMode, main
```

**New:**
```python
# Import from new modular structure
from src.config import PerformanceMode
from src.main import ZoneDetectionApp
from src.detector import YOLODetector
from src.camera import CameraManager
```

### If You Modified the Configuration

**Old:**
```python
# Edit infer_zone_mqtt_unified.py directly
class PerformanceMode:
    BALANCED = {
        "resolution": (640, 480),  # Your change
        ...
    }
```

**New:**
```python
# Edit src/config.py
class PerformanceMode:
    BALANCED = {
        "resolution": (800, 600),  # Your change
        ...
    }
```

### If You Modified Camera Settings

**Old:**
```python
# Edit camera setup in main() function
cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
```

**New:**
```python
# Edit src/camera.py in CameraConfig class
class CameraConfig:
    BRIGHTNESS = 150  # Your change
    CONTRAST = 150
```

### If You Modified MQTT Logic

**Old:**
```python
# Edit main() function directly
payload = {
    "event": "inside_zone",
    ...
}
mqttc.publish(MQTT_TOPIC, json.dumps(payload))
```

**New:**
```python
# Edit src/mqtt_client.py
class MQTTPublisher:
    def publish_zone_event(self, ...):
        payload = {
            "event": "inside_zone",
            "custom_field": "your_value",  # Your addition
            ...
        }
```

## Feature Parity

All features from the old version are preserved:

✅ **All performance modes** (ultra_fast, maximum_fps, balanced, high_accuracy)  
✅ **Camera support** (USB, built-in, auto-exposure)  
✅ **Zone detection** (PolygonZone with configurable size)  
✅ **MQTT publishing** (same event format)  
✅ **Object tracking** (ByteTrack)  
✅ **FPS monitoring** (PerformanceMonitor)  
✅ **Display scaling** (2.5× window)  
✅ **Frame skipping** (for performance)  

## Backward Compatibility

### Old Script Still Available

The old `infer_zone_mqtt_unified.py` is preserved for reference but deprecated. It will be removed in v2.0.0.

```bash
# Still works (deprecated)
python infer_zone_mqtt_unified.py --camera 1 --mode balanced

# Recommended
python -m src.main --camera 1 --mode balanced
```

### Migration Timeline

- **v1.0.0** - New structure released, old script deprecated
- **v1.x.x** - Both old and new scripts work
- **v2.0.0** - Old script removed (future release)

## Benefits of New Structure

### For Users

✅ **Easier setup** - Automated setup script  
✅ **Better docs** - Comprehensive guides  
✅ **Quick start** - run.bat/run.sh scripts  
✅ **Clearer errors** - Better error messages  

### For Developers

✅ **Modular code** - Separated concerns  
✅ **Testable** - Each module can be tested independently  
✅ **Maintainable** - Easy to find and fix issues  
✅ **Extensible** - Add features without touching core code  
✅ **Documented** - Comprehensive inline docs  

### For Integrations

✅ **Import specific modules** - Don't import everything  
✅ **Custom configurations** - Easy to subclass  
✅ **Event handling** - Clean MQTT client interface  
✅ **Camera management** - Reusable camera utilities  

## Troubleshooting Migration

### "Module not found" error

```python
# Error: No module named 'infer_zone_mqtt_unified'
# Solution: Update imports
from src.config import PerformanceMode
from src.main import ZoneDetectionApp
```

### "Cannot find infer_zone_mqtt_unified.py"

```bash
# Error: python: can't open file 'infer_zone_mqtt_unified.py'
# Solution: Use new command
python -m src.main --camera 1 --mode balanced
```

### Old configuration not working

```bash
# Old config in infer_zone_mqtt_unified.py ignored
# Solution: Move changes to src/config.py
```

### Custom modifications lost

1. **Backup your old file**: `cp infer_zone_mqtt_unified.py infer_zone_mqtt_unified.py.backup`
2. **Identify your changes**: `diff infer_zone_mqtt_unified.py.backup infer_zone_mqtt_unified.py`
3. **Apply to new structure**: Move changes to appropriate files in `src/`

## Getting Help

### Check Documentation

- **Installation**: `docs/INSTALLATION.md`
- **Usage**: `docs/USAGE.md`
- **Changelog**: `CHANGELOG.md`

### Compare Old vs New

```bash
# See what changed
git diff v0.9.0 v1.0.0

# View old version
git show v0.9.0:infer_zone_mqtt_unified.py
```

### Ask for Help

Open a GitHub issue with:
- What you're trying to do
- Old code that worked
- New code that doesn't work
- Error messages

## Quick Reference

| Old | New | Notes |
|-----|-----|-------|
| `infer_zone_mqtt_unified.py` | `src/main.py` | Main entry point |
| `find_cameras.py` | `tools/find_cameras.py` | Camera detection |
| `mqtt_subscriber.py` | `tools/mqtt_subscriber.py` | Event monitoring |
| `model_optimizer.py` | `scripts/export_models.py` | Model export |
| Direct imports | `from src import ...` | Module imports |
| Edit main file | Edit `src/config.py` | Configuration |
| Single command | Module commands | Execution |

---

**Migration complete!** 🎉 Enjoy the new modular structure!

For questions, see [USAGE.md](USAGE.md) or open a GitHub issue.
