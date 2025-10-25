# Changelog

All notable changes to the YOLO Zone Detection system.

## [1.0.0] - 2024-01-15

### 🎉 Initial Release - Professional Repository Structure

Complete restructuring of the codebase into a professional, modular Git repository.

### Added

#### Core Application (`src/`)
- **Modular architecture** with separated concerns:
  - `main.py` - Application entry point and main loop
  - `config.py` - Configuration classes for all settings
  - `camera.py` - Camera management and initialization
  - `detector.py` - YOLO detection and tracking logic
  - `mqtt_client.py` - MQTT event publishing
  - `performance.py` - FPS monitoring and performance tracking

#### Scripts (`scripts/`)
- `setup.py` - Automated installation and setup script
- `export_models.py` - YOLO model export to OpenVINO format

#### Tools (`tools/`)
- `find_cameras.py` - Camera detection and testing utility
- `mqtt_subscriber.py` - MQTT event monitoring tool

#### Documentation (`docs/`)
- `INSTALLATION.md` - Complete installation guide
- `USAGE.md` - Comprehensive usage documentation

#### Infrastructure
- `requirements.txt` - Pinned Python dependencies
- `.gitignore` - Comprehensive ignore rules
- `README.md` - Professional project documentation
- `CHANGELOG.md` - This file

### Changed
- Restructured entire codebase from single-file to modular design
- Separated configuration from application logic
- Improved error handling throughout
- Enhanced MQTT event format with more metadata

### Features

#### Performance Modes
- **Ultra Fast**: YOLOv8n @ 320×320, maximum FPS
- **Maximum FPS**: YOLOv8n @ 416×416, high speed
- **Balanced**: YOLOv8s @ 640×480, recommended
- **High Accuracy**: YOLOv8s @ 640×480, best quality

#### Detection Capabilities
- Zone-based detection (configurable box/polygon)
- ByteTrack object tracking for persistent IDs
- Real-time FPS monitoring
- MQTT event publishing for detected objects
- 80 COCO object classes support

#### Camera Support
- USB camera support with DirectShow backend
- Auto-exposure optimization (15 FPS)
- Brightness/contrast enhancement
- Multiple camera support

### Technical Details

#### Dependencies
- `ultralytics==8.3.221` - YOLOv8 framework
- `openvino==2024.6.0` - Intel CPU optimization
- `opencv-python==4.11.0.86` - Computer vision
- `supervision==0.26.1` - Detection utilities
- `paho-mqtt==2.1.0` - MQTT client
- `numpy` - Numerical operations
- `tqdm` - Progress bars

#### Performance Metrics
- **FPS**: 10-30 depending on mode and hardware
- **Latency**: <100ms inference on Intel CPUs
- **Accuracy**: COCO mAP 40+ (YOLOv8s)

---

## Pre-Release Development

### Key Milestones

#### Detection Stability Fix
- **Issue**: Random crashes after few seconds
- **Root Cause**: Label/detection count mismatch in ByteTrack tracker
- **Solution**: Added detection caching and count validation
- **Result**: Stable operation for 450+ frames without issues

#### Camera Optimization
- **Investigation**: FPS limited to 15 instead of expected 30
- **Finding**: Hardware limitation with auto-exposure mode
- **Testing**: Manual exposure achieved 30 FPS but 90% brightness loss
- **Decision**: Keep auto-exposure at 15 FPS for usable image quality
- **Enhancement**: Added brightness/contrast boost (both set to 128)

#### Zone Enhancement
- **Change**: Switched from LineZone to PolygonZone
- **Configuration**: 30% width × 40% height centered box
- **Benefit**: More intuitive object containment detection

#### Display Improvement
- **Scaling**: Increased from 2× to 2.5× for better visibility
- **Overlays**: Added FPS, inference time, object count
- **Zone visualization**: Red box outline with annotations

#### MQTT Integration
- **Event format**: Standardized JSON with metadata
- **Topic**: `cv/zone_events`
- **Data**: tracker_id, class_name, confidence, timestamp, FPS, mode
- **Reliability**: Error handling for broker disconnections

---

## Future Roadmap

### Planned Features
- [ ] Web dashboard for monitoring
- [ ] Multi-zone support
- [ ] Recording and playback
- [ ] Alert system (email/webhook)
- [ ] Model auto-tuning
- [ ] GPU acceleration support
- [ ] REST API

### Performance Improvements
- [ ] INT8 quantization for faster inference
- [ ] Dynamic batching
- [ ] Frame prediction/interpolation
- [ ] Background subtraction optimization

### Documentation
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] Video tutorials
- [ ] Example integrations

---

**Note**: This project follows [Semantic Versioning](https://semver.org/).
