# Changelog

All notable changes to the YOLO Zone Detection System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-13

### Added
- 🚀 Initial release of YOLO Zone Detection System
- ✨ YOLOv8 object detection with OpenVINO optimization
- 📦 Zone-based detection with configurable geometry
- 📡 MQTT event publishing in Tapway format
- 🌐 Optional web dashboard for real-time visualization
- 🎯 ByteTrack object tracking for persistent IDs
- 📷 Multi-camera support (USB, RTSP, HTTP, FILE)
- 🔧 JSON-based configuration system
- 📚 Comprehensive documentation (README, QUICKSTART, CONTRIBUTING)
- 🛠️ Utility tools (find_cameras, mqtt_subscriber)
- 🐳 Docker-based MQTT broker setup
- 📝 MIT License

### Documentation
- Added detailed README.md with full setup instructions
- Created QUICKSTART.md for rapid deployment
- Added CONTRIBUTING.md for contributors
- Added comprehensive inline code documentation
- Created example configuration files

### Tools
- `tools/find_cameras.py` - Camera discovery utility
- `tools/mqtt_subscriber.py` - MQTT event monitor
- `scripts/export.py` - Model export to OpenVINO
- `scripts/setup.py` - Automated setup script

### Configuration
- JSON-based camera configuration
- MQTT broker settings
- Detection parameters (confidence, IOU)
- Zone geometry customization

### Performance
- Optimized for Intel CPUs/GPUs using OpenVINO
- FP16 precision for faster inference
- Configurable frame skipping
- ByteTrack for efficient object tracking

---

## [Unreleased]

### Planned Features
- [ ] Multi-zone detection support
- [ ] Advanced polygon zones
- [ ] GPU acceleration (Intel iGPU)
- [ ] REST API for remote control
- [ ] Database integration for event storage
- [ ] Email/SMS notification system
- [ ] Mobile app integration
- [ ] Cloud MQTT broker examples

### Under Consideration
- Support for other YOLO versions (YOLOv5, YOLOv9)
- TensorRT optimization for NVIDIA GPUs
- Kubernetes deployment examples
- Docker Compose full-stack deployment
- CI/CD pipeline setup

---

## Version History

### Version Numbering
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)

### Release Notes Format
Each release includes:
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

---

## Migration Guides

### From Pre-1.0 Versions
If upgrading from development versions:
1. Update `cameras_config.json` format (see examples)
2. Re-export models with new `scripts/export.py`
3. Update import statements if using as library
4. Check MQTT topic structure (now uses Tapway format)

---

## Support

For questions about specific versions:
- Check the [README.md](README.md) for current version
- See [Issues](https://github.com/AdibSuid/yolo-zone-detection/issues) for known problems
- Read [QUICKSTART.md](QUICKSTART.md) for setup help

---

**Note**: This changelog is manually maintained. Please report any discrepancies.
