# Architecture Overview

Visual guide to the YOLO Zone Detection system architecture.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOLO Zone Detection System                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ run.bat  │  │ run.sh   │  │  CLI     │  │ Display  │      │
│  │ / .sh    │  │          │  │  Args    │  │ Window   │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────▲─────┘      │
│       │             │             │             │              │
└───────┼─────────────┼─────────────┼─────────────┼──────────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER (src/)                     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │                  main.py (Entry Point)                 │    │
│  │                 ZoneDetectionApp class                 │    │
│  │  • Initialize components                              │    │
│  │  • Run main loop                                      │    │
│  │  • Handle shutdown                                    │    │
│  └───────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐           │
│         │                    │                    │           │
│         ▼                    ▼                    ▼           │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐       │
│  │ camera.py│        │detector  │        │  mqtt    │       │
│  │          │        │   .py    │        │ _client  │       │
│  │ Camera   │◄───────│          │───────►│   .py    │       │
│  │ Manager  │        │ YOLO     │        │ MQTT     │       │
│  │          │        │ Detector │        │ Publisher│       │
│  └────┬─────┘        └────┬─────┘        └────┬─────┘       │
│       │                   │                   │              │
│       │                   │                   │              │
│  ┌────┴─────┐        ┌────┴─────┐        ┌────┴─────┐       │
│  │performance│        │ config.py│        │          │       │
│  │   .py     │        │          │        │          │       │
│  │           │        │ • Performance      │          │       │
│  │ Perf      │        │   Modes  │        │          │       │
│  │ Monitor   │        │ • Camera │        │          │       │
│  │           │        │   Config │        │          │       │
│  └───────────┘        │ • Zone   │        │          │       │
│                       │   Config │        │          │       │
│                       │ • MQTT   │        │          │       │
│                       │   Config │        │          │       │
│                       │ • Display│        │          │       │
│                       │   Config │        │          │       │
│                       └──────────┘        │          │       │
└─────────────────────────────────────────────┼──────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EXTERNAL SYSTEMS                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Camera     │  │ OpenVINO     │  │ MQTT Broker  │        │
│  │   Hardware   │  │ Runtime      │  │ (Mosquitto)  │        │
│  │              │  │              │  │              │        │
│  │ USB/Built-in │  │ YOLOv8n/s    │  │ localhost    │        │
│  │ DirectShow   │  │ IR Models    │  │ :1883        │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
main.py
  ├── config.py (PerformanceMode, CameraConfig, ZoneConfig, etc.)
  ├── camera.py (CameraManager)
  │     └── config.py (CameraConfig)
  ├── detector.py (YOLODetector)
  │     └── ultralytics (YOLO)
  │     └── supervision (ByteTrack, Detections)
  ├── mqtt_client.py (MQTTPublisher)
  │     └── paho.mqtt.client
  │     └── config.py (MQTTConfig)
  └── performance.py (PerformanceMonitor)
```

## Data Flow

```
Camera Frame
    │
    ▼
┌─────────────┐
│  Camera     │  (camera.py)
│  Manager    │  • Read frame
│             │  • Auto-exposure
└──────┬──────┘
       │ Raw frame
       ▼
┌─────────────┐
│  YOLO       │  (detector.py)
│  Detector   │  • Run inference
│             │  • Track objects
│             │  • Cache results
└──────┬──────┘
       │ Detections
       ▼
┌─────────────┐
│  Zone       │  (main.py)
│  Detection  │  • Check in zone
│             │  • Trigger events
└──────┬──────┘
       │
       ├──────────────────┐
       │                  │
       ▼                  ▼
┌─────────────┐    ┌─────────────┐
│  Display    │    │  MQTT       │
│  Annotate   │    │  Publisher  │
│             │    │             │
│  • Boxes    │    │  • Event    │
│  • Labels   │    │    JSON     │
│  • Zone     │    │  • Publish  │
│  • FPS      │    └──────┬──────┘
└──────┬──────┘           │
       │                  │
       ▼                  ▼
   cv2.imshow      MQTT Broker
   (2.5x scaled)   (topic: cv/zone_events)
```

## Configuration Hierarchy

```
PerformanceMode
  ├── ULTRA_FAST
  ├── MAXIMUM_FPS
  ├── BALANCED (default)
  └── HIGH_ACCURACY
      │
      └── Contains:
          • model: Path to OpenVINO model
          • resolution: (width, height)
          • frame_skip: Number of frames to skip
          • conf_threshold: Detection confidence
          • annotation_thickness: Line width
          • text_scale: Font size

CameraConfig
  • AUTO_EXPOSURE_MODE
  • BRIGHTNESS
  • CONTRAST
  • BUFFER_SIZE
  • WARMUP_TIME
  • MAX_FAILED_READS

ZoneConfig
  • BOX_WIDTH_RATIO
  • BOX_HEIGHT_RATIO
  • create_box_polygon() → NumPy array

MQTTConfig
  • DEFAULT_BROKER
  • DEFAULT_PORT
  • TOPIC
  • get_client_id()

DisplayConfig
  • SCALE_FACTOR
  • WINDOW_NAME_TEMPLATE
```

## Execution Flow

```
1. Parse CLI arguments
   └── mode, camera, mqtt_broker, mqtt_port

2. Create ZoneDetectionApp instance
   └── Initialize all components

3. app.initialize()
   ├── Camera initialization
   │   ├── Open camera
   │   ├── Configure settings
   │   └── Warm up
   ├── Load YOLO model
   ├── Create detection zone
   └── Connect to MQTT broker

4. app.run() - Main Loop
   ├── Read frame
   │
   ├── Should run inference?
   │   ├── YES: Run YOLO detection
   │   │   ├── Inference
   │   │   ├── Tracking (ByteTrack)
   │   │   ├── Zone check
   │   │   └── MQTT publish
   │   └── NO: Use cached detections
   │
   ├── Annotate frame
   │   ├── Draw boxes
   │   ├── Add labels
   │   ├── Draw zone
   │   └── Add overlays
   │
   ├── Display frame (2.5x)
   │
   └── Check for 'q' key
       ├── YES: break
       └── NO: continue

5. app.shutdown()
   ├── Release camera
   ├── Close display window
   ├── Disconnect MQTT
   └── Print summary
```

## Class Diagram

```
┌─────────────────────────┐
│  ZoneDetectionApp       │
├─────────────────────────┤
│ - config                │
│ - camera                │
│ - detector              │
│ - mqtt                  │
│ - perf_monitor          │
│ - polygon_zone          │
├─────────────────────────┤
│ + initialize()          │
│ + run()                 │
│ + process_frame()       │
│ + shutdown()            │
└──┬──────────────────────┘
   │
   │ uses
   │
   ├───► CameraManager
   │       │ - cap
   │       │ + initialize()
   │       │ + read()
   │       │ + release()
   │
   ├───► YOLODetector
   │       │ - model
   │       │ - tracker
   │       │ + load_model()
   │       │ + detect()
   │
   ├───► MQTTPublisher
   │       │ - client
   │       │ + connect()
   │       │ + publish_zone_event()
   │
   └───► PerformanceMonitor
           │ - times
           │ + tick()
           │ + get_fps()
```

## File Organization

```
src/
├── __init__.py          # Package initialization
├── main.py              # Application entry point (300+ lines)
├── config.py            # Configuration classes (150+ lines)
├── camera.py            # Camera management (100+ lines)
├── detector.py          # YOLO detection (100+ lines)
├── mqtt_client.py       # MQTT publisher (80+ lines)
└── performance.py       # FPS monitoring (50+ lines)

Total: ~800 lines
Previous: ~600 lines in one file
Improvement: Better organized, more maintainable
```

## Technology Stack

```
┌─────────────────────────────────────────┐
│           Application Layer             │
│  Python 3.8+ with modular structure     │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│         Core Dependencies               │
├─────────────────────────────────────────┤
│ • ultralytics 8.3.221  (YOLOv8)        │
│ • openvino 2024.6.0    (Intel CPU)     │
│ • opencv-python 4.11   (Computer Vision)│
│ • supervision 0.26.1   (Detection Utils)│
│ • paho-mqtt 2.1.0      (MQTT Client)   │
│ • numpy                (Arrays)         │
└─────────────────────────────────────────┘
```

## Deployment Architecture

```
Development:
  Python venv → Local files → Testing

Production:
  Git clone → venv setup → Export models → Run

Docker (future):
  Dockerfile → Container → Run
```

---

This architecture ensures:
✅ **Modularity** - Each component has a single responsibility  
✅ **Maintainability** - Easy to find and fix issues  
✅ **Extensibility** - Simple to add new features  
✅ **Testability** - Components can be tested independently  
✅ **Clarity** - Clear data flow and dependencies  
