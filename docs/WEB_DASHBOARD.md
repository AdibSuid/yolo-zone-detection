# Web Dashboard Guide

## Overview

The web dashboard provides a real-time browser-based interface for monitoring the YOLO zone detection system. It displays live video feed, detection statistics, and detailed information about detected objects.

## Features

### 📹 Live Video Feed
- Real-time MJPEG video stream from the camera
- Shows all detection overlays (bounding boxes, labels, zone outline)
- Updates continuously during inference

### 📊 Statistics Panel
- **Total Object Count**: Cumulative count of all objects detected in the zone
- **Camera ID**: Fixed identifier for the camera (001)
- **Camera Name**: Friendly name (Ringo)
- **Model Name**: YOLOv8 model being used
- **Model Logic Type**: Detection mode (IN/OUT)
- **Model Zone**: Zone configuration (Random)

### 📸 Recent Detections
- List of the 50 most recent detections
- Each detection shows:
  - **Cropped image** of the detected object
  - **Object class** (person, car, etc.)
  - **Confidence score** (percentage)
  - **Direction** (IN/OUT badge)
  - **Timestamp** (YYYY/MM/DD HH:MM:SS format)

## Accessing the Dashboard

### Local Access
When you start the detection system, you'll see output like:
```
🌐 Starting Web Dashboard on http://0.0.0.0:5000
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.68.112:5000
📊 Open dashboard at http://localhost:5000
```

Open your browser and navigate to:
- **http://localhost:5000** (on the same computer)
- **http://192.168.68.112:5000** (from other devices on the network - use your actual IP)

### Network Access
The dashboard is accessible from any device on your local network:
1. Find your computer's IP address (shown in the console output)
2. On another device (phone, tablet, another computer), open a browser
3. Navigate to `http://[your-ip]:5000`

## Real-Time Updates

The dashboard uses WebSocket (SocketIO) for real-time updates:
- **Video Feed**: Updates continuously via MJPEG stream
- **Detection Events**: New detections appear instantly
- **Object Count**: Updates immediately when objects are detected
- No page refresh needed!

## Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│  YOLO Zone Detection Dashboard                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────────┐  ┌─────────────────────┐   │
│  │                       │  │  Total Objects      │   │
│  │   Live Video Feed     │  │       42            │   │
│  │                       │  └─────────────────────┘   │
│  │                       │                            │
│  │                       │  Camera Info:              │
│  │                       │  ├─ ID: 001                │
│  │                       │  ├─ Name: Ringo            │
│  └───────────────────────┘  ├─ Model: yolov8         │
│                             ├─ Logic: IN/OUT          │
│                             └─ Zone: Random           │
├─────────────────────────────────────────────────────────┤
│  Recent Detections                                     │
├─────────────────────────────────────────────────────────┤
│  ┌──────┬────────┬────────┬───────┬──────────────────┐│
│  │ IMG  │ Class  │ Conf.  │ Dir.  │ Time             ││
│  ├──────┼────────┼────────┼───────┼──────────────────┤│
│  │ [🖼] │ person │ 95.3%  │  IN   │ 2024/01/15 14:23 ││
│  │ [🖼] │ car    │ 87.2%  │  OUT  │ 2024/01/15 14:22 ││
│  │ [🖼] │ dog    │ 92.1%  │  IN   │ 2024/01/15 14:21 ││
│  └──────┴────────┴────────┴───────┴──────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## Technical Details

### Technology Stack
- **Backend**: Flask 3.0 + Flask-SocketIO 5.3
- **Frontend**: HTML5 + JavaScript + Socket.IO client
- **Video Streaming**: MJPEG over HTTP
- **Real-time Updates**: WebSocket (SocketIO)

### API Endpoints
- `GET /`: Dashboard HTML page
- `GET /video_feed`: MJPEG video stream
- WebSocket events:
  - `new_detection`: Sent when object detected
  - `update_count`: Sent when count changes
  - `initial_data`: Sent on client connection

### Detection Direction Logic
- **IN**: Object enters the detection zone (first detection)
- **OUT**: Object exits the detection zone (tracked objects leaving)

Note: Currently all detections are marked as "IN" as they are detected inside the zone. Full IN/OUT tracking based on zone entry/exit will be enhanced in future updates.

### Image Format
Cropped object images are:
- Extracted from detection bounding boxes
- Encoded as Base64 PNG
- Embedded directly in the HTML (no separate image files)

## Troubleshooting

### Dashboard not loading
1. Check if the application is running
2. Verify the URL matches the console output
3. Ensure port 5000 is not blocked by firewall
4. Try http://127.0.0.1:5000 instead of localhost

### Video feed not showing
1. Check camera is working (OpenCV window should show video)
2. Refresh the browser page
3. Check browser console for errors (F12)

### Detections not appearing
1. Verify objects are detected in the OpenCV window
2. Check if objects are inside the detection zone (red box)
3. Refresh the page to reload the SocketIO connection

### Count not updating
1. Check browser console for SocketIO connection errors
2. Ensure objects are being detected in the zone
3. Verify the zone is correctly configured

## Configuration

### Changing Port
Edit `src/main.py`:
```python
app = ZoneDetectionApp(
    mode=args.mode,
    camera_index=args.camera,
    mqtt_broker=args.mqtt_broker,
    mqtt_port=args.mqtt_port,
    enable_web=True,
    web_port=8080  # Change to desired port
)
```

### Disabling Web Dashboard
Set `enable_web=False` in the ZoneDetectionApp initialization.

### Camera/Model Names
Edit the fixed values in `src/main.py` `__init__` method:
```python
self.web_dashboard = WebDashboard(
    camera_id="002",        # Change camera ID
    camera_name="MyCamera", # Change camera name
    model_name="yolov8n"    # Change model name
)
```

## Browser Compatibility

Tested and working on:
- ✅ Chrome/Edge (Chromium-based)
- ✅ Firefox
- ✅ Safari
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## Performance

- Video feed: ~10-15 FPS (depends on detection FPS)
- Detection latency: <100ms (real-time updates)
- Maximum detections stored: 50 (configurable in web_dashboard.py)
- Network bandwidth: ~1-2 Mbps for video stream

## Security Notes

⚠️ **Important**: The web server binds to `0.0.0.0` (all network interfaces) for easy access from other devices. This means:
- Anyone on your local network can access the dashboard
- The server is NOT secured with authentication
- Do NOT expose port 5000 to the internet without proper security

For production use, consider:
- Adding authentication (Flask-Login, HTTP Basic Auth)
- Using HTTPS (Flask-Talisman, reverse proxy)
- Restricting network access (firewall rules)
- Using a production WSGI server (Gunicorn, uWSGI)

## Examples

### Starting with Web Dashboard
```bash
# Default (web enabled by default)
python -m src.main --camera 1 --mode balanced

# Access dashboard
http://localhost:5000
```

### Viewing from Another Device
```bash
# On detection computer, note the IP shown:
# Running on http://192.168.1.100:5000

# On phone/tablet browser:
http://192.168.1.100:5000
```

## Future Enhancements

Potential improvements:
- [ ] Full IN/OUT direction tracking based on zone crossing
- [ ] Object trajectory visualization
- [ ] Historical statistics and graphs
- [ ] Export detection data (CSV, JSON)
- [ ] Multi-camera support
- [ ] Alert/notification system
- [ ] Dark mode toggle
- [ ] Customizable zone editing from web UI

---

**Need Help?** Check the main README.md or open an issue on GitHub.
