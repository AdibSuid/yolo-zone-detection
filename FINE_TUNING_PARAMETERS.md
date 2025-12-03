# YOLO Zone Detection - Fine Tuning Parameters

This document provides a comprehensive guide to all configurable parameters for fine-tuning the YOLO zone detection system's performance and behavior.

## 📋 Quick Reference Table

| Parameter | Default Value | Location | Purpose | Range/Units |
|-----------|---------------|----------|---------|-------------|
| **Confidence Threshold** | 0.5 | `config.py:79` | Detection sensitivity | 0.0 - 1.0 |
| **IOU Threshold** | 0.5 | `config.py:80` | Overlap filtering | 0.0 - 1.0 |
| **Proximity Matching** | 50 pixels | `main.py:147` | Same object detection | pixels |
| **Stability Frames** | 10 frames | `main.py:151` | Anti-flicker delay | frames |
| **Missing Frames Tolerance** | 10 frames | `main.py:148` | Tracking tolerance | frames |
| **Zone Width Ratio** | 0.3 (30%) | `config.py:125` | Detection zone width | 0.0 - 1.0 |
| **Zone Height Ratio** | 0.4 (40%) | `config.py:126` | Detection zone height | 0.0 - 1.0 |
| **Dwell Time** | 0.33 seconds | `config.py:197` | Minimum zone presence | seconds |

## 🎯 Detection Sensitivity Parameters

### 1. Confidence Threshold
**Location**: `src/config.py` - Line 79
```python
CUSTOM = {
    "conf_threshold": 0.5,  # Change this value (0.0-1.0)
}
```
- **Lower values (0.3-0.4)**: More sensitive, detects more objects but may include false positives
- **Higher values (0.6-0.8)**: Less sensitive, only high-confidence detections but may miss some objects
- **Recommended range**: 0.4-0.7 for retail environments

### 2. IOU (Intersection over Union) Threshold
**Location**: `src/config.py` - Line 80
```python
CUSTOM = {
    "iou_threshold": 0.5,   # Change this value (0.0-1.0)
}
```
- **Lower values (0.3-0.4)**: More aggressive filtering of overlapping detections
- **Higher values (0.6-0.7)**: Allows more overlapping detections
- **Recommended range**: 0.4-0.6 for crowded retail scenarios

## 🎯 Tracking & Stability Parameters

### 3. Proximity Matching Distance
**Location**: `src/main.py` - Line 147
```python
self.proximity_threshold = 50  # Change this value (pixels)
```
- **Purpose**: Determines if a "new" detection is actually the same object that lost tracking
- **Lower values (20-30px)**: Stricter matching, may create duplicate events for same person
- **Higher values (70-100px)**: Looser matching, better continuity but may merge different people
- **Recommended**: 40-60 pixels for typical retail camera distances

### 4. Stability Frames (Anti-flicker)
**Location**: `src/main.py` - Line 151
```python
self.required_stability_frames = 10  # Change this value (frames)
```
- **Purpose**: Object must remain in zone for this many consecutive frames before triggering event
- **Time Calculation**: `time_seconds = frames / fps`
  - At 30 FPS: 10 frames = 0.33 seconds
  - At 25 FPS: 10 frames = 0.40 seconds
  - At 15 FPS: 10 frames = 0.67 seconds

**Tuning Guidelines**:
- **Lower values (5-7 frames)**: Faster response, but more false triggers
- **Higher values (15-20 frames)**: More stable, but slower response to genuine entries
- **Recommended**: 8-12 frames for self-checkout scenarios

### 5. Missing Frames Tolerance
**Location**: `src/main.py` - Line 148
```python
self.max_missing_frames = 10  # Change this value (frames)
```
- **Purpose**: How long to remember an object after it disappears from detection
- **Lower values (5-8)**: Faster cleanup, but may lose tracking more easily
- **Higher values (15-20)**: Better tracking continuity, but uses more memory
- **Recommended**: 8-15 frames depending on detection reliability

## 🎯 Zone Configuration Parameters

### 6. Detection Zone Size
**Location**: `src/config.py` - Lines 125-126
```python
class ZoneConfig:
    BOX_WIDTH_RATIO = 0.3   # 30% of frame width
    BOX_HEIGHT_RATIO = 0.4  # 40% of frame height
```

**Zone Positioning**: Always centered in the frame

**Tuning Guidelines**:
- **Smaller zones (0.2 width, 0.3 height)**: More precise triggering, good for specific checkout areas
- **Larger zones (0.5 width, 0.6 height)**: Broader coverage, good for general area monitoring
- **Recommended**: 0.25-0.4 width, 0.3-0.5 height for self-checkout

## 🔧 Configuration Methods

### Method 1: Direct Code Modification
Edit the parameter values directly in the source files as shown above.

### Method 2: JSON Configuration (Recommended)
Add parameters to `cameras_config.json` for dynamic configuration:

```json
{
  "cameras": {
    "cam1": {
      "enabled": true,
      "source_type": "USB",
      "path": 0
    }
  },
  "mqtt": {
    "broker": "localhost",
    "port": 1883,
    "username": "tapway-admin",
    "password": "T@pw4yAdm1n"
  },
  "detection": {
    "site_id": "TAPWAY",
    "subgroup_id": "Live Cam",
    "conf_threshold": 0.5,
    "iou_threshold": 0.5,
    "dwell_time_seconds": 0.33,
    "proximity_threshold": 50,
    "max_missing_frames": 10,
    "zone_width_ratio": 0.3,
    "zone_height_ratio": 0.4
  }
}
```

## 📊 Performance Impact Guidelines

### High Performance (Speed Priority)
- Confidence: 0.6+
- IOU: 0.4-0.5
- Stability frames: 5-8
- Proximity threshold: 40-50px

### High Accuracy (Precision Priority)
- Confidence: 0.4-0.5
- IOU: 0.5-0.6
- Stability frames: 10-15
- Proximity threshold: 50-70px

### Balanced (Recommended for Self-Checkout)
- Confidence: 0.5
- IOU: 0.5
- Stability frames: 10
- Proximity threshold: 50px

## 🧪 Testing & Calibration

### Step 1: Baseline Testing
1. Use default parameters
2. Record performance metrics (false positives, missed detections)
3. Note problematic scenarios

### Step 2: Sensitivity Adjustment
1. If missing people: Lower confidence threshold (0.4-0.45)
2. If too many false detections: Raise confidence threshold (0.55-0.6)

### Step 3: Stability Tuning
1. If premature triggers: Increase stability frames (12-15)
2. If slow response: Decrease stability frames (7-9)

### Step 4: Zone Optimization
1. If triggers outside intended area: Reduce zone size
2. If missing edge detections: Increase zone size

## ⚡ Real-time Parameter Modification

To change parameters without code restart, implement hot-reload by:
1. Adding parameter monitoring in the main loop
2. Reloading configuration file periodically
3. Using environment variables for critical parameters

## 📈 Monitoring & Metrics

Key metrics to monitor during tuning:
- **Detection Rate**: Detections per minute
- **False Positive Rate**: Invalid triggers per hour  
- **Response Time**: Time from entry to MQTT event
- **Tracking Continuity**: Percentage of successful object tracking
- **MQTT Success Rate**: Percentage of successful message publishes

## 🔍 Troubleshooting Common Issues

### Issue: Too Many False Triggers
**Solutions**:
- Increase confidence threshold (0.55-0.6)
- Increase stability frames (12-15)
- Reduce zone size

### Issue: Missing Real Detections  
**Solutions**:
- Decrease confidence threshold (0.4-0.45)
- Increase proximity threshold (60-70px)
- Check lighting conditions

### Issue: Duplicate Events for Same Person
**Solutions**:
- Increase proximity threshold (60-80px)
- Increase missing frames tolerance (15-20)
- Check tracking algorithm performance

### Issue: Slow Response Time
**Solutions**:
- Decrease stability frames (7-9)
- Optimize model inference
- Check camera FPS settings

---

## 📝 Change Log Template

When modifying parameters, document changes:

```
Date: YYYY-MM-DD
Parameter: [parameter_name]
Old Value: [old_value]
New Value: [new_value]  
Reason: [why changed]
Result: [performance impact]
```

---

*Last Updated: December 3, 2025*
*System Version: YOLO Zone Detection v1.0*