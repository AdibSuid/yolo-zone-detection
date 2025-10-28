# Quick Start: Custom YOLOv5 Model

## 🚀 3-Step Setup

### Step 1: Place Your Model
```bash
# Copy your best.pt to the project root
cp /path/to/your/best.pt ./best.pt
```

### Step 2: Export to OpenVINO
```bash
python scripts/export_custom_model.py
```

### Step 3: Run Detection
```bash
python -m src.main --camera 1 --mode custom
```

## 📋 Full Workflow

```bash
# 1. Place model file
ls best.pt  # Should exist in project root

# 2. Export to OpenVINO (one-time setup)
python scripts/export_custom_model.py
# Output: custom_yolov5_openvino_model/

# 3. Test model export
python -c "from ultralytics import YOLO; model = YOLO('custom_yolov5_openvino_model/'); print('✅ Model loaded')"

# 4. List available modes (should show 'custom')
python -m src.main --list-modes

# 5. Run with custom model
python -m src.main --camera 1 --mode custom

# 6. Run without web dashboard (for testing)
python -m src.main --camera 1 --mode custom --no-web
```

## ⚙️ Configuration

**File:** `src/config.py`

```python
CUSTOM = {
    "name": "Custom YOLOv5",
    "model": "custom_yolov5_openvino_model/",
    "resolution": (640, 640),      # ⬅️ Match your training size!
    "frame_skip": 1,
    "conf_threshold": 0.35,        # ⬅️ Adjust for your model
    "annotation_thickness": 2,
    "text_scale": 0.5
}
```

## 🔧 Common Adjustments

### Your model was trained on 320x320
Edit `scripts/export_custom_model.py` line 16:
```python
'imgsz': 320  # Changed from 640
```

Edit `src/config.py` CUSTOM mode:
```python
"resolution": (320, 320),  # Changed from (640, 640)
```

### Too many/few detections
Edit `src/config.py`:
```python
"conf_threshold": 0.25,  # Lower = more detections
"conf_threshold": 0.50,  # Higher = fewer, more confident
```

### Performance issues
Edit `src/config.py`:
```python
"frame_skip": 2,          # Process every 2nd frame
"resolution": (416, 416), # Smaller resolution
```

## 🎯 What Gets Detected?

Your custom model's classes will be used automatically!

```bash
# Check what classes your model detects
python -c "from ultralytics import YOLO; m = YOLO('best.pt'); print(m.names)"
```

## 📊 File Structure After Export

```
yolo-zone-detection/
├── best.pt                              # Your original model
├── custom_yolov5_openvino_model/        # Exported OpenVINO model
│   ├── yolov5.xml                       # Model architecture
│   ├── yolov5.bin                       # Model weights
│   └── metadata.yaml                    # Model metadata
├── scripts/
│   └── export_custom_model.py           # Export script
└── src/
    └── config.py                         # Configuration (CUSTOM mode)
```

## ✅ Verification Checklist

- [ ] `best.pt` exists in project root
- [ ] Export script completed successfully
- [ ] `custom_yolov5_openvino_model/` directory created
- [ ] Can run: `python -m src.main --list-modes` (shows "custom")
- [ ] Application starts without errors
- [ ] Objects are detected and tracked
- [ ] Zone detection works
- [ ] MQTT events publish (if broker running)
- [ ] Web dashboard shows live feed

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `best.pt not found` | Copy file to project root |
| `Export fails` | Check ultralytics: `pip install -U ultralytics` |
| `No detections` | Lower conf_threshold in config.py |
| `Too slow` | Increase frame_skip or reduce resolution |
| `Wrong classes` | Verify you exported the correct best.pt |
| `Model error` | Test: `YOLO('best.pt').predict('image.jpg')` |

## 📚 See Also

- Full documentation: `docs/CUSTOM_MODEL.md`
- Configuration guide: `docs/USAGE.md`
- Web dashboard setup: `docs/WEB_DASHBOARD.md`
