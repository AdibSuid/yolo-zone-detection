# Using Your Custom YOLOv5 Model

This guide explains how to use your custom YOLOv5 `best.pt` model with the YOLO Zone Detection system, optimized with OpenVINO.

## Prerequisites

1. Your trained YOLOv5 `best.pt` model file
2. Python environment with all dependencies installed
3. OpenVINO toolkit (automatically handled by ultralytics)

## Step-by-Step Instructions

### 1. Place Your Model File

Copy your `best.pt` model file to the project root directory:

```
yolo-zone-detection/
├── best.pt  ← Place your model here
├── src/
├── scripts/
└── ...
```

### 2. Export Model to OpenVINO Format

Run the export script:

```bash
python scripts/export_custom_model.py
```

This will:
- Load your `best.pt` model
- Export it to OpenVINO format (optimized for Intel CPU)
- Create `custom_yolov5_openvino_model/` directory
- Validate the exported model

**Expected output:**
```
✅ Custom YOLOv5 Model exported successfully to custom_yolov5_openvino_model
✅ Model validation passed
```

### 3. Run Detection with Your Custom Model

Use the `--mode custom` flag:

```bash
# With web dashboard (default)
python -m src.main --camera 1 --mode custom

# Without web dashboard
python -m src.main --camera 1 --mode custom --no-web

# List all available modes
python -m src.main --list-modes
```

## Configuration

### Adjust Model Parameters

Edit `src/config.py` to fine-tune the custom model settings:

```python
CUSTOM = {
    "name": "Custom YOLOv5",
    "model": "custom_yolov5_openvino_model/",
    "resolution": (640, 640),      # Match your training size
    "frame_skip": 1,               # Process every frame
    "conf_threshold": 0.35,        # Confidence threshold (0-1)
    "annotation_thickness": 2,
    "text_scale": 0.5
}
```

**Key parameters to adjust:**

- `resolution`: Should match the size your model was trained on (e.g., 320, 416, 640)
- `conf_threshold`: Lower = more detections, Higher = fewer but more confident
- `frame_skip`: 1 = every frame, 2 = every other frame (for performance)

### Model Training Size

If your YOLOv5 model was trained with a specific image size, update the export script:

In `scripts/export_custom_model.py`, line 16:
```python
'imgsz': 640  # Change to 320, 416, 512, or your training size
```

Then re-run the export script.

## Troubleshooting

### Model Export Fails

**Error:** `Model file 'best.pt' not found!`
- **Solution:** Ensure `best.pt` is in the project root directory

**Error:** `Failed to load model`
- **Solution:** Verify your `best.pt` file is a valid YOLOv5 PyTorch model
- Try: `python -c "from ultralytics import YOLO; model = YOLO('best.pt'); print(model.names)"`

### Low Detection Accuracy

1. **Lower confidence threshold:**
   - Edit `src/config.py`, change `conf_threshold` from `0.35` to `0.25` or lower

2. **Check resolution:**
   - Ensure the resolution in config matches your training size

3. **Test model directly:**
   ```bash
   python -c "from ultralytics import YOLO; model = YOLO('custom_yolov5_openvino_model/'); results = model.predict('path/to/test/image.jpg', show=True)"
   ```

### Performance Issues

1. **Increase frame skip:**
   - Change `frame_skip` from `1` to `2` or `3` in config

2. **Reduce resolution:**
   - Use smaller resolution like `(416, 416)` or `(320, 320)`

3. **Use faster mode as base:**
   - Copy settings from `ULTRA_FAST` or `MAXIMUM_FPS` modes

## Model Information

Your custom model will automatically:
- ✅ Use your custom class names (from training)
- ✅ Use your custom number of classes
- ✅ Be optimized for Intel CPU using OpenVINO
- ✅ Support all zone detection features
- ✅ Work with MQTT publishing
- ✅ Display in web dashboard

## Example Commands

```bash
# Run with custom model on camera 1
python -m src.main --camera 1 --mode custom

# Run with custom model and specific MQTT broker
python -m src.main --camera 1 --mode custom --mqtt-broker 192.168.1.100

# Run without web dashboard (OpenCV window only)
python -m src.main --camera 1 --mode custom --no-web

# View all available performance modes
python -m src.main --list-modes
```

## Performance Comparison

| Mode | Model | Resolution | Speed | Accuracy |
|------|-------|------------|-------|----------|
| ultra_fast | YOLOv8n | 320x320 | Fastest | Good |
| maximum_fps | YOLOv8n | 416x416 | Very Fast | Good |
| balanced | YOLOv8s | 640x480 | Medium | Better |
| high_accuracy | YOLOv8s | 640x480 | Slower | Best |
| **custom** | **Your YOLOv5** | **Configurable** | **Depends** | **Your Model** |

## Advanced: Re-export with Different Settings

If you need to re-export with different settings:

```bash
# Remove old export
rm -rf custom_yolov5_openvino_model/

# Edit scripts/export_custom_model.py if needed
# Then re-export
python scripts/export_custom_model.py
```

## Next Steps

1. Test your custom model with different camera feeds
2. Adjust confidence threshold for optimal detection
3. Configure zone size in `src/config.py` → `ZoneConfig`
4. Set up MQTT broker for event publishing
5. Access web dashboard at `http://localhost:5000`

## Support

If you encounter issues:
1. Check that `best.pt` is a valid YOLOv5/YOLOv8 model
2. Verify ultralytics package is up to date: `pip install -U ultralytics`
3. Ensure OpenVINO export completed successfully
4. Test model with simple prediction before using in main app
