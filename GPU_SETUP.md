# Intel Iris Xe GPU Setup Guide

## Quick Start

To run your YOLO zone detection system using Intel Iris Xe iGPU instead of CPU:

### 1. Check GPU Availability
```bash
python -m tools.check_gpu
```

### 2. Run with GPU Acceleration
```bash
# Option A: Use existing model with auto GPU detection
python -m src.main --camera 0 --mode custom

# Option B: Use dedicated GPU mode
python -m src.main --camera 0 --mode custom_gpu
```

### 3. Export Model for GPU (Optional, for best performance)
```bash
python -m scripts.export_gpu
python -m src.main --camera 0 --mode custom_gpu
```

## What Changed

### 1. Device Configuration
- Modified `src/detector.py` to accept device parameter
- Added auto-detection of available devices
- GPU uses FP16 precision, CPU uses FP32

### 2. New Performance Mode
- Added `custom_gpu` mode in `src/config.py`
- Forces GPU usage for inference
- Optimized settings for Intel GPU

### 3. New Scripts
- `tools/check_gpu.py` - Check Intel GPU availability
- `scripts/export_gpu.py` - Export model optimized for GPU

### 4. Command Line Options
```bash
# Available modes now include:
--mode custom      # CPU inference (auto-detect best device)
--mode custom_gpu  # Force Intel GPU inference
```

## Expected Performance Improvements

| Configuration | FPS | Inference Time | Use Case |
|--------------|-----|----------------|----------|
| CPU (current) | 0.8-2 FPS | 500-1250ms | Low power |
| Intel Iris Xe | 3-8 FPS | 125-330ms | Real-time |

## Prerequisites

1. **Hardware**: Intel CPU with Iris Xe graphics (11th gen+)
2. **Drivers**: Latest Intel Graphics drivers
3. **Software**: OpenVINO toolkit (already installed)

## Troubleshooting

### GPU Not Detected
1. Update Intel Graphics drivers from Intel website
2. Check Device Manager for "Intel(R) Iris(R) Xe Graphics"
3. Enable iGPU in BIOS if using discrete GPU
4. Install Intel Graphics Command Center

### Poor GPU Performance
1. Ensure FP16 precision is enabled (automatic in GPU mode)
2. Close other applications using GPU
3. Check thermal throttling
4. Try lower resolution: modify `resolution` in config

### Fallback to CPU
The system automatically falls back to CPU if:
- No Intel GPU detected
- GPU drivers not installed
- GPU busy with other tasks

## Performance Tips

1. **Use GPU mode**: `--mode custom_gpu` for best performance
2. **Export for GPU**: Run `scripts/export_gpu.py` first
3. **Lower resolution**: Change resolution in config for more FPS
4. **Close other apps**: Free up GPU resources
5. **Update drivers**: Keep Intel Graphics drivers current

## Verification

After setup, you should see:
```
🖥️  Target device: GPU
📱 Available OpenVINO devices: ['CPU', 'GPU']
✅ Intel GPU detected and will be used!
```

Expected output:
- FPS should increase from ~1 to 3-8 FPS
- Inference time should decrease significantly
- GPU utilization visible in Task Manager