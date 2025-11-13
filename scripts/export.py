"""
YOLO Model Export Script

Converts YOLOv8 PyTorch model (.pt) to OpenVINO format for optimized
inference on Intel CPUs and GPUs.

Prerequisites:
    - Place your trained model file (best.pt) in the project root
    - Ensure ultralytics and openvino packages are installed

Usage:
    python scripts/export.py

Output:
    Creates best_openvino_model/ directory with:
    - model.xml (network structure)
    - model.bin (weights)
    - metadata.yaml (model configuration)

For custom models:
    1. Change "best.pt" to your model filename
    2. Adjust imgsz parameter if needed
    3. Run this script

Export Options:
    - half=True: FP16 precision (faster, slight accuracy loss)
    - dynamic=True: Variable input sizes
    - nms=True: Includes NMS in exported model
    - imgsz: Input image size (height, width)
"""
from ultralytics import YOLO

print("=" * 60)
print("YOLO Model Export to OpenVINO Format")
print("=" * 60)

# Load the YOLOv8 model
print("\n📥 Loading YOLOv8 model from: best.pt")
model = YOLO("best.pt")

# Export the model to OpenVINO format
print("🔄 Exporting to OpenVINO format...")
print("   ├── Format: OpenVINO (.xml + .bin)")
print("   ├── Precision: FP16 (half precision)")
print("   ├── Dynamic shapes: Enabled")
print("   ├── NMS: Included")
print("   └── Input size: 480x640\n")

model.export(
    format="openvino",  # Export to OpenVINO format
    half=True,          # Use FP16 precision for speed
    dynamic=True,       # Allow dynamic input shapes
    nms=True,           # Include NMS in the model
    imgsz=(1080, 1920)    # Input image size (H, W)
)

print("\n" + "=" * 60)
print("✅ Export complete!")
print("=" * 60)
print("📁 Output directory: best_openvino_model/")
print("   ├── model.xml  (network structure)")
print("   ├── model.bin  (weights)")
print("   └── metadata.yaml  (configuration)")
print("\n💡 Next steps:")
print("   1. Verify files exist: ls -la best_openvino_model/")
print("   2. Configure camera: cp cameras_config_examples.json cameras_config.json")
print("   3. Run detection: python -m src.main --camera-id cam1")
print("=" * 60)

