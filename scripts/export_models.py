"""Export YOLO models to OpenVINO format optimized for Intel CPU."""
import os
from ultralytics import YOLO
import numpy as np


def export_yolo_models():
    """Export YOLOv8 models to OpenVINO format with CPU optimizations."""
    
    print("🔄 Exporting YOLO models to OpenVINO format...")
    print("=" * 60)
    
    # Model configurations
    models_to_export = [
        {
            'name': 'yolov8n.pt',
            'output': 'yolov8n_openvino_model',
            'description': 'YOLOv8 Nano (fastest, good for real-time)',
            'imgsz': 640
        },
        {
            'name': 'yolov8s.pt',
            'output': 'yolov8s_openvino_model',
            'description': 'YOLOv8 Small (balanced accuracy/speed)',
            'imgsz': 640
        },
    ]
    
    for model_config in models_to_export:
        model_name = model_config['name']
        output_dir = model_config['output']
        description = model_config['description']
        imgsz = model_config['imgsz']
        
        print(f"\n📦 Processing {description}...")
        
        # Check if model already exported
        if os.path.exists(output_dir):
            print(f"✅ {output_dir} already exists, skipping export")
            continue
        
        try:
            # Load model (will download if not present)
            print(f"   Loading {model_name}...")
            model = YOLO(model_name)
            
            # Export to OpenVINO format
            print(f"   Exporting to OpenVINO format...")
            model.export(
                format="openvino",
                dynamic=False,      # Static shapes for better CPU performance
                half=False,         # FP32 for Intel CPU (better accuracy)
                int8=False,         # Keep FP32 for now (can quantize later)
                imgsz=imgsz,        # Standard input size
                workspace=4         # Limit workspace for CPU
            )
            
            print(f"✅ {description} exported successfully to {output_dir}")
            
            # Quick validation test
            print(f"   Testing exported model...")
            test_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
            openvino_model = YOLO(output_dir, task='detect')
            results = openvino_model.predict(test_image, verbose=False)
            
            if results:
                print(f"✅ Model validation passed")
            
        except Exception as e:
            print(f"❌ Failed to export {description}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("📋 Model Recommendations:")
    print("   🏃 Ultra Fast:   python -m src.main --mode ultra_fast")
    print("   ⚡ Maximum FPS:  python -m src.main --mode maximum_fps")
    print("   ⚖️  Balanced:     python -m src.main --mode balanced")
    print("   🎯 High Accuracy: python -m src.main --mode high_accuracy")
    print("=" * 60)
    
    print("\n✅ Export completed!")


def main():
    """Main entry point for model export script."""
    export_yolo_models()


if __name__ == "__main__":
    main()
