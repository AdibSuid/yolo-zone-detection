"""Export custom YOLOv8 model to OpenVINO format optimized for Intel CPU."""
import os
from ultralytics import YOLO
import numpy as np


def export_custom_yolov5_model():
    """Export your custom YOLOv8 best.pt model to OpenVINO format."""
    
    print("🔄 Exporting Custom YOLOv8 Model to OpenVINO format...")
    print("=" * 60)
    
    # Custom model configuration
    model_config = {
        'name': 'best.pt',  # Your custom YOLOv5 model
        'output': 'custom_yolov5_openvino_model',
        'description': 'Custom YOLOv5 Model',
        'imgsz': 640  # Change this if your model was trained with different size
    }
    
    model_name = model_config['name']
    output_dir = model_config['output']
    description = model_config['description']
    imgsz = model_config['imgsz']
    
    print(f"\n📦 Processing {description}...")
    
    # Check if model file exists
    if not os.path.exists(model_name):
        print(f"❌ Model file '{model_name}' not found!")
        print(f"💡 Please place your 'best.pt' model in the root directory:")
        print(f"   {os.path.abspath(model_name)}")
        return False
    
    # Check if already exported
    if os.path.exists(output_dir):
        print(f"⚠️  {output_dir} already exists!")
        response = input("   Do you want to re-export? (y/n): ")
        if response.lower() != 'y':
            print("   Skipping export")
            return True
        print("   Removing existing directory...")
        import shutil
        shutil.rmtree(output_dir)
    
    try:
        # Load your custom model
        print(f"   Loading {model_name}...")
        model = YOLO(model_name)
        
        # Get model info
        print(f"   Model loaded successfully!")
        print(f"   Model type: {model.task}")
        print(f"   Number of classes: {len(model.names) if hasattr(model, 'names') else 'Unknown'}")
        if hasattr(model, 'names'):
            print(f"   Classes: {list(model.names.values())[:10]}...")  # Show first 10 classes
        
        # Export to OpenVINO format
        print(f"\n   Exporting to OpenVINO format...")
        print(f"   This may take a few minutes...")
        model.export(
            format="openvino",
            dynamic=False,      # Static shapes for better CPU performance
            half=False,         # FP32 for Intel CPU (better accuracy)
            int8=False,         # Keep FP32 (can quantize later for speed)
            imgsz=imgsz,        # Input size (should match your training size)
            workspace=4         # Limit workspace for CPU
        )
        
        # Rename the export directory to our expected name
        default_export_dir = model_name.replace('.pt', '_openvino_model')
        if os.path.exists(default_export_dir) and default_export_dir != output_dir:
            print(f"\n   Renaming {default_export_dir} to {output_dir}...")
            import shutil
            shutil.move(default_export_dir, output_dir)
        
        print(f"\n✅ {description} exported successfully to {output_dir}")
        
        # Quick validation test
        print(f"\n   Testing exported model...")
        test_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        openvino_model = YOLO(output_dir, task='detect')
        results = openvino_model.predict(test_image, verbose=False)
        
        if results:
            print(f"✅ Model validation passed")
            print(f"   Model is ready to use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to export {description}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for custom model export script."""
    print("=" * 60)
    print("Custom YOLOv8 Model Export Tool")
    print("=" * 60)
    print("\n📋 Prerequisites:")
    print("   1. Place your 'best.pt' model in the project root directory")
    print("   2. Ensure ultralytics package is installed")
    print("   3. Make sure you have OpenVINO toolkit installed\n")
    
    success = export_custom_yolov5_model()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 Export completed successfully!")
        print("\n📋 Next Steps:")
        print("   1. Update config.py to use 'custom_yolov5_openvino_model/'")
        print("   2. Or use --model flag when running:")
        print("      python -m src.main --camera 1 --mode custom")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Export failed. Please check the error messages above.")
        print("=" * 60)


if __name__ == "__main__":
    main()
