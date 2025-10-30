"""Export YOLO model optimized for Intel GPU inference."""
import os
from pathlib import Path


def export_for_gpu():
    """Export model with GPU-optimized settings."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║    Export YOLO Model for Intel Iris Xe GPU              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        from ultralytics import YOLO
        
        # Check if model exists
        if not os.path.exists("best.pt"):
            print("❌ best.pt not found!")
            print("💡 Make sure you have your trained model in the root directory")
            return False
        
        print("🔄 Loading model...")
        model = YOLO("best.pt")
        print("✅ Model loaded")
        
        # Export with GPU-optimized settings
        print("\n🔄 Exporting for Intel GPU...")
        print("   Settings:")
        print("   - Format: OpenVINO")
        print("   - Precision: FP16 (GPU optimized)")
        print("   - Image size: 640")
        print("   - Batch: 1")
        print("   - Dynamic: False")
        print("   - Simplify: True")
        print()
        print("⏳ This will take 1-2 minutes...")
        
        # Export with GPU optimization
        export_path = model.export(
            format="openvino",
            half=True,          # FP16 for GPU acceleration
            imgsz=640,
            batch=1,
            dynamic=False,      # Static shapes for better GPU performance
            simplify=True,      # Simplify ONNX graph
            int8=False,         # No int8 quantization
            optimize=True,      # Enable optimizations
        )
        
        print(f"\n✅ Export complete!")
        print(f"   Path: {export_path}")
        
        # Rename to indicate GPU optimization
        export_dir = Path(export_path)
        gpu_export_dir = export_dir.parent / "best_openvino_model_gpu"
        
        if gpu_export_dir.exists():
            import shutil
            shutil.rmtree(gpu_export_dir)
        
        export_dir.rename(gpu_export_dir)
        print(f"   Renamed to: {gpu_export_dir}")
        
        # Update config to use GPU model
        print(f"\n💡 To use GPU model, either:")
        print(f"   1. Run with: python -m src.main --mode custom_gpu")
        print(f"   2. Or manually update src/config.py to point to: {gpu_export_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gpu_support():
    """Check if Intel GPU is available."""
    print("\n🔍 Checking Intel GPU availability...")
    
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        
        print(f"📱 Available devices: {devices}")
        
        if 'GPU' in devices:
            print("✅ Intel GPU detected!")
            # Try to get GPU info
            try:
                gpu_name = core.get_property('GPU', 'FULL_DEVICE_NAME')
                print(f"   GPU Name: {gpu_name}")
            except:
                print("   GPU Name: Could not retrieve")
            return True
        else:
            print("❌ No Intel GPU detected")
            print("💡 Make sure Intel GPU drivers are installed")
            return False
            
    except Exception as e:
        print(f"❌ Could not check GPU support: {e}")
        return False


def main():
    """Main entry point."""
    gpu_available = check_gpu_support()
    
    if not gpu_available:
        print("\n⚠️  Warning: No Intel GPU detected!")
        print("   You can still export, but it won't provide benefits")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    success = export_for_gpu()
    
    if success:
        print("\n" + "=" * 60)
        print("GPU Export Complete! 🎉")
        print("=" * 60)
        print("""
Next steps:

1. Test GPU performance:
   python -m src.main --mode custom_gpu --camera 0

2. Compare with CPU:
   python -m src.main --mode custom --camera 0

3. Expected improvements with Intel Iris Xe:
   - 2-4x faster inference
   - Better power efficiency
   - Reduced CPU usage

Note: First run may be slower due to GPU driver initialization.
        """)


if __name__ == "__main__":
    main()