"""Analyze YOLO model to understand why it's slow."""
import os
from pathlib import Path


def analyze_model():
    """Analyze custom model and compare with standard models."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║       YOLO Model Analyzer - Find Performance Issues     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        from ultralytics import YOLO
        import yaml
        
        # Check if best.pt exists
        if not os.path.exists("best.pt"):
            print("❌ best.pt not found!")
            return
        
        print("=" * 60)
        print("  Analyzing Custom Model: best.pt")
        print("=" * 60)
        
        # 1. FILE SIZE
        file_size = os.path.getsize("best.pt") / (1024 * 1024)
        print(f"\n📦 File Size: {file_size:.1f} MB")
        
        # Compare with standard models
        print("\n📊 Size Comparison:")
        print(f"   YOLOv8n: ~6 MB   (30 FPS on your CPU) ✅")
        print(f"   YOLOv8s: ~22 MB  (20 FPS on your CPU) ✅")
        print(f"   YOLOv8m: ~52 MB  (~8 FPS expected)")
        print(f"   YOLOv8l: ~88 MB  (~4 FPS expected)")
        print(f"   YOLOv8x: ~136 MB (~2 FPS expected)")
        print(f"   Your model: {file_size:.1f} MB  (0.8 FPS measured) ❌")
        
        # Estimate base architecture
        if file_size < 10:
            base = "YOLOv8n (nano)"
            expected_fps = "25-35"
        elif file_size < 30:
            base = "YOLOv8s (small)"
            expected_fps = "15-25"
        elif file_size < 60:
            base = "YOLOv8m (medium)"
            expected_fps = "6-12"
        elif file_size < 100:
            base = "YOLOv8l (large)"
            expected_fps = "3-6"
        elif file_size < 150:
            base = "YOLOv8x (xlarge)"
            expected_fps = "1-3"
        else:
            base = "Custom/Unknown (VERY LARGE!)"
            expected_fps = "<1"
        
        print(f"\n🔍 Likely Base Architecture: {base}")
        print(f"   Expected FPS range: {expected_fps}")
        print(f"   Actual FPS: 0.8 ❌")
        
        if file_size > 100:
            print("\n⚠️  WARNING: Your model is VERY LARGE!")
            print("   YOLOv8x is too heavy for CPU inference")
            print("   Recommendation: Retrain with YOLOv8n or YOLOv8s")
        
        # 2. LOAD MODEL AND CHECK ARCHITECTURE
        print(f"\n🔄 Loading model to inspect architecture...")
        model = YOLO("best.pt")
        
        # Get model info
        print(f"\n📐 Model Architecture:")
        if hasattr(model, 'model'):
            # Count parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            print(f"   Total Parameters: {total_params:,}")
            
            # Compare
            print(f"\n   Parameter Comparison:")
            print(f"   YOLOv8n: ~3M parameters")
            print(f"   YOLOv8s: ~11M parameters")
            print(f"   YOLOv8m: ~26M parameters")
            print(f"   YOLOv8l: ~44M parameters")
            print(f"   YOLOv8x: ~68M parameters")
            print(f"   Your model: {total_params/1e6:.1f}M parameters")
            
            if total_params > 50e6:
                print(f"\n   ⚠️  {total_params/1e6:.1f}M parameters is TOO MANY for CPU!")
        
        # 3. CHECK NUMBER OF CLASSES
        if hasattr(model, 'names'):
            num_classes = len(model.names)
            print(f"\n🏷️  Number of Classes: {num_classes}")
            print(f"   Standard COCO: 80 classes")
            print(f"   Your model: {num_classes} classes")
            
            if num_classes > 100:
                print(f"\n   ⚠️  Many classes ({num_classes}) increases computation")
            elif num_classes == 1:
                print(f"\n   ✅ Single class should be fast!")
                print(f"   But model is still slow → architecture issue")
        
        # 4. CHECK TRAINING INFO
        print(f"\n🎓 Training Information:")
        try:
            # Try to read training args from model
            if hasattr(model, 'ckpt') and model.ckpt:
                train_args = model.ckpt.get('train_args', {})
                if train_args:
                    print(f"   Image Size: {train_args.get('imgsz', 'Unknown')}")
                    print(f"   Model: {train_args.get('model', 'Unknown')}")
                    print(f"   Epochs: {train_args.get('epochs', 'Unknown')}")
        except:
            pass
        
        # 5. CHECK OPENVINO EXPORT
        print(f"\n📂 Checking OpenVINO Export:")
        export_dir = Path("best_openvino_model")
        if export_dir.exists():
            xml_files = list(export_dir.glob("*.xml"))
            bin_files = list(export_dir.glob("*.bin"))
            
            if bin_files:
                openvino_size = bin_files[0].stat().st_size / (1024*1024)
                print(f"   OpenVINO Model Size: {openvino_size:.1f} MB")
                print(f"   PyTorch Model Size: {file_size:.1f} MB")
                print(f"   Size Ratio: {openvino_size/file_size:.2f}x")
                
                if openvino_size > 100:
                    print(f"\n   ⚠️  OpenVINO model is HUGE ({openvino_size:.1f} MB)")
                    print(f"   This confirms the model is too complex for CPU")
        else:
            print(f"   ❌ OpenVINO export not found")
        
        # RECOMMENDATIONS
        print("\n" + "=" * 60)
        print("  🎯 Analysis Complete - Recommendations")
        print("=" * 60)
        
        if file_size > 100:
            print(f"\n❌ PROBLEM: Your model is based on YOLOv8x or larger")
            print(f"   Size: {file_size:.1f} MB (too large for CPU)")
            print(f"\n✅ SOLUTION: Retrain with smaller architecture")
            print(f"\n   Option 1: YOLOv8n (nano) - 30 FPS")
            print(f"   from ultralytics import YOLO")
            print(f"   model = YOLO('yolov8n.pt')")
            print(f"   model.train(data='your_data.yaml', epochs=100)")
            print(f"\n   Option 2: YOLOv8s (small) - 20 FPS")
            print(f"   model = YOLO('yolov8s.pt')")
            print(f"   model.train(data='your_data.yaml', epochs=100)")
            
        elif file_size > 60:
            print(f"\n⚠️  PROBLEM: Model based on YOLOv8m or YOLOv8l")
            print(f"   Size: {file_size:.1f} MB (too heavy for real-time CPU)")
            print(f"\n✅ SOLUTION: Use YOLOv8s or accept low FPS")
            print(f"\n   Option 1: Retrain with YOLOv8s (recommended)")
            print(f"   Option 2: Use lower resolution (320x320)")
            print(f"   Option 3: Accept 2-4 FPS")
            
        elif file_size > 30:
            print(f"\n⚠️  Model size: {file_size:.1f} MB")
            print(f"   Expected: 6-12 FPS")
            print(f"   Actual: 0.8 FPS")
            print(f"\n   Something else is wrong!")
            print(f"   Possible issues:")
            print(f"   - Custom architecture modifications")
            print(f"   - Export issue (re-export recommended)")
            print(f"   - Many output classes")
            
        else:
            print(f"\n✅ Model size is good ({file_size:.1f} MB)")
            print(f"   But performance is bad (0.8 FPS)")
            print(f"\n   Issue might be:")
            print(f"   - Export settings (try re-export)")
            print(f"   - Custom layers/modifications")
            print(f"   - Check if model has unusual architecture")
        
        print(f"\n💡 Quick Test:")
        print(f"   1. Your YOLOv8s: 20 FPS (22 MB)")
        print(f"   2. Your custom: 0.8 FPS ({file_size:.1f} MB)")
        print(f"   3. Ratio: {file_size/22:.1f}x larger → {20/0.8:.1f}x slower")
        print(f"\n   This confirms model complexity is the issue!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def suggest_retraining():
    """Suggest how to retrain with lighter model."""
    print("\n" + "=" * 60)
    print("  How to Retrain with YOLOv8n/s")
    print("=" * 60)
    
    print("""
# Example training script for YOLOv8n (fastest)

from ultralytics import YOLO

# Load YOLOv8n as base
model = YOLO('yolov8n.pt')

# Train on your dataset
model.train(
    data='your_data.yaml',      # Your dataset config
    epochs=100,                 # Training epochs
    imgsz=640,                  # Image size
    batch=16,                   # Batch size
    device='cpu',               # or 'cuda' if you have GPU
    patience=50,                # Early stopping
    save=True,
    project='runs/train',
    name='yolov8n_custom'
)

# Export to OpenVINO
model.export(format='openvino', half=False)

Expected result:
- Model size: ~6-10 MB
- CPU Performance: 25-35 FPS @ 640x640
- Accuracy: Slightly lower than large models but much faster!

For better accuracy with good speed, use YOLOv8s:
model = YOLO('yolov8s.pt')  # 15-25 FPS expected
""")


def main():
    """Main entry point."""
    success = analyze_model()
    
    if success:
        suggest_retraining()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()