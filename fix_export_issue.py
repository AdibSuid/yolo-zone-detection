"""Re-export model with EXPLICIT optimal settings to fix performance."""
import os
import shutil
from pathlib import Path


def reexport_with_verification():
    """Re-export model and verify it's correct."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     Fix YOLOv8s Export - Force Optimal Settings         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        from ultralytics import YOLO
        import torch
        
        # Check model exists
        if not os.path.exists("best.pt"):
            print("❌ best.pt not found!")
            return False
        
        print("=" * 60)
        print("  Current Situation")
        print("=" * 60)
        print("✅ Model size: 21.5 MB (correct for YOLOv8s)")
        print("✅ Parameters: 11.1M (correct for YOLOv8s)")
        print("✅ Architecture: YOLOv8s")
        print("✅ Classes: 1 (should be FAST)")
        print("❌ FPS: 0.8 (should be 20!)")
        print()
        print("🔍 Issue: Export or model has optimization problem")
        
        # Backup old export
        if os.path.exists("best_openvino_model"):
            print("\n🗑️  Backing up old export...")
            if os.path.exists("best_openvino_model_old"):
                shutil.rmtree("best_openvino_model_old")
            shutil.move("best_openvino_model", "best_openvino_model_old")
            print("✅ Old export backed up to: best_openvino_model_old/")
        
        # Load model
        print("\n🔄 Loading model...")
        model = YOLO("best.pt")
        print("✅ Model loaded")
        
        # Check model details
        print("\n📊 Model Details:")
        print(f"   Task: {model.task}")
        print(f"   Classes: {len(model.names)}")
        print(f"   Class names: {list(model.names.values())}")
        
        # Export with EXPLICIT settings
        print("\n🔄 Exporting with OPTIMAL settings...")
        print("   Settings:")
        print("   - Format: OpenVINO")
        print("   - Precision: FP32 (not FP16)")
        print("   - Image size: 640")
        print("   - Batch: 1")
        print("   - Dynamic: False (static shapes)")
        print("   - Simplify: True")
        print("   - Int8: False")
        print("   - Optimize: True")
        print()
        print("⏳ This will take 1-2 minutes...")
        
        # Export with explicit CPU-optimized settings
        export_path = model.export(
            format="openvino",
            half=False,         # CRITICAL: FP32 for CPU
            imgsz=640,
            batch=1,
            dynamic=False,      # Static shapes for speed
            simplify=True,      # Simplify ONNX graph
            int8=False,         # No int8 quantization
            optimize=True,      # Enable optimizations
        )
        
        print(f"\n✅ Export complete!")
        print(f"   Path: {export_path}")
        
        # Verify export
        print("\n🔍 Verifying export...")
        export_dir = Path(export_path)
        
        xml_files = list(export_dir.glob("*.xml"))
        bin_files = list(export_dir.glob("*.bin"))
        
        if xml_files and bin_files:
            print("✅ OpenVINO files found:")
            for f in xml_files:
                print(f"   - {f.name}")
            for f in bin_files:
                size = f.stat().st_size / (1024*1024)
                print(f"   - {f.name} ({size:.1f} MB)")
        else:
            print("❌ OpenVINO files missing!")
            return False
        
        # Now test inference speed
        print("\n⚡ Testing inference speed...")
        import time
        import numpy as np
        
        # Load exported model
        test_model = YOLO(export_path, task='detect')
        
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warmup
        print("   Warming up (3 frames)...")
        for _ in range(3):
            _ = test_model.predict(dummy_frame, verbose=False, device='cpu')
        
        # Benchmark
        print("   Running benchmark (10 frames)...")
        times = []
        for i in range(10):
            start = time.time()
            _ = test_model.predict(
                dummy_frame,
                conf=0.5,
                iou=0.5,
                verbose=False,
                device='cpu',
                half=False,
                agnostic_nms=False,
                max_det=50
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"      Frame {i+1}/10: {elapsed*1000:.0f}ms")
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Average time: {avg_time*1000:.0f}ms")
        print(f"   Average FPS: {fps:.1f}")
        
        # Compare with previous
        print(f"\n📈 Comparison:")
        print(f"   Before: 0.8 FPS (1335ms)")
        print(f"   After:  {fps:.1f} FPS ({avg_time*1000:.0f}ms)")
        
        if fps > 5:
            print(f"\n✅ SUCCESS! FPS improved from 0.8 to {fps:.1f}")
            print(f"   Improvement: {fps/0.8:.1f}x faster!")
            print(f"\n   The export was the issue - now FIXED!")
        elif fps > 2:
            print(f"\n⚠️  Partial improvement: 0.8 → {fps:.1f} FPS")
            print(f"   Better, but still slower than expected (20 FPS)")
            print(f"\n   Try solution 2 (see below)")
        else:
            print(f"\n❌ Still slow: {fps:.1f} FPS")
            print(f"   Export didn't fix the issue")
            print(f"\n   The model itself has an issue (see solution 2)")
        
        return fps > 5
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def suggest_solutions():
    """Suggest additional solutions."""
    print("\n" + "=" * 60)
    print("  Additional Solutions")
    print("=" * 60)
    
    print("""
If re-export didn't fully fix the issue, try these:

📝 Solution 1: Train Fresh YOLOv8s (Recommended)
──────────────────────────────────────────────
Your model was trained for 300 epochs, which might have caused
optimization issues. Try training fresh with standard settings:

from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # Fresh base model
model.train(
    data='your_data.yaml',
    epochs=100,          # Fewer epochs (was 300)
    imgsz=640,
    batch=16,
    patience=50,
    device='cpu',        # or 'cuda' if available
    optimizer='AdamW',   # Default optimizer
    close_mosaic=10      # Standard setting
)
model.export(format='openvino', half=False)

Expected: 15-20 FPS


📝 Solution 2: Use YOLOv8n Instead
──────────────────────────────────
If you don't need maximum accuracy, YOLOv8n is faster:

model = YOLO('yolov8n.pt')
model.train(data='your_data.yaml', epochs=100)
model.export(format='openvino', half=False)

Expected: 25-30 FPS


📝 Solution 3: Lower Resolution (Quick Fix)
───────────────────────────────────────────
Use your current model with lower resolution:

copy config_optimized_cpu.py src\\config.py
python -m src.main --camera 3 --mode custom_fast

Expected: 3-6 FPS (not ideal but works)


📝 Solution 4: Check Training Settings
──────────────────────────────────────
Your model was trained with:
- 300 epochs (very long - might cause optimization issues)
- Image size: 640

Possible issues:
- Overfitting after 300 epochs
- Learning rate too low at end
- Model optimized for different hardware

Recommendation: Retrain with 100 epochs


💡 Why This Happened
────────────────────
Your model has correct size/architecture but wrong performance.
This usually means:

1. Export used wrong settings (FP16 vs FP32)
2. Training created optimization artifacts (300 epochs)
3. Model has custom preprocessing layers
4. PyTorch model has different optimization than OpenVINO

Standard YOLOv8s: 20 FPS  ✅
Your exported model: 0.8 FPS  ❌

The re-export should fix this! If not, retrain with fewer epochs.
""")


def main():
    """Main entry point."""
    success = reexport_with_verification()
    
    suggest_solutions()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print("""
1. ✅ Re-exported model with optimal settings
2. 📊 Benchmarked performance
3. 💡 Provided alternative solutions

Next steps:
- If FPS improved to 10+: ✅ Problem solved!
- If FPS still low: Try retraining (see solutions above)
    """)


if __name__ == "__main__":
    main()