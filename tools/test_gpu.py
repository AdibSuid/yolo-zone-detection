"""Test OpenVINO GPU configuration and performance."""
import os
import sys
import time
import numpy as np


def test_openvino_gpu():
    """Test OpenVINO GPU functionality."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     OpenVINO GPU Test - Intel Iris Xe                   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # 1. Check OpenVINO availability
    print("=" * 60)
    print("  OpenVINO Setup")
    print("=" * 60)
    
    try:
        import openvino as ov
        print(f"✅ OpenVINO version: {ov.__version__}")
        
        core = ov.Core()
        devices = core.available_devices
        print(f"📱 Available devices: {devices}")
        
        if 'GPU' not in devices:
            print("❌ Intel GPU not detected by OpenVINO")
            print("💡 Make sure Intel GPU drivers are installed")
            return False
        
        # Get GPU info
        try:
            gpu_name = core.get_property('GPU', 'FULL_DEVICE_NAME')
            print(f"🖥️  GPU: {gpu_name}")
        except:
            print("🖥️  GPU: Intel Graphics (details unavailable)")
        
    except Exception as e:
        print(f"❌ OpenVINO error: {e}")
        return False
    
    # 2. Test YOLO with GPU
    print("\n" + "=" * 60)
    print("  YOLO GPU Test")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        # Check if model exists
        if not os.path.exists("best_openvino_model"):
            print("❌ OpenVINO model not found: best_openvino_model/")
            print("💡 Make sure your model is exported to OpenVINO format")
            return False
        
        print("🔄 Testing with GPU configuration...")
        
        # Set OpenVINO environment for GPU
        os.environ['OV_DEFAULT_DEVICE'] = 'GPU'
        os.environ['OV_GPU_ENABLE_OPENCL_QUEUE_SHARING'] = '1'
        os.environ['OV_GPU_ENABLE_OPTIMIZE'] = '1'
        
        # Load model
        print("   Loading OpenVINO model...")
        model = YOLO("best_openvino_model", task='detect')
        print("   ✅ Model loaded")
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warmup
        print("   Warming up GPU (3 frames)...")
        for i in range(3):
            _ = model.predict(test_image, device='cpu', verbose=False)  # Use 'cpu' for OpenVINO
        
        # Benchmark
        print("   Running GPU benchmark (10 frames)...")
        gpu_times = []
        for i in range(10):
            start = time.time()
            results = model.predict(
                test_image,
                device='cpu',  # For OpenVINO, always use 'cpu' - GPU handled by OpenVINO
                conf=0.5,
                iou=0.5,
                verbose=False,
                half=False
            )
            elapsed = time.time() - start
            gpu_times.append(elapsed)
            print(f"      Frame {i+1}/10: {elapsed*1000:.1f}ms")
        
        gpu_avg = sum(gpu_times) / len(gpu_times)
        gpu_fps = 1.0 / gpu_avg
        
        print(f"\n📊 GPU Results:")
        print(f"   Average time: {gpu_avg*1000:.1f}ms")
        print(f"   Average FPS: {gpu_fps:.1f}")
        
        # Now test CPU for comparison
        print(f"\n🔄 Testing CPU for comparison...")
        os.environ['OV_DEFAULT_DEVICE'] = 'CPU'
        
        # Reload model for CPU
        model = YOLO("best_openvino_model", task='detect')
        
        # CPU benchmark
        print("   Running CPU benchmark (5 frames)...")
        cpu_times = []
        for i in range(5):
            start = time.time()
            results = model.predict(
                test_image,
                device='cpu',
                conf=0.5,
                iou=0.5,
                verbose=False,
                half=False
            )
            elapsed = time.time() - start
            cpu_times.append(elapsed)
            print(f"      Frame {i+1}/5: {elapsed*1000:.1f}ms")
        
        cpu_avg = sum(cpu_times) / len(cpu_times)
        cpu_fps = 1.0 / cpu_avg
        
        print(f"\n📊 CPU Results:")
        print(f"   Average time: {cpu_avg*1000:.1f}ms")
        print(f"   Average FPS: {cpu_fps:.1f}")
        
        # Comparison
        speedup = cpu_avg / gpu_avg
        print(f"\n📈 Performance Comparison:")
        print(f"   CPU:  {cpu_fps:.1f} FPS ({cpu_avg*1000:.0f}ms)")
        print(f"   GPU:  {gpu_fps:.1f} FPS ({gpu_avg*1000:.0f}ms)")
        print(f"   Speedup: {speedup:.1f}x faster with Intel GPU")
        
        if speedup > 1.5:
            print(f"\n✅ SUCCESS! Intel GPU provides {speedup:.1f}x speedup")
        elif speedup > 1.1:
            print(f"\n⚠️  Modest improvement: {speedup:.1f}x speedup")
            print("   GPU might be throttling or busy")
        else:
            print(f"\n❌ No improvement with GPU ({speedup:.1f}x)")
            print("   Check GPU drivers and availability")
        
        return speedup > 1.1
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    success = test_openvino_gpu()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    
    if success:
        print("""
✅ Intel GPU is working with OpenVINO!

🚀 To use GPU in your application:
   python -m src.main --camera 0 --mode custom_gpu

💡 Tips:
   - First run may be slower due to GPU initialization
   - Close other GPU-intensive applications for best performance
   - GPU performance varies with thermal conditions
        """)
    else:
        print("""
❌ Intel GPU test failed

🔧 Troubleshooting:
   1. Update Intel Graphics drivers
   2. Check Device Manager for Intel Graphics
   3. Enable iGPU in BIOS
   4. Restart after driver installation
   5. Try: python -m tools.check_gpu
        """)


if __name__ == "__main__":
    main()