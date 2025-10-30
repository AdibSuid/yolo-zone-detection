"""Diagnostic script to check OpenVINO performance and configuration."""
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_openvino():
    """Check OpenVINO installation and version."""
    print_header("OpenVINO Check")
    try:
        import openvino as ov
        print(f"✅ OpenVINO installed: {ov.__version__}")
        
        # Check available devices
        core = ov.Core()
        devices = core.available_devices
        print(f"📱 Available devices: {devices}")
        
        for device in devices:
            print(f"\n   Device: {device}")
            if device == 'CPU':
                cpu_info = core.get_property(device, "FULL_DEVICE_NAME")
                print(f"   CPU: {cpu_info}")
        
        return True
    except Exception as e:
        print(f"❌ OpenVINO check failed: {e}")
        return False


def check_model_format():
    """Check if model is properly exported to OpenVINO format."""
    print_header("Model Format Check")
    
    model_path = "best_openvino_model"
    
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        print("💡 Run: python scripts/export.py")
        return False
    
    print(f"✅ Model directory exists: {model_path}")
    
    # Check for OpenVINO files
    xml_files = list(Path(model_path).glob("*.xml"))
    bin_files = list(Path(model_path).glob("*.bin"))
    yaml_files = list(Path(model_path).glob("*.yaml"))
    
    print(f"\n📁 Files found:")
    print(f"   .xml files: {len(xml_files)} {[f.name for f in xml_files]}")
    print(f"   .bin files: {len(bin_files)} {[f.name for f in bin_files]}")
    print(f"   .yaml files: {len(yaml_files)} {[f.name for f in yaml_files]}")
    
    if len(xml_files) == 0 or len(bin_files) == 0:
        print("\n❌ Missing OpenVINO model files (.xml and .bin)")
        print("💡 Your model is NOT in OpenVINO format!")
        print("💡 Run: python scripts/export.py")
        return False
    
    print("\n✅ Model appears to be in OpenVINO format")
    return True


def benchmark_inference():
    """Benchmark inference speed with different configurations."""
    print_header("Inference Speed Benchmark")
    
    try:
        from ultralytics import YOLO
        
        model_path = "best_openvino_model"
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        print(f"🔄 Loading model: {model_path}")
        
        # Set OpenVINO optimizations
        os.environ['OV_CPU_THREADS'] = str(os.cpu_count())
        os.environ['OV_CPU_BIND_THREAD'] = 'YES'
        os.environ['OV_CPU_THROUGHPUT_STREAMS'] = '1'
        
        model = YOLO(model_path, task='detect')
        print(f"✅ Model loaded")
        
        # Create dummy frame (640x640)
        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        print(f"\n🔥 Warming up (3 frames)...")
        for i in range(3):
            _ = model.predict(dummy_frame, verbose=False, device='cpu')
        
        print(f"\n⚡ Running benchmark (10 frames)...")
        times = []
        for i in range(10):
            start = time.time()
            results = model.predict(
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
            print(f"   Frame {i+1}/10: {elapsed*1000:.1f}ms")
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Average inference time: {avg_time*1000:.1f}ms")
        print(f"   Average FPS: {fps:.1f}")
        print(f"   Min time: {min(times)*1000:.1f}ms")
        print(f"   Max time: {max(times)*1000:.1f}ms")
        
        if fps < 5:
            print(f"\n⚠️  WARNING: FPS is very low ({fps:.1f})")
            print("   Possible issues:")
            print("   1. Model not in OpenVINO format (check above)")
            print("   2. CPU is slow or overloaded")
            print("   3. Model is too large for CPU")
            print("   4. OpenVINO not properly configured")
        elif fps < 10:
            print(f"\n⚠️  FPS is lower than expected ({fps:.1f})")
            print("   Consider using a lighter model or lower resolution")
        else:
            print(f"\n✅ FPS looks good ({fps:.1f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cpu_info():
    """Check CPU information."""
    print_header("CPU Information")
    
    print(f"CPU Cores: {os.cpu_count()}")
    
    try:
        import platform
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Processor: {platform.processor()}")
        print(f"Architecture: {platform.machine()}")
    except:
        pass
    
    # Check if running in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment")


def check_dependencies():
    """Check installed package versions."""
    print_header("Dependencies Check")
    
    packages = [
        'ultralytics',
        'openvino',
        'opencv-python',
        'numpy',
        'supervision'
    ]
    
    for package in packages:
        try:
            if package == 'opencv-python':
                import cv2
                version = cv2.__version__
                package_name = 'opencv-python'
            else:
                module = __import__(package)
                version = module.__version__
                package_name = package
            
            print(f"✅ {package_name}: {version}")
        except Exception as e:
            print(f"❌ {package_name}: Not found or error")


def main():
    """Run all diagnostics."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║       YOLO OpenVINO Performance Diagnostic Tool         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run all checks
    check_cpu_info()
    check_dependencies()
    openvino_ok = check_openvino()
    model_ok = check_model_format()
    
    if openvino_ok and model_ok:
        benchmark_inference()
    else:
        print("\n❌ Cannot run benchmark - fix issues above first")
    
    print("\n" + "=" * 60)
    print("Diagnostic Complete!")
    print("=" * 60)
    
    print("\n💡 Next Steps:")
    print("   1. If model is not in OpenVINO format:")
    print("      python scripts/export.py")
    print("   2. If FPS is still low after export:")
    print("      - Close other applications")
    print("      - Use lighter model (yolov8n)")
    print("      - Reduce resolution in config")
    print("   3. Replace detector.py with optimized version")


if __name__ == "__main__":
    main()