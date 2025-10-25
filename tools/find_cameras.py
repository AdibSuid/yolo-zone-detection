"""Find and test available cameras."""
import cv2


def find_cameras(max_cameras=10):
    """Find all available camera indices.
    
    Args:
        max_cameras: Maximum number of cameras to check
        
    Returns:
        List of camera information dictionaries
    """
    print("🔍 Searching for available cameras...")
    print("=" * 50)
    
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to confirm it works
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                print(f"✅ Camera {i} found:")
                print(f"   Resolution: {width}x{height}")
                print(f"   FPS: {fps}")
                print(f"   Frame shape: {frame.shape}")
                
                # Try to get backend info
                backend = cap.getBackendName()
                print(f"   Backend: {backend}")
                print()
                
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend
                })
            cap.release()
    
    if not available_cameras:
        print("❌ No cameras found!")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure your USB camera is plugged in")
        print("   2. Check if camera is being used by another application")
        print("   3. Try different USB ports")
    else:
        print("=" * 50)
        print(f"📊 Found {len(available_cameras)} camera(s)")
        print("\n📝 To use a camera, run with --camera option:")
        for cam in available_cameras:
            print(f"   python -m src.main --camera {cam['index']}  # {cam['width']}x{cam['height']} @ {cam['fps']} FPS")
        
        print("\n💡 Common setup:")
        print("   Index 0 = Built-in webcam (usually)")
        print("   Index 1 = First external USB camera")
        print("   Index 2 = Second external USB camera")
    
    return available_cameras


def test_cameras_interactive(cameras):
    """Test each camera with live preview.
    
    Args:
        cameras: List of camera information from find_cameras()
    """
    if not cameras:
        return
    
    print("\n" + "=" * 50)
    print("🧪 Testing each camera (press 'q' to skip to next)")
    print("=" * 50)
    
    for cam in cameras:
        idx = cam['index']
        print(f"\n📷 Testing Camera {idx}...")
        
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print("   Press 'q' to skip to next camera")
            print("   Press 'ESC' to exit all tests")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("   ⚠️ Failed to read frame")
                    break
                
                # Add text overlay
                cv2.putText(frame, f"Camera {idx} - Press 'q' for next, ESC to exit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"{cam['width']}x{cam['height']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(f"Camera Test - Index {idx}", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == 27:  # ESC key
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\n👋 Test completed!")
                    return
            
            cap.release()
            cv2.destroyAllWindows()
    
    print("\n✅ All camera tests completed!")


def main():
    """Main entry point for camera finder tool."""
    cameras = find_cameras()
    
    if cameras:
        test_cameras_interactive(cameras)
        
        print("\nUpdate your command with the camera index you want to use:")
        print("   python -m src.main --camera X  # Replace X with your chosen index")
    
    print("\n👋 Done!")


if __name__ == "__main__":
    main()
