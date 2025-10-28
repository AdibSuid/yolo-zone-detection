"""Optimized camera capture for maximum FPS on Intel CPU."""
import cv2
import time
from .config import CameraConfig


class CameraManager:
    """Optimized camera manager for retail detection."""
    
    def __init__(self, camera_index, resolution):
        """Initialize camera manager."""
        self.camera_index = camera_index
        self.resolution = resolution
        self.cap = None
        
    def initialize(self):
        """Initialize and configure camera for optimal performance."""
        print(f"📷 Opening camera {self.camera_index}...")
        
        # Use DirectShow for Windows (CAP_DSHOW) for better performance
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_index}")
            print("💡 Run: python -m tools.find_cameras")
            return False
        
        # Verify camera works
        ret, test_frame = self.cap.read()
        if not ret:
            self.cap.release()
            print(f"❌ Camera {self.camera_index} can't read frames")
            return False
        
        print(f"✅ Camera ready: {test_frame.shape}")
        
        # Configure for maximum performance
        self._configure_camera_optimized()
        
        # Get actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"📐 Resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
        
        return True
    
    def _configure_camera_optimized(self):
        """Apply optimized camera settings for Intel CPU inference."""
        frame_width, frame_height = self.resolution
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        # Set FPS target
        self.cap.set(cv2.CAP_PROP_FPS, CameraConfig.FPS_TARGET)
        
        # OPTIMIZED: Minimize buffer for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CameraConfig.BUFFER_SIZE)
        
        # Let camera use default auto settings for brightness/exposure/contrast
        
        # OPTIMIZED: Reduced warmup
        print("⏳ Warming up camera...")
        time.sleep(CameraConfig.WARMUP_TIME)
        for _ in range(CameraConfig.WARMUP_FRAMES):
            self.cap.read()
        print("✅ Camera warmed up")
    
    def read(self):
        """Read frame from camera."""
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def get_resolution(self):
        """Get actual camera resolution."""
        if self.cap is None:
            return self.resolution
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None