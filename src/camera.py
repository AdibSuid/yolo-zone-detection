"""Camera capture and management."""
import cv2
import time
from .config import CameraConfig


class CameraManager:
    """Manage camera initialization and configuration."""
    
    def __init__(self, camera_index, resolution):
        """Initialize camera manager.
        
        Args:
            camera_index: Camera device index
            resolution: Tuple of (width, height)
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.cap = None
        
    def initialize(self):
        """Initialize and configure the camera.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"📷 Opening camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_index}")
            print("💡 Run: python tools/find_cameras.py to find available cameras")
            return False
        
        # Verify camera works
        ret, test_frame = self.cap.read()
        if not ret:
            self.cap.release()
            print(f"❌ Camera {self.camera_index} opened but can't read frames")
            return False
        
        print(f"✅ Camera ready: {test_frame.shape}")
        
        # Configure camera settings
        self._configure_camera()
        
        # Get actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📐 Resolution: {actual_width}x{actual_height}")
        
        return True
    
    def _configure_camera(self):
        """Apply camera configuration settings."""
        frame_width, frame_height = self.resolution
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CameraConfig.BUFFER_SIZE)
        
        # Configure auto-exposure for good brightness (15 FPS)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, CameraConfig.AUTO_EXPOSURE_MODE)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, CameraConfig.BRIGHTNESS)
        self.cap.set(cv2.CAP_PROP_CONTRAST, CameraConfig.CONTRAST)
        
        # Warm up camera and stabilize auto-exposure
        time.sleep(CameraConfig.WARMUP_TIME)
        for _ in range(CameraConfig.WARMUP_FRAMES):
            self.cap.read()
    
    def read(self):
        """Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def get_resolution(self):
        """Get actual camera resolution.
        
        Returns:
            Tuple of (width, height)
        """
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
