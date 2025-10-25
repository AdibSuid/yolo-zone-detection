"""Performance monitoring utilities."""
import time
from collections import deque


class PerformanceMonitor:
    """Monitor and calculate FPS and other performance metrics."""
    
    def __init__(self, window_size=30):
        """Initialize performance monitor.
        
        Args:
            window_size: Number of samples to use for FPS calculation
        """
        self.times = deque(maxlen=window_size)
        self.last_time = time.time()
        self.frame_count = 0
    
    def tick(self):
        """Record a frame timing."""
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed > 0:
            self.times.append(elapsed)
        self.last_time = current_time
        self.frame_count += 1
    
    def get_fps(self):
        """Calculate current FPS.
        
        Returns:
            Current FPS as float
        """
        if len(self.times) == 0:
            return 0
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / avg_time if avg_time > 0 else 0
    
    def get_summary(self):
        """Get performance summary statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "fps": self.get_fps(),
            "frames_processed": self.frame_count,
            "avg_frame_time_ms": (sum(self.times) / len(self.times) * 1000) if self.times else 0
        }
