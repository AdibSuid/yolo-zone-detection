"""YOLO detection and tracking logic."""
import traceback
from ultralytics import YOLO
import supervision as sv


class YOLODetector:
    """YOLO model inference and tracking."""
    
    def __init__(self, model_path, confidence_threshold):
        """Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.tracker = sv.ByteTrack()
        
        # Cache for smooth display on frame skips
        self.last_detections = None
        self.last_labels = []
    
    def load_model(self):
        """Load the YOLO model.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"🔄 Loading model: {self.model_path}")
        try:
            self.model = YOLO(self.model_path, task='detect')
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            if "yolov8n" in self.model_path:
                print("💡 Run: python scripts/export_model.py to create YOLOv8 models")
            return False
    
    def detect(self, frame):
        """Run detection on a frame.
        
        Args:
            frame: Input frame (NumPy array)
            
        Returns:
            Supervision Detections object or None on error
        """
        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                verbose=False,
                device='cpu'
            )
            
            if not results or len(results) == 0:
                print("Warning: Model returned empty results, using cached detections")
                return self.last_detections
            
            result = results[0]
            detections = sv.Detections.from_ultralytics(result)
            
            # Update tracker
            try:
                detections = self.tracker.update_with_detections(detections)
            except Exception as e:
                print(f"Tracker error: {e}, resetting tracker")
                self.tracker.reset()
                detections = self.tracker.update_with_detections(detections)
            
            # Cache detections and labels
            self.last_detections = detections
            if len(detections) > 0:
                self.last_labels = [
                    f"{self.model.names[int(c)]} {p:.2f}" 
                    for c, p in zip(detections.class_id, detections.confidence)
                ]
            else:
                self.last_labels = []
            
            return detections
            
        except Exception as e:
            print(f"Inference error: {e}")
            traceback.print_exc()
            return self.last_detections
    
    def get_cached_detections(self):
        """Get last cached detections for frame skipping.
        
        Returns:
            Tuple of (detections, labels)
        """
        return self.last_detections, self.last_labels
    
    def get_class_name(self, class_id):
        """Get class name from ID.
        
        Args:
            class_id: Class ID
            
        Returns:
            Class name string
        """
        if self.model is None:
            return "unknown"
        return self.model.names[int(class_id)]
