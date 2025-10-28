"""Optimized YOLO detection for retail object tracking."""
import traceback
from ultralytics import YOLO
import supervision as sv


class YOLODetector:
    """Optimized YOLO detector for Intel CPU."""
    
    def __init__(self, model_path, confidence_threshold, iou_threshold=0.5):
        """Initialize optimized YOLO detector."""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        
        # OPTIMIZED: ByteTrack with retail-optimized parameters
        # Using simplified initialization for compatibility
        self.tracker = sv.ByteTrack()
        
        # Cache for smooth display
        self.last_detections = None
        self.last_labels = []
    
    def load_model(self):
        """Load YOLO model with OpenVINO optimization."""
        print(f"🔄 Loading OpenVINO model: {self.model_path}")
        try:
            # Load OpenVINO model
            self.model = YOLO(self.model_path, task='detect')
            
            # Verify it's OpenVINO format
            if 'openvino' not in self.model_path.lower():
                print("⚠️  Warning: Model may not be OpenVINO format")
                print("💡 Run: python scripts/export_custom_model.py")
            
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            traceback.print_exc()
            return False
    
    def detect(self, frame):
        """Run optimized detection on frame."""
        try:
            # OPTIMIZED: Minimal predict parameters for speed
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device='cpu',
                # OPTIMIZED: Additional speed parameters
                half=False,          # FP32 for CPU
                agnostic_nms=False,  # Class-specific NMS is faster
                max_det=50           # Limit detections (retail typically <50 objects)
            )
            
            if not results or len(results) == 0:
                return self.last_detections
            
            result = results[0]
            detections = sv.Detections.from_ultralytics(result)
            
            # Update tracker with error handling
            try:
                detections = self.tracker.update_with_detections(detections)
            except Exception as e:
                print(f"Tracker error: {e}, resetting")
                self.tracker.reset()
                detections = self.tracker.update_with_detections(detections)
            
            # Cache for frame skipping
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
        """Get cached detections."""
        return self.last_detections, self.last_labels
    
    def get_class_name(self, class_id):
        """Get class name from ID."""
        if self.model is None:
            return "unknown"
        return self.model.names[int(class_id)]