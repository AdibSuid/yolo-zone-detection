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
            import numpy as np
        
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
        
            # Validate detections before tracking
            if len(detections) == 0:
                return self.last_detections
        
            # Check for invalid bounding boxes
            if detections.xyxy is not None and len(detections.xyxy) > 0:
                # Remove any detections with invalid coordinates (NaN, inf, or zero area)
                valid_mask = []
                for bbox in detections.xyxy:
                    x1, y1, x2, y2 = bbox
                    # Check for NaN or inf
                    if not (np.isfinite(x1) and np.isfinite(y1) and 
                            np.isfinite(x2) and np.isfinite(y2)):
                        valid_mask.append(False)
                        continue
                    # Check for valid area
                    width = x2 - x1
                    height = y2 - y1
                    if width > 0 and height > 0:
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
            
                if not any(valid_mask):
                    return self.last_detections
            
                # Filter detections to keep only valid ones
                valid_mask = np.array(valid_mask)
                detections.xyxy = detections.xyxy[valid_mask]
                if detections.confidence is not None:
                    detections.confidence = detections.confidence[valid_mask]
                if detections.class_id is not None:
                    detections.class_id = detections.class_id[valid_mask]
        
            # Update tracker with error handling and numpy error suppression
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    detections = self.tracker.update_with_detections(detections)
            except Exception as e:
                print(f"Tracker error: {e}, resetting")
                self.tracker.reset()
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        detections = self.tracker.update_with_detections(detections)
                except Exception as e2:
                    print(f"Tracker reset failed: {e2}, using raw detections")
                    # Continue without tracking
        
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