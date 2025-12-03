"""
YOLO Zone Detection System - Main Application Module

This module provides the core detection system that:
- Captures video from USB/RTSP/HTTP cameras or files
- Runs YOLOv8 object detection optimized with OpenVINO for Intel hardware
- Tracks objects using ByteTrack algorithm
- Monitors configurable detection zones
- Publishes MQTT events in Tapway format
- Optionally provides web dashboard visualization

Usage:
    # Using camera from configuration file
    python -m src.main --camera-id cam1
    
    # With web dashboard
    python -m src.main --camera-id cam1 --web
    
    # Using camera index directly
    python -m src.main --camera 0
    
    # Maximum performance (no display)
    python -m src.main --camera-id cam1 --no-display

Author: YOLO Detection System
License: MIT
"""
import sys
import cv2
import time
import traceback
import argparse
import threading
import supervision as sv
import numpy as np
import warnings
warnings.filterwarnings('error')  # Convert warnings to exceptions

# Suppress runtime warnings from ByteTrack tracker
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

from .config import PerformanceMode, ZoneConfig, DisplayConfig, CameraConfig, CameraConfigManager, MQTTConfig
from .camera import CameraManager
from .detector import YOLODetector
from .mqtt_client import TapwayMQTTPublisher, MQTTPublisher
from .performance import PerformanceMonitor

# Import web dashboard only if needed
try:
    from .web_dashboard import WebDashboard
    WEB_DASHBOARD_AVAILABLE = True
except ImportError:
    WEB_DASHBOARD_AVAILABLE = False


class ZoneDetectionApp:
    """Optimized YOLO zone detection for Intel CPU."""
    
    def __init__(self, mode, camera_index=None, camera_id=None, mqtt_broker=None, mqtt_port=None, 
                 enable_web=False, web_port=5000, enable_display=True):
        """Initialize optimized application.
        
        Args:
            mode: Performance mode name
            camera_index: Camera device index (for legacy compatibility)
            camera_id: Camera ID from config file (e.g., 'cam-1')
            mqtt_broker: MQTT broker address (optional, loaded from config if not provided)
            mqtt_port: MQTT broker port (optional, loaded from config if not provided)
            enable_web: Enable web dashboard (default False for performance)
            web_port: Web dashboard port
            enable_display: Enable OpenCV display window
        """
        self.config = PerformanceMode.get_mode(mode)
        self.mode = mode
        self.camera_id = camera_id
        self.camera_index = camera_index
        self.enable_display = enable_display
        
        # Load camera configuration if camera_id is provided
        if camera_id:
            camera_config = CameraConfigManager.get_camera_config(camera_id)
            if not camera_config:
                raise ValueError(f"Camera '{camera_id}' not found in configuration")
            if not camera_config.get("enabled", False):
                raise ValueError(f"Camera '{camera_id}' is disabled in configuration")
            
            # Get camera source from config
            self.camera_source = camera_config.get("path", 0)
            self.camera_resolution = tuple(camera_config.get("stream_resolution", [640, 480]))
        else:
            # Legacy mode: use camera_index
            self.camera_source = camera_index or 0
            self.camera_resolution = self.config['resolution']
            self.camera_id = f"camera_{self.camera_source}"
        
        # Initialize components
        self.camera = CameraManager(self.camera_source, self.camera_resolution)
        
        iou_threshold = self.config.get('iou_threshold', 0.5)
        self.detector = YOLODetector(
            self.config['model'], 
            self.config['conf_threshold'],
            iou_threshold
        )
        
        # Initialize MQTT with Tapway format
        self.mqtt = TapwayMQTTPublisher(self.camera_id, mode)
        self.perf_monitor = PerformanceMonitor()
        
        # Web dashboard (optional, disabled by default)
        self.enable_web = enable_web
        self.web_port = web_port
        self.web_dashboard = None
        if enable_web:
            if WEB_DASHBOARD_AVAILABLE:
                self.web_dashboard = WebDashboard(
                    camera_id="001",
                    camera_name="Camera",
                    model_name="yolov8"
                )
            else:
                print("⚠️  Web dashboard requested but dependencies missing")
                print("💡 Install: pip install flask flask-socketio flask-cors")
                self.enable_web = False
        
        # Supervision components
        self.box_annotator = sv.BoxAnnotator(
            thickness=self.config['annotation_thickness']
        )
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=1,
            text_scale=self.config['text_scale']
        )
        self.polygon_zone = None
        self.zone_annotator = None
        
        # State
        self.running = False
        self.frame_idx = 0
        self.failed_reads = 0
        
        # Enhanced object tracking for zone detection
        self.objects_in_zone = set()  # tracker_ids currently in zone
        self.object_states = {}  # tracker_id -> {"center": (x,y), "class_id": int, "last_seen": frame_idx}
        self.published_objects = set()  # tracker_ids that already published entry event
        self.proximity_threshold = 50  # pixels for matching lost objects
        self.max_missing_frames = 10  # frames to keep object state after disappearing
        
        # Staging area for 1-second delay before MQTT publishing
        self.staging_objects = {}  # tracker_id -> {"first_seen": frame_idx, "last_in_zone": frame_idx, "data": {...}}
        self.required_stability_frames = 5  # ~1 second at 30 FPS
        
        # Performance tracking
        self.inference_times = []
    
    def initialize(self):
        """Initialize all components."""
        print(f"🚀 OPTIMIZED YOLO Zone Detection for Intel CPU")
        print(f"   Mode: {self.config['name']}")
        print(f"   Model: {self.config['model']}")
        print(f"   Resolution: {self.config['resolution'][0]}x{self.config['resolution'][1]}")
        print(f"   Confidence: {self.config['conf_threshold']}")
        print(f"   IOU Threshold: {self.config['iou_threshold']}")
        print(f"   Camera: Index {self.camera_index}")
        print(f"   Display: {'Enabled' if self.enable_display else 'Disabled'}")
        print(f"   Web Dashboard: {'Enabled' if self.enable_web else 'Disabled (use --web to enable)'}")
        print("=" * 60)
        
        # Initialize camera
        if not self.camera.initialize():
            return False
        
        # Initialize zone
        width, height = self.camera.get_resolution()
        box_polygon = ZoneConfig.create_box_polygon(width, height)
        self.polygon_zone = sv.PolygonZone(polygon=box_polygon)
        self.zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.polygon_zone,
            color=sv.Color.RED,
            thickness=2,
            text_thickness=0,
            text_scale=0,
            display_in_zone_count=False
        )
        
        # Load YOLO model
        if not self.detector.load_model():
            return False
        
        # Connect to MQTT
        self.mqtt.connect()
        
        # Start web dashboard
        if self.enable_web and self.web_dashboard:
            print(f"🌐 Starting Web Dashboard on http://0.0.0.0:{self.web_port}")
            web_thread = threading.Thread(
                target=self.web_dashboard.start,
                kwargs={'host': '0.0.0.0', 'port': self.web_port, 'debug': False},
                daemon=True
            )
            web_thread.start()
            time.sleep(2)
        
        print("🎬 Starting inference... Press 'q' to quit")
        if self.enable_web:
            print(f"📊 Dashboard: http://localhost:{self.web_port}")
        print("=" * 60)
        return True
    
    def process_frame(self, frame):
        """Process single frame with optimizations."""
        # Determine if inference should run
        should_run_inference = (
            self.config['frame_skip'] == 1 or 
            self.frame_idx % self.config['frame_skip'] == 0
        )
        
        inference_time = 0
        
        if should_run_inference:
            self.perf_monitor.tick()
            self._last_frame = frame.copy()
            
            # Run detection
            inference_start = time.time()
            detections = self.detector.detect(frame)
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            
            # Keep last 100 inference times for stats
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            if detections is not None and len(detections) > 0:
                try:
                    detections_in_zone = self.polygon_zone.trigger(detections=detections)
                    self._publish_zone_events(detections, detections_in_zone)
                except Exception as e:
                    print(f"Zone detection error: {e}")
        
        # Get cached detections
        detections, labels = self.detector.get_cached_detections()
        
        # Annotate frame (only if display or web enabled)
        if self.enable_display or self.enable_web:
            if detections is not None and len(detections) > 0:
                try:
                    frame = self.box_annotator.annotate(scene=frame, detections=detections)
                    if len(labels) == len(detections):
                        frame = self.label_annotator.annotate(
                            scene=frame,
                            detections=detections,
                            labels=labels
                        )
                except Exception as e:
                    print(f"Annotation error: {e}")
            
            # Draw zone
            try:
                frame = self.zone_annotator.annotate(scene=frame)
            except Exception as e:
                print(f"Zone annotation error: {e}")
            
            # Add performance overlay
            self._add_overlay(frame, inference_time, should_run_inference, detections)
        
        return frame
    
    def _publish_zone_events(self, detections, zone_mask):
        """Publish MQTT events with enhanced tracking and 1-second delay."""
        fps = self.perf_monitor.get_fps()
        current_in_zone = set()
        current_states = {}
        
        # Clean up old missing objects and staging
        self._cleanup_missing_objects()
        self._cleanup_staging_objects()
        
        if any(zone_mask):
            for det_idx, inside_zone in enumerate(zone_mask):
                if inside_zone:
                    tracker_id = int(detections.tracker_id[det_idx])
                    class_id = int(detections.class_id[det_idx])
                    confidence = float(detections.confidence[det_idx])
                    class_name = self.detector.get_class_name(class_id)
                    
                    # Calculate object center
                    bbox = detections.xyxy[det_idx]
                    x1, y1, x2, y2 = bbox
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    current_in_zone.add(tracker_id)
                    current_states[tracker_id] = {
                        "center": (center_x, center_y),
                        "class_id": class_id,
                        "last_seen": self.frame_idx
                    }
                    
                    # Handle staging for 1-second delay
                    self._handle_object_staging(tracker_id, class_id, class_name, confidence, bbox, center_x, center_y, fps)
        
        # Process staged objects ready for publishing
        self._process_staged_objects()
        
        # Update tracking state
        self.objects_in_zone = current_in_zone
        self.object_states.update(current_states)
        
        # Don't immediately remove objects from published set - let cleanup handle it
        # This prevents duplicate events when objects temporarily lose tracking
    
    def _should_publish_entry(self, tracker_id, center_x, center_y, class_id):
        """Determine if an object entry should trigger MQTT publication."""
        # If already published, don't publish again
        if tracker_id in self.published_objects:
            return False
        
        # Check if this might be a re-tracked existing object (spatial proximity check)
        for existing_id, state in self.object_states.items():
            if existing_id in self.published_objects:
                # Calculate distance to existing published object
                existing_x, existing_y = state["center"]
                distance = ((center_x - existing_x) ** 2 + (center_y - existing_y) ** 2) ** 0.5
                
                # If very close to a recently published object of same class, likely same object
                if distance < self.proximity_threshold and state["class_id"] == class_id:
                    frames_since_seen = self.frame_idx - state["last_seen"]
                    if frames_since_seen <= self.max_missing_frames:
                        print(f"🔄 Tracking continuity: ID {tracker_id} matches published object {existing_id} (distance: {distance:.1f}px)")
                        # Mark as published to prevent duplicate
                        self.published_objects.add(tracker_id)
                        return False
        
        # New genuine entry - not previously published and not near any published object
        return True
    
    def _cleanup_missing_objects(self):
        """Remove old object states to prevent memory buildup."""
        cutoff_frame = self.frame_idx - self.max_missing_frames * 2
        to_remove = []
        
        for obj_id, state in self.object_states.items():
            if state["last_seen"] < cutoff_frame:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            self.object_states.pop(obj_id, None)
            self.published_objects.discard(obj_id)
        
        # Also remove published objects that haven't been seen for a long time
        published_to_remove = []
        for obj_id in self.published_objects:
            if obj_id not in self.object_states:
                # Object not in current states - it's been gone for max_missing_frames * 2
                published_to_remove.append(obj_id)
        
        for obj_id in published_to_remove:
            self.published_objects.discard(obj_id)
    
    def _handle_object_staging(self, tracker_id, class_id, class_name, confidence, bbox, center_x, center_y, fps):
        """Handle object staging for 1-second delay before MQTT publishing."""
        # Check if this is a genuine new entry
        should_stage = self._should_stage_entry(tracker_id, center_x, center_y, class_id)
        
        if should_stage:
            # Add to staging area if not already there
            if tracker_id not in self.staging_objects:
                self.staging_objects[tracker_id] = {
                    "first_seen": self.frame_idx,
                    "last_in_zone": self.frame_idx,
                    "data": {
                        "tracker_id": tracker_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": bbox,
                        "center": (center_x, center_y),
                        "fps": fps
                    }
                }
                print(f"⏰ Staging: {class_name} (ID:{tracker_id}) - waiting for 0.3s in zone...")
        
        # Update staging data for objects already being tracked
        if tracker_id in self.staging_objects:
            # Update last seen in zone timestamp
            self.staging_objects[tracker_id]["last_in_zone"] = self.frame_idx
            # Update object data
            self.staging_objects[tracker_id]["data"].update({
                "confidence": confidence,
                "bbox": bbox,
                "center": (center_x, center_y)
            })
    
    def _should_stage_entry(self, tracker_id, center_x, center_y, class_id):
        """Determine if an object should be staged for delayed publication."""
        # If already published, don't stage again
        if tracker_id in self.published_objects:
            return False
            
        # If already staging, don't stage again
        if tracker_id in self.staging_objects:
            return False
        
        # Check if this might be a re-tracked existing object (spatial proximity check)
        for existing_id, state in self.object_states.items():
            if existing_id in self.published_objects:
                # Calculate distance to existing published object
                existing_x, existing_y = state["center"]
                distance = ((center_x - existing_x) ** 2 + (center_y - existing_y) ** 2) ** 0.5
                
                # If very close to a recently published object of same class, likely same object
                if distance < self.proximity_threshold and state["class_id"] == class_id:
                    frames_since_seen = self.frame_idx - state["last_seen"]
                    if frames_since_seen <= self.max_missing_frames:
                        print(f"🔄 Tracking continuity: ID {tracker_id} matches published object {existing_id} (distance: {distance:.1f}px)")
                        # Mark as published to prevent duplicate
                        self.published_objects.add(tracker_id)
                        return False
        
        # New genuine entry - should be staged for delay
        return True
    
    def _process_staged_objects(self):
        """Process staged objects that have been continuously in zone for 1 second."""
        objects_to_publish = []
        objects_to_remove = []
        
        for tracker_id, staged_data in self.staging_objects.items():
            frames_since_first_seen = self.frame_idx - staged_data["first_seen"]
            frames_since_last_in_zone = self.frame_idx - staged_data["last_in_zone"]
            
            # Check if object left zone (not seen in current frame)
            if tracker_id not in self.objects_in_zone:
                # Object left zone - remove from staging
                print(f"❌ Unstaged: Object (ID:{tracker_id}) left zone after {frames_since_first_seen} frames")
                objects_to_remove.append(tracker_id)
                continue
            
            # Check if object has been continuously in zone for required frames
            if (frames_since_first_seen >= self.required_stability_frames and 
                frames_since_last_in_zone == 0):  # Still in zone this frame
                objects_to_publish.append(tracker_id)
        
        # Remove objects that left the zone
        for tracker_id in objects_to_remove:
            del self.staging_objects[tracker_id]
        
        # Publish stable objects
        for tracker_id in objects_to_publish:
            staged_data = self.staging_objects[tracker_id]
            data = staged_data["data"]
            
            # Mark as published
            self.published_objects.add(tracker_id)
            
            # Publish MQTT event
            success = self.mqtt.publish_zone_event(
                data["tracker_id"], data["class_id"], data["class_name"], 
                data["confidence"], data["fps"]
            )
            
            if success:
                print(f"🎯 Delayed Entry Event: {data['class_name']} (ID:{tracker_id}) - Confidence: {data['confidence']:.2f} [Stable for {frames_since_first_seen} frames]")
            
                # Update web dashboard
                if self.enable_web and self.web_dashboard:
                    direction = "IN"
                    x1, y1, x2, y2 = map(int, data["bbox"])
                    if hasattr(self, '_last_frame'):
                        cropped_img = self._last_frame[y1:y2, x1:x2]
                        self.web_dashboard.add_detection(
                            data["class_name"], data["confidence"], direction, cropped_img
                        )
            
            # Remove from staging
            del self.staging_objects[tracker_id]
    
    def _cleanup_staging_objects(self):
        """Clean up staging objects that are no longer in zone."""
        # This is now handled in _process_staged_objects to avoid double processing
        # Only clean up objects that have been staging for an excessive amount of time
        staging_to_remove = []
        
        for tracker_id, staged_data in self.staging_objects.items():
            frames_since_first_seen = self.frame_idx - staged_data["first_seen"]
            frames_since_last_in_zone = self.frame_idx - staged_data["last_in_zone"]
            
            # Remove objects that haven't been in zone for too long (safety cleanup)
            if frames_since_last_in_zone > self.max_missing_frames:
                staging_to_remove.append(tracker_id)
        
        for tracker_id in staging_to_remove:
            print(f"🧹 Cleanup staging: Object (ID:{tracker_id}) absent from zone too long")
            del self.staging_objects[tracker_id]
    
    def _add_overlay(self, frame, inference_time, ran_inference, detections):
        """Add performance overlay to frame."""
        try:
            fps = self.perf_monitor.get_fps()
            
            # Performance text
            if ran_inference and inference_time > 0:
                avg_inference = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
                text = f"FPS: {fps:.1f} | Inf: {inference_time*1000:.0f}ms (avg: {avg_inference*1000:.0f}ms)"
            else:
                text = f"FPS: {fps:.1f} | Skipped frame"
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
            
            # Object count and tracking info
            num_objects = len(detections) if detections is not None else 0
            cv2.putText(frame, f"Objects: {num_objects} | In Zone: {len(self.objects_in_zone)} | Staging: {len(self.staging_objects)} | Published: {len(self.published_objects)}", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Cam: {self.camera_index} | Mode: {self.mode}", 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        except Exception as e:
            print(f"Overlay error: {e}")
    
    def run(self):
        """Main optimized loop."""
        self.running = True
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    self.failed_reads += 1
                    if self.failed_reads >= CameraConfig.MAX_FAILED_READS:
                        print("🔄 Reconnecting camera...")
                        self.camera.release()
                        time.sleep(1)
                        self.camera.initialize()
                        self.failed_reads = 0
                    time.sleep(0.1)
                    continue
                
                self.failed_reads = 0
                self.frame_idx += 1
                
                # Process frame
                try:
                    frame = self.process_frame(frame)
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    traceback.print_exc()
                    continue
                
                # Display (optimized)
                if self.enable_display or self.enable_web:
                    try:
                        # OPTIMIZED: No scaling by default
                        if DisplayConfig.SCALE_FACTOR != 1.0:
                            display_frame = cv2.resize(
                                frame, None,
                                fx=DisplayConfig.SCALE_FACTOR,
                                fy=DisplayConfig.SCALE_FACTOR,
                                interpolation=cv2.INTER_LINEAR
                            )
                        else:
                            display_frame = frame
                        
                        # Update web dashboard
                        if self.enable_web and self.web_dashboard:
                            self.web_dashboard.update_frame(display_frame)
                        
                        # Show OpenCV window only if display enabled and web disabled
                        if self.enable_display and not self.enable_web:
                            window_name = DisplayConfig.WINDOW_NAME_TEMPLATE.format(
                                mode_name=self.config['name']
                            )
                            cv2.imshow(window_name, display_frame)
                        
                        # Always process key events
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("\n⚠️  'q' pressed - exiting")
                            break
                            
                    except Exception as e:
                        print(f"Display error: {e}")
                        break
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n💥 Fatal error: {e}")
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown."""
        print("\n🛑 Shutting down...")
        self.running = False
        
        try:
            self.camera.release()
        except:
            pass
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        self.mqtt.disconnect()
        
        if self.enable_web and self.web_dashboard:
            try:
                self.web_dashboard.stop()
            except:
                pass
        
        self._print_summary()
    
    def _print_summary(self):
        """Print performance summary."""
        width, height = self.camera.get_resolution()
        summary = self.perf_monitor.get_summary()
        
        avg_inference = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        
        print("=" * 60)
        print("📊 Performance Summary:")
        print(f"   Mode: {self.config['name']}")
        print(f"   Average FPS: {summary['fps']:.2f}")
        print(f"   Frames processed: {summary['frames_processed']}")
        print(f"   Average inference: {avg_inference*1000:.1f}ms")
        print(f"   Resolution: {width}x{height}")
        print(f"   Objects tracked: {len(self.objects_in_zone)}")
        print(f"   Published objects: {len(self.published_objects)}")
        print(f"   Object states: {len(self.object_states)}")
        print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Optimized YOLO Zone Detection for Intel CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using camera ID from config file
  python -m src.main --camera-id cam-1

  # Legacy mode using camera index  
  python -m src.main --camera 1

  # With web dashboard enabled
  python -m src.main --camera-id cam-2 --web

  # Maximum performance (no display, no web)
  python -m src.main --camera-id cam-1 --no-display
        """
    )
    
    parser.add_argument("--mode", type=str, default="custom",
                       choices=["custom"],
                       help="Performance mode (default: custom)")
    
    parser.add_argument("--camera", type=int, default=None,
                       help="Camera index (legacy mode, use --camera-id instead)")
    
    parser.add_argument("--camera-id", type=str, default=None,
                       help="Camera ID from config file (e.g., cam-1, cam-2)")
    
    parser.add_argument("--mqtt-broker", type=str, default=None,
                       help="MQTT broker address (optional, loaded from config)")
    
    parser.add_argument("--mqtt-port", type=int, default=None,
                       help="MQTT broker port (optional, loaded from config)")
    
    parser.add_argument("--web", action="store_true",
                       help="Enable web dashboard (DISABLED by default, reduces FPS by 20-30%%)")
    
    parser.add_argument("--web-port", type=int, default=5000,
                       help="Web dashboard port (default: 5000)")
    
    parser.add_argument("--no-display", action="store_true",
                       help="Disable OpenCV display window (increases FPS)")
    
    parser.add_argument("--list-modes", action="store_true",
                       help="List all performance modes and exit")
    
    parser.add_argument("--list-cameras", action="store_true",
                       help="List all enabled cameras from config and exit")
    
    args = parser.parse_args()
    
    if args.list_modes:
        PerformanceMode.list_modes()
        return
    
    if args.list_cameras:
        print("📋 Enabled Cameras from Configuration:")
        print("=" * 50)
        enabled_cameras = CameraConfigManager.get_enabled_cameras()
        if enabled_cameras:
            for camera_id in enabled_cameras:
                camera_config = CameraConfigManager.get_camera_config(camera_id)
                print(f"🎥 {camera_id}:")
                print(f"   Source: {camera_config.get('path', 'N/A')}")
                print(f"   Type: {camera_config.get('source_type', 'N/A')}")
                print(f"   Resolution: {camera_config.get('stream_resolution', 'N/A')}")
                print(f"   FPS: {camera_config.get('stream_fps', 'N/A')}")
                print()
        else:
            print("⚠️  No enabled cameras found in configuration")
        return
    
    # Validate camera arguments
    if args.camera_id and args.camera:
        print("⚠️  Cannot specify both --camera-id and --camera. Use --camera-id for config-based cameras.")
        return
    
    if not args.camera_id and not args.camera:
        print("⚠️  Must specify either --camera-id (from config) or --camera (index). Use --list-cameras to see available options.")
        return
    
    # Performance warning
    if args.web:
        print("⚠️  Web dashboard enabled - expect 20-30% FPS reduction")
        print("   Use default (no --web flag) for maximum performance\n")
    
    # Create and run application
    app = ZoneDetectionApp(
        mode=args.mode,
        camera_index=args.camera,
        camera_id=args.camera_id,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        enable_web=args.web,  # Disabled by default
        web_port=args.web_port,
        enable_display=not args.no_display
    )
    
    if app.initialize():
        app.run()


if __name__ == "__main__":
    main()