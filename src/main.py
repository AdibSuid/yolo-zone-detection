"""Optimized main application for Intel CPU retail detection."""
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

from .config import PerformanceMode, ZoneConfig, DisplayConfig, CameraConfig
from .camera import CameraManager
from .detector import YOLODetector
from .mqtt_client import MQTTPublisher
from .performance import PerformanceMonitor

# Import web dashboard only if needed
try:
    from .web_dashboard import WebDashboard
    WEB_DASHBOARD_AVAILABLE = True
except ImportError:
    WEB_DASHBOARD_AVAILABLE = False
    print("⚠️  Web dashboard dependencies not available")


class ZoneDetectionApp:
    """Optimized YOLO zone detection for Intel CPU."""
    
    def __init__(self, mode, camera_index, mqtt_broker, mqtt_port, 
                 enable_web=False, web_port=5000, enable_display=True):
        """Initialize optimized application.
        
        Args:
            mode: Performance mode name
            camera_index: Camera device index
            mqtt_broker: MQTT broker address
            mqtt_port: MQTT broker port
            enable_web: Enable web dashboard (default False for performance)
            web_port: Web dashboard port
            enable_display: Enable OpenCV display window
        """
        self.config = PerformanceMode.get_mode(mode)
        self.mode = mode
        self.camera_index = camera_index
        self.enable_display = enable_display
        
        # Initialize components
        self.camera = CameraManager(camera_index, self.config['resolution'])
        
        iou_threshold = self.config.get('iou_threshold', 0.5)
        self.detector = YOLODetector(
            self.config['model'], 
            self.config['conf_threshold'],
            iou_threshold
        )
        
        self.mqtt = MQTTPublisher(mqtt_broker, mqtt_port, mode)
        self.perf_monitor = PerformanceMonitor()
        
        # Web dashboard (optional, disabled by default)
        self.enable_web = enable_web
        self.web_port = web_port
        self.web_dashboard = None
        if enable_web:
            if WEB_DASHBOARD_AVAILABLE:
                self.web_dashboard = WebDashboard(
                    camera_id="001",
                    camera_name="Ringo",
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
        self.objects_in_zone = set()
        
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
        """Publish MQTT events for detections in zone."""
        if not any(zone_mask):
            return
        
        fps = self.perf_monitor.get_fps()
        
        for det_idx, inside_zone in enumerate(zone_mask):
            if inside_zone:
                tracker_id = int(detections.tracker_id[det_idx])
                class_id = int(detections.class_id[det_idx])
                confidence = float(detections.confidence[det_idx])
                class_name = self.detector.get_class_name(class_id)
                
                self.mqtt.publish_zone_event(
                    tracker_id, class_id, class_name, confidence, fps
                )
                
                # Update web dashboard
                if self.enable_web and self.web_dashboard:
                    if tracker_id not in self.objects_in_zone:
                        direction = "IN"
                        bbox = detections.xyxy[det_idx]
                        x1, y1, x2, y2 = map(int, bbox)
                        if hasattr(self, '_last_frame'):
                            cropped_img = self._last_frame[y1:y2, x1:x2]
                            self.web_dashboard.add_detection(
                                class_name, confidence, direction, cropped_img
                            )
                        self.objects_in_zone.add(tracker_id)
    
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
            
            # Object count
            num_objects = len(detections) if detections is not None else 0
            cv2.putText(frame, f"Objects: {num_objects} | Cam: {self.camera_index} | Mode: {self.mode}", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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
        print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Optimized YOLO Zone Detection for Intel CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Custom YOLOv8 model (web dashboard OFF by default)
  python -m src.main --mode custom --camera 1

  # With web dashboard enabled
  python -m src.main --mode custom --camera 1 --web

  # Maximum performance (no display, no web)
  python -m src.main --mode custom --camera 1 --no-display

  # List all modes
  python -m src.main --list-modes
        """
    )
    
    parser.add_argument("--mode", type=str, default="custom",
                       choices=["custom", "retail_optimized", "ultra_fast", "maximum_fps", "balanced", "high_accuracy"],
                       help="Performance mode (default: custom)")
    
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index (default: 0)")
    
    parser.add_argument("--mqtt-broker", type=str, default="localhost",
                       help="MQTT broker address (default: localhost)")
    
    parser.add_argument("--mqtt-port", type=int, default=1883,
                       help="MQTT broker port (default: 1883)")
    
    parser.add_argument("--web", action="store_true",
                       help="Enable web dashboard (DISABLED by default, reduces FPS by 20-30%%)")
    
    parser.add_argument("--web-port", type=int, default=5000,
                       help="Web dashboard port (default: 5000)")
    
    parser.add_argument("--no-display", action="store_true",
                       help="Disable OpenCV display window (increases FPS)")
    
    parser.add_argument("--list-modes", action="store_true",
                       help="List all performance modes and exit")
    
    args = parser.parse_args()
    
    if args.list_modes:
        PerformanceMode.list_modes()
        return
    
    # Performance warning
    if args.web:
        print("⚠️  Web dashboard enabled - expect 20-30% FPS reduction")
        print("   Use default (no --web flag) for maximum performance\n")
    
    # Create and run application
    app = ZoneDetectionApp(
        mode=args.mode,
        camera_index=args.camera,
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