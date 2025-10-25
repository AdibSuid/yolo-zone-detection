"""Main application entry point for YOLO zone detection system."""
import sys
import cv2
import time
import traceback
import argparse
import supervision as sv

from .config import PerformanceMode, ZoneConfig, DisplayConfig, CameraConfig
from .camera import CameraManager
from .detector import YOLODetector
from .mqtt_client import MQTTPublisher
from .performance import PerformanceMonitor


class ZoneDetectionApp:
    """Main application for YOLO zone detection with MQTT."""
    
    def __init__(self, mode, camera_index, mqtt_broker, mqtt_port):
        """Initialize application.
        
        Args:
            mode: Performance mode name
            camera_index: Camera device index
            mqtt_broker: MQTT broker address
            mqtt_port: MQTT broker port
        """
        self.config = PerformanceMode.get_mode(mode)
        self.mode = mode
        self.camera_index = camera_index
        
        # Initialize components
        self.camera = CameraManager(camera_index, self.config['resolution'])
        self.detector = YOLODetector(self.config['model'], self.config['conf_threshold'])
        self.mqtt = MQTTPublisher(mqtt_broker, mqtt_port, mode)
        self.perf_monitor = PerformanceMonitor()
        
        # Supervision components
        self.box_annotator = sv.BoxAnnotator(thickness=self.config['annotation_thickness'])
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
    
    def initialize(self):
        """Initialize all components.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"🚀 Starting YOLO Zone Detection")
        print(f"   Mode: {self.config['name']}")
        print(f"   Model: {self.config['model']}")
        print(f"   Resolution: {self.config['resolution'][0]}x{self.config['resolution'][1]}")
        print(f"   Camera: Index {self.camera_index}")
        print("=" * 60)
        
        # Initialize camera
        if not self.camera.initialize():
            return False
        
        # Initialize zone based on actual camera resolution
        width, height = self.camera.get_resolution()
        box_polygon = ZoneConfig.create_box_polygon(width, height)
        self.polygon_zone = sv.PolygonZone(polygon=box_polygon)
        self.zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.polygon_zone,
            color=sv.Color.RED,
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )
        
        # Load YOLO model
        if not self.detector.load_model():
            return False
        
        # Connect to MQTT
        self.mqtt.connect()
        
        print("🎬 Starting inference... Press 'q' to quit")
        print("=" * 60)
        return True
    
    def process_frame(self, frame):
        """Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Annotated frame
        """
        # Determine if we should run inference
        should_run_inference = (
            self.config['frame_skip'] == 1 or 
            self.frame_idx % self.config['frame_skip'] == 0
        )
        
        inference_time = 0
        
        if should_run_inference:
            self.perf_monitor.tick()
            
            # Run detection
            inference_start = time.time()
            detections = self.detector.detect(frame)
            inference_time = time.time() - inference_start
            
            if detections is not None and len(detections) > 0:
                # Check zone
                try:
                    detections_in_zone = self.polygon_zone.trigger(detections=detections)
                    
                    # Publish MQTT events for objects in zone
                    self._publish_zone_events(detections, detections_in_zone)
                    
                except Exception as e:
                    print(f"Zone detection error: {e}")
                    traceback.print_exc()
        
        # Get cached detections for annotation (smooth display on frame skip)
        detections, labels = self.detector.get_cached_detections()
        
        # Annotate frame
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
        """Publish MQTT events for detections in zone.
        
        Args:
            detections: Supervision Detections object
            zone_mask: Boolean mask for objects in zone
        """
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
    
    def _add_overlay(self, frame, inference_time, ran_inference, detections):
        """Add performance overlay to frame.
        
        Args:
            frame: Frame to annotate
            inference_time: Time taken for inference
            ran_inference: Whether inference was run this frame
            detections: Current detections
        """
        try:
            fps = self.perf_monitor.get_fps()
            
            # FPS and inference time
            if ran_inference and inference_time > 0:
                text = f"FPS: {fps:.1f} | Inference: {inference_time*1000:.0f}ms | Mode: {self.mode}"
            else:
                text = f"FPS: {fps:.1f} | Skipped frame | Mode: {self.mode}"
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
            
            # Object count
            num_objects = len(detections) if detections is not None else 0
            cv2.putText(frame, f"Objects: {num_objects} | Camera: {self.camera_index}", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        except Exception as e:
            print(f"Overlay error: {e}")
    
    def run(self):
        """Main application loop."""
        self.running = True
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    self.failed_reads += 1
                    if self.failed_reads >= CameraConfig.MAX_FAILED_READS:
                        print(f"❌ Too many failed reads ({self.failed_reads}). Exiting.")
                        break
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
                
                # Display
                try:
                    display_frame = cv2.resize(
                        frame, None,
                        fx=DisplayConfig.SCALE_FACTOR,
                        fy=DisplayConfig.SCALE_FACTOR,
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    window_name = DisplayConfig.WINDOW_NAME_TEMPLATE.format(
                        mode_name=self.config['name']
                    )
                    cv2.imshow(window_name, display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\n⚠️  'q' pressed - exiting")
                        break
                        
                except cv2.error as e:
                    print(f"OpenCV error: {e}")
                    traceback.print_exc()
                    break
                except Exception as e:
                    print(f"Display error: {e}")
                    traceback.print_exc()
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n💥 Fatal error in main loop: {e}")
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of all components."""
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
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print performance summary."""
        width, height = self.camera.get_resolution()
        summary = self.perf_monitor.get_summary()
        
        print("=" * 60)
        print("📊 Performance Summary:")
        print(f"   Mode: {self.config['name']}")
        print(f"   Average FPS: {summary['fps']:.2f}")
        print(f"   Frames processed: {summary['frames_processed']}")
        print(f"   Resolution: {width}x{height}")
        print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YOLO Zone Detection with MQTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --mode balanced
  python -m src.main --mode ultra_fast --camera 1
  python -m src.main --list-modes
  python -m src.main --mode maximum_fps --mqtt-broker 192.168.1.100
        """
    )
    
    parser.add_argument("--mode", type=str, default="balanced",
                       choices=["ultra_fast", "maximum_fps", "balanced", "high_accuracy"],
                       help="Performance mode (default: balanced)")
    
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index (0=built-in, 1=USB, etc.)")
    
    parser.add_argument("--mqtt-broker", type=str, default="localhost",
                       help="MQTT broker address (default: localhost)")
    
    parser.add_argument("--mqtt-port", type=int, default=1883,
                       help="MQTT broker port (default: 1883)")
    
    parser.add_argument("--list-modes", action="store_true",
                       help="List all performance modes and exit")
    
    args = parser.parse_args()
    
    if args.list_modes:
        PerformanceMode.list_modes()
        return
    
    # Create and run application
    app = ZoneDetectionApp(
        mode=args.mode,
        camera_index=args.camera,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port
    )
    
    if app.initialize():
        app.run()


if __name__ == "__main__":
    main()
