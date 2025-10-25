"""Web dashboard for live detection visualization."""
import cv2
import base64
import threading
import time
from datetime import datetime
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from flask_cors import CORS
import numpy as np


class WebDashboard:
    """Web dashboard for displaying live detection results."""
    
    def __init__(self, camera_id="001", camera_name="Ringo", model_name="yolov8"):
        """Initialize web dashboard.
        
        Args:
            camera_id: Camera identifier
            camera_name: Camera name
            model_name: Model name
        """
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Camera and model info (fixed values)
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.model_name = model_name
        self.model_logic_type = "IN/OUT"
        self.model_zone = "Random"
        
        # Live data
        self.current_frame = None
        self.total_object_count = 0  # Cumulative count
        self.recent_detections = []  # List of recent detection events
        self.lock = threading.Lock()
        
        # Setup routes
        self._setup_routes()
        
        # Start background thread for sending updates
        self.running = False
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve the main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route."""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
    
    def _generate_frames(self):
        """Generate frames for video stream."""
        while self.running:
            with self.lock:
                if self.current_frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', self.current_frame)
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS max
    
    def update_frame(self, frame):
        """Update the current frame.
        
        Args:
            frame: OpenCV frame (numpy array)
        """
        with self.lock:
            self.current_frame = frame.copy()
    
    def add_detection(self, class_name, confidence, direction="IN", cropped_image=None):
        """Add a new detection event.
        
        Args:
            class_name: Detected object class
            confidence: Detection confidence (0-1)
            direction: "IN" or "OUT"
            cropped_image: Cropped image of detected object
        """
        with self.lock:
            # Increment total count
            self.total_object_count += 1
            
            # Create detection event
            timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            
            # Encode cropped image to base64 if provided
            image_base64 = None
            if cropped_image is not None:
                ret, buffer = cv2.imencode('.jpg', cropped_image)
                if ret:
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            detection = {
                'camera_id': self.camera_id,
                'camera_name': self.camera_name,
                'timestamp': timestamp,
                'object_class': class_name,
                'confidence': round(confidence * 100, 2),  # Convert to percentage
                'direction': direction,
                'model_name': self.model_name,
                'model_logic_type': self.model_logic_type,
                'model_zone': self.model_zone,
                'image': image_base64,
                'count': self.total_object_count
            }
            
            # Keep only last 50 detections
            self.recent_detections.insert(0, detection)
            if len(self.recent_detections) > 50:
                self.recent_detections.pop()
            
            # Emit to connected clients
            self.socketio.emit('new_detection', detection)
            self.socketio.emit('update_count', {'count': self.total_object_count})
    
    def get_stats(self):
        """Get current statistics.
        
        Returns:
            Dictionary with current stats
        """
        with self.lock:
            return {
                'camera_id': self.camera_id,
                'camera_name': self.camera_name,
                'total_count': self.total_object_count,
                'recent_detections': self.recent_detections[:10],  # Last 10
                'model_name': self.model_name,
                'model_logic_type': self.model_logic_type,
                'model_zone': self.model_zone
            }
    
    def start(self, host='0.0.0.0', port=5000, debug=False):
        """Start the web server.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        self.running = True
        
        # Setup SocketIO event handlers
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            # Send current stats to new client
            self.socketio.emit('initial_data', self.get_stats())
        
        print(f"🌐 Web Dashboard starting on http://{host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    
    def stop(self):
        """Stop the web server."""
        self.running = False
