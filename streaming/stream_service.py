"""Streaming service for Flask web applications."""

import time
import math
from typing import Dict, Any, Generator, List, Deque
from collections import deque
import cv2
import numpy as np

from fall_prediction_app.services.camera_service import CameraService
from fall_prediction_app.services.object_detector import ObjectDetector
from fall_prediction_app.services.pose_analyzer import PoseAnalyzer
from fall_prediction_app.services.alert_service import AlertService


class StreamService:
    """Handles streaming functionality for Flask applications."""
    
    def __init__(self, pose_complexity: int = 1):
        """Initialize stream service.
        
        Args:
            pose_complexity: MediaPipe pose complexity
        """
        self.pose_complexity = pose_complexity
        
        # Initialize services
        self.camera_service = None
        self.object_detector = None
        self.pose_analyzer = None
        self.alert_service = None
        
        # Runtime state
        self.last_pose_landmarks = None
        self.missing_pose_count = 0
        
        # Smoothing buffers
        self.trunk_angle_hist = deque(maxlen=8)
        self.nsar_hist = deque(maxlen=8)
        
        # NSAR baseline
        self.nsar_baseline_samples = deque(maxlen=30)
        self.nsar_baseline = None
        
        # Plumb line fall detection
        self.plumb_fall_counter = 0
        self.plumb_fall_threshold_frames = 60
        
        # Status tracking
        self.last_status_text = "Normal"
        self.foundry_response_global = None
        
        # Metrics buffer
        self.metrics_buffer = deque(maxlen=300)  # ~5 minutes at 1 Hz
        self.last_buffer_time = 0.0
        
        # Latest statistics
        self._latest_stats = {
            "fps": None,
            "trunk_angle": None,
            "nsar": None,
            "theta_u": None,
            "theta_d": None,
            "fall_detected": False,
        }
        
        # Initialization flag
        self._stream_inited = False
    
    def initialize_streaming(self) -> None:
        """Initialize streaming resources."""
        if self._stream_inited:
            return
        
        self.camera_service = CameraService()
        self.object_detector = ObjectDetector()
        self.pose_analyzer = PoseAnalyzer(self.pose_complexity)
        self.alert_service = AlertService()
        
        self._stream_inited = True
    
    def get_latest_stats(self) -> Dict[str, Any]:
        """Get the most recently computed frame statistics.
        
        Returns:
            Dictionary with latest statistics
        """
        return self._latest_stats.copy()
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get buffered metrics history.
        
        Returns:
            List of buffered metrics
        """
        return list(self.metrics_buffer)
    
    def process_frame(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Process a single frame and return annotated version.
        
        Args:
            frame: Input frame
            fps: Current FPS
            
        Returns:
            Annotated frame
        """
        if not self._stream_inited:
            self.initialize_streaming()
        
        # Initialize frame metrics
        trunk_angle_value = None
        nsar_value = None
        theta_u_value = None
        theta_d_value = None
        frame_fall_detected = False
        
        # Detect objects
        person_boxes, other_objects = self.object_detector.detect_objects(frame)
        
        # Draw non-person objects
        self.object_detector.draw_objects(frame, other_objects)
        
        # Process person detections
        for person_idx, box in enumerate(person_boxes, start=1):
            # Draw person boxes
            self.object_detector.draw_person_boxes(frame, [box])
            
            # Analyze pose for first person
            if person_idx == 1:
                pose_metrics = self.pose_analyzer.analyze_person(frame, box, fps)
                
                # Extract metrics
                trunk_angle_value = pose_metrics.trunk_angle
                nsar_value = pose_metrics.nsar
                theta_u_value = pose_metrics.theta_u
                theta_d_value = pose_metrics.theta_d
                frame_fall_detected = pose_metrics.status == "Fall Detected"
        
        # Update latest stats
        self._latest_stats.update({
            "fps": fps,
            "trunk_angle": trunk_angle_value,
            "nsar": nsar_value,
            "theta_u": theta_u_value,
            "theta_d": theta_d_value,
            "fall_detected": frame_fall_detected,
        })
        
        # Check for alerts
        if frame_fall_detected and self.last_status_text != "Fall Detected":
            metrics_dict = {
                "trunk_angle": trunk_angle_value,
                "nsar": nsar_value,
                "theta_d": theta_d_value
            }
            self.foundry_response_global = self.alert_service.check_and_send_alerts(
                "Fall Detected", metrics_dict
            )
        
        self.last_status_text = "Fall Detected" if frame_fall_detected else "Normal"
        
        # Buffer metrics
        current_time = time.perf_counter()
        if current_time - self.last_buffer_time >= 1.0:
            self.metrics_buffer.append({
                "timestamp": current_time,
                "trunk_angle": trunk_angle_value,
                "nsar": nsar_value,
                "theta_u": theta_u_value,
                "theta_d": theta_d_value,
                "fall_detected": frame_fall_detected,
            })
            self.last_buffer_time = current_time
        
        return frame
    
    def stream_frames(self) -> Generator[bytes, None, None]:
        """Generate streaming frames for Flask.
        
        Yields:
            JPEG-encoded frame bytes
        """
        if not self._stream_inited:
            self.initialize_streaming()
        
        prev_time = time.perf_counter()
        
        try:
            while True:
                # Read frame
                success, frame = self.camera_service.read_frame()
                if not success:
                    break
                
                # Calculate FPS
                current_time = time.perf_counter()
                fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 30.0
                prev_time = current_time
                
                # Process frame
                annotated_frame = self.process_frame(frame, fps)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                
                # Yield frame data
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up streaming resources."""
        if self.camera_service:
            self.camera_service.release()
        if self.pose_analyzer:
            self.pose_analyzer.close()
        self._stream_inited = False


# Global stream service instance for Flask
_stream_service = None


def get_stream_service() -> StreamService:
    """Get or create global stream service instance.
    
    Returns:
        StreamService instance
    """
    global _stream_service
    if _stream_service is None:
        _stream_service = StreamService()
    return _stream_service


def stream_frames_flask() -> Generator[bytes, None, None]:
    """Flask-compatible frame stream generator.
    
    Yields:
        JPEG-encoded frame bytes
    """
    service = get_stream_service()
    yield from service.stream_frames() 