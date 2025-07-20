"""Refactored Webcam Person Detection (YOLOv8)

Live webcam feed with real-time person detection powered by the lightweight
YOLOv8n model from the *ultralytics* package. A bounding box and confidence
score are drawn around each detected person. Press the 'q' key to quit.

This refactored version follows SOLID principles with separate service classes
for different responsibilities.
"""

import argparse
import time
from dotenv import load_dotenv

from services.camera_service import CameraService
from services.object_detector import ObjectDetector
from services.pose_analyzer import PoseAnalyzer
from services.alert_service import AlertService
from services.display_service import DisplayService
from services.statistics_service import StatisticsService

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for optional multi-camera processing.

    The feature is **disabled by default** and must be explicitly enabled with
    the ``--multi-view`` flag.  A comma-separated ``--sources`` list can be
    provided to point to additional camera indices or video URLs.
    """
    parser = argparse.ArgumentParser(
        description="Webcam / multi-camera fall detection with YOLOv8 + MediaPipe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--sources",
        type=str,
        default="0",
        help=(
            "Comma-separated list of video sources.  Each token may be an integer "
            "(webcam index) or a path / URL (e.g. '0,1' or 'rtsp://<ip>/stream')."
        ),
    )

    parser.add_argument(
        "--multi-view",
        action="store_true",
        help="Enable multi-camera & multi-view fusion (off by default).",
    )

    parser.add_argument(
        "--pose-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe Pose parameter: 0=lite (fast), 1=full (accurate), 2=heavy (very accurate).",
    )

    return parser.parse_args()


class FallDetectionApp:
    """Main application class for fall detection."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the fall detection application.
        
        Args:
            args: Parsed command line arguments
        """
        # Initialize all services
        self.camera_service = CameraService(args.sources, args.multi_view)
        self.object_detector = ObjectDetector()
        self.pose_analyzer = PoseAnalyzer(args.pose_complexity)
        self.alert_service = AlertService()
        self.display_service = DisplayService()
        self.statistics_service = StatisticsService()
        
        # Application state
        self.running = True
    
    def run(self) -> None:
        """Run the main application loop."""
        try:
            while self.running:
                self._process_frame()
                
                # Check for exit condition
                if self.display_service.check_exit():
                    break
                    
        finally:
            self._cleanup()
    
    def _process_frame(self) -> None:
        """Process a single frame through the pipeline."""
        # Read frame from camera
        success, frame = self.camera_service.read_frame()
        if not success:
            return
        
        current_time = time.perf_counter()
        fps = self.display_service.update_fps(current_time)
        
        # Detect objects and separate person detections
        person_boxes, other_objects = self.object_detector.detect_objects(frame)
        
        # Draw non-person objects
        self.object_detector.draw_objects(frame, other_objects)
        
        # Initialize frame metrics
        frame_metrics = {
            "trunk_angle": None,
            "nsar": None,
            "theta_u": None,
            "theta_d": None,
            "status": "Normal",
            "abnormal_sec": 0.0
        }
        
        # Process person detections for pose analysis
        if person_boxes:
            self.object_detector.draw_person_boxes(frame, person_boxes)
            
            # Analyze pose for the first person (primary person)
            if person_boxes:
                pose_metrics = self.pose_analyzer.analyze_person(
                    frame, person_boxes[0], fps
                )
                
                # Update frame metrics with pose analysis results
                frame_metrics.update({
                    "trunk_angle": pose_metrics.trunk_angle,
                    "nsar": pose_metrics.nsar,
                    "theta_u": pose_metrics.theta_u,
                    "theta_d": pose_metrics.theta_d,
                    "status": pose_metrics.status,
                    "abnormal_sec": pose_metrics.abnormal_sec
                })
        
        # Update statistics
        self.statistics_service.update_frame_count(len(person_boxes) > 0)
        self.statistics_service.update_latency_tracking(frame_metrics["status"], current_time)
        
        # Check for alerts
        foundry_response = self.alert_service.check_and_send_alerts(
            frame_metrics["status"], frame_metrics
        )
        
        # Buffer metrics for time-series analysis
        self.display_service.buffer_metrics(current_time, frame_metrics)
        
        # Get statistics for display
        stats = self.statistics_service.get_statistics()
        
        # Draw UI elements
        self.display_service.draw_hud(frame, frame_metrics, stats)
        self.display_service.draw_fall_alerts(frame, frame_metrics)
        self.display_service.draw_foundry_response(frame, foundry_response)
        
        # Display the frame
        self.display_service.display_frame(frame)
    
    def _cleanup(self) -> None:
        """Clean up all resources."""
        self.camera_service.release()
        self.pose_analyzer.close()
        self.display_service.cleanup()


def main() -> None:
    """Start the fall detection application."""
    args = parse_args()
    app = FallDetectionApp(args)
    app.run()


if __name__ == "__main__":
    main() 