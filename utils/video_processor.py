"""Video processing utilities for fall detection analysis."""

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from fall_prediction_app.services.pose_analyzer import PoseAnalyzer
from fall_prediction_app.services.object_detector import ObjectDetector


class VideoProcessor:
    """Handles video processing for fall detection analysis."""
    
    def __init__(self, pose_complexity: int = 1, show_preview: bool = False):
        """Initialize video processor.
        
        Args:
            pose_complexity: MediaPipe pose complexity (0=lite, 1=full, 2=heavy)
            show_preview: Whether to show preview during processing
        """
        self.pose_complexity = pose_complexity
        self.show_preview = show_preview
        
        # Initialize services
        self.object_detector = ObjectDetector()
        self.pose_analyzer = PoseAnalyzer(pose_complexity)
        
        # Processing state
        self.metrics_buffer = deque(maxlen=300)  # ~5 minutes at 1 Hz
        self.last_buffer_time = 0.0
    
    def process_video(self, input_path: Path, output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Process a video file and extract fall detection metrics.
        
        Args:
            input_path: Path to input video file
            output_path: Optional output path for metrics JSON
            
        Returns:
            List of frame metrics
        """
        if output_path is None:
            output_path = self._ensure_output_path(input_path)
        
        # Initialize video capture
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file '{input_path}'.")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        
        metrics = []
        frame_idx = 0
        prev_time = time.perf_counter()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # EOF
                
                frame_idx += 1
                current_time = time.perf_counter()
                
                # Process frame
                frame_metrics = self._process_frame(frame, fps_video)
                frame_metrics['frame_idx'] = frame_idx
                frame_metrics['timestamp'] = current_time - prev_time
                
                metrics.append(frame_metrics)
                
                # Buffer metrics for time-series analysis
                if current_time - self.last_buffer_time >= 1.0:
                    self.metrics_buffer.append(frame_metrics)
                    self.last_buffer_time = current_time
                
                # Show preview if requested
                if self.show_preview:
                    self._show_preview(frame, frame_metrics, frame_idx, total_frames)
                
                # Progress indicator
                if frame_idx % 30 == 0:
                    print(f"Processed frame {frame_idx}/{total_frames or '?'}")
                    
        finally:
            cap.release()
            if self.show_preview:
                cv2.destroyAllWindows()
        
        # Save metrics
        self._save_metrics(metrics, output_path)
        
        return metrics
    
    def _process_frame(self, frame: np.ndarray, fps: float) -> Dict[str, Any]:
        """Process a single frame and extract metrics.
        
        Args:
            frame: Input frame
            fps: Video FPS
            
        Returns:
            Dictionary containing frame metrics
        """
        # Detect objects
        person_boxes, other_objects = self.object_detector.detect_objects(frame)
        
        # Initialize metrics
        metrics = {
            "trunk_angle": None,
            "nsar": None,
            "theta_u": None,
            "theta_d": None,
            "fall_detected": False,
            "person_count": len(person_boxes)
        }
        
        # Process first person (primary person)
        if person_boxes:
            pose_metrics = self.pose_analyzer.analyze_person(frame, person_boxes[0], fps)
            
            # Extract metrics
            metrics.update({
                "trunk_angle": pose_metrics.trunk_angle,
                "nsar": pose_metrics.nsar,
                "theta_u": pose_metrics.theta_u,
                "theta_d": pose_metrics.theta_d,
                "fall_detected": pose_metrics.status == "Fall Detected"
            })
        
        return metrics
    
    def _show_preview(self, frame: np.ndarray, metrics: Dict[str, Any], 
                     frame_idx: int, total_frames: Optional[int]) -> None:
        """Show preview of processing.
        
        Args:
            frame: Frame to display
            metrics: Frame metrics
            frame_idx: Current frame index
            total_frames: Total number of frames
        """
        # Create annotated frame
        annotated = frame.copy()
        
        # Draw metrics on frame
        base_x, base_y = 10, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        
        # Frame info
        frame_info = f"Frame: {frame_idx}"
        if total_frames:
            frame_info += f"/{total_frames}"
        cv2.putText(annotated, frame_info, (base_x, base_y), font, scale, color, thickness)
        base_y += 25
        
        # Metrics
        if metrics["trunk_angle"] is not None:
            cv2.putText(annotated, f"Trunk: {metrics['trunk_angle']:.1f}°", 
                       (base_x, base_y), font, scale, color, thickness)
            base_y += 25
        
        if metrics["nsar"] is not None:
            cv2.putText(annotated, f"NSAR: {metrics['nsar']:.2f}", 
                       (base_x, base_y), font, scale, color, thickness)
            base_y += 25
        
        if metrics["fall_detected"]:
            cv2.putText(annotated, "FALL DETECTED!", (base_x, base_y), 
                       font, scale, (0, 0, 255), thickness)
        
        # Display
        cv2.imshow("Video Processing Preview", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt
    
    def _ensure_output_path(self, input_path: Path) -> Path:
        """Ensure output path exists.
        
        Args:
            input_path: Input video path
            
        Returns:
            Output path for metrics
        """
        out_dir = Path("metrics_json")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{input_path.stem}_metrics.json"
    
    def _save_metrics(self, metrics: List[Dict[str, Any]], output_path: Path) -> None:
        """Save metrics to JSON file.
        
        Args:
            metrics: List of frame metrics
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {output_path}")
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get buffered metrics history.
        
        Returns:
            List of buffered metrics
        """
        return list(self.metrics_buffer)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.pose_analyzer.close()


def main():
    """Command-line interface for video processing."""
    parser = argparse.ArgumentParser(
        description="Offline fall-detection metrics extraction (YOLOv8 + MediaPipe)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to an input video file.")
    parser.add_argument(
        "--output",
        help="Destination JSON file; defaults to metrics_json/<input-stem>_metrics.json",
    )
    parser.add_argument(
        "--pose-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe Pose complexity (0=lite, 1=full, 2=heavy).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display an annotated preview while processing (slower).",
    )
    
    args = parser.parse_args()
    
    # Process video
    processor = VideoProcessor(args.pose_complexity, args.show)
    try:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else None
        
        processor.process_video(input_path, output_path)
    finally:
        processor.cleanup()


if __name__ == "__main__":
    main() 