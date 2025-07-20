"""Webcam Person Detection (YOLOv8) - Refactored Version

Live webcam feed with real-time person detection powered by the lightweight
YOLOv8n model from the *ultralytics* package. A bounding box and confidence
score are drawn around each detected person. Press the 'q' key to quit.

This is a refactored version that uses modular services while maintaining
the same interface as the original webcam_feed.py.
"""

import cv2
import numpy as np
import math
import time
import ctypes
from collections import deque
import argparse
from typing import List, Union
import os
from dotenv import load_dotenv

# Import refactored services
from fall_prediction_app.services.camera_service import CameraService
from fall_prediction_app.services.object_detector import ObjectDetector
from fall_prediction_app.services.pose_analyzer import PoseAnalyzer
from fall_prediction_app.services.alert_service import AlertService
from fall_prediction_app.services.display_service import DisplayService
from fall_prediction_app.services.statistics_service import StatisticsService

load_dotenv()

# Tuneable constants --------------------------------------------------------
# Margin added around detected person bounding box before pose estimation.
ROI_MARGIN = 0.25  # 25% expansion to include limbs that may fall outside box
TRUNK_ANGLE_FALL_THRESHOLD_DEG = math.degrees(0.59)  # ≈ 33.8° trunk angle threshold for fall detection


def _parse_args() -> argparse.Namespace:
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


def _parse_sources(sources: str) -> List[Union[int, str]]:
    """Convert a comma-separated source string to a list of typed sources."""
    result: List[Union[int, str]] = []
    for token in sources.split(","):
        token = token.strip()
        if not token:
            continue
        result.append(int(token) if token.isdigit() else token)
    return result or [0]


def _fuse_frames(frames: List[np.ndarray]) -> np.ndarray:
    """Placeholder multi-view fusion strategy.

    For the initial implementation, simply returns the first frame.  This stub
    exists so that future work can combine detections from multiple viewpoints
    to mitigate occlusions and varying perspectives.
    """
    return frames[0]


def main() -> None:
    """Start the video processing loop.

    By default the script runs exactly as before (single webcam at index 0).  If
    ``--multi-view`` is specified **and** more than one source is provided via
    ``--sources``, the script will open all sources and perform a rudimentary
    frame-level fusion.  The fusion logic is currently a stub that selects the
    first frame; the structure is in place for future multi-view occlusion
    handling.
    """

    args = _parse_args()

    # Initialize refactored services
    camera_service = CameraService(args.sources, args.multi_view)
    object_detector = ObjectDetector()
    pose_analyzer = PoseAnalyzer(args.pose_complexity)
    alert_service = AlertService()
    display_service = DisplayService()
    statistics_service = StatisticsService()

    # Legacy variables for compatibility (keeping same names as original)
    source_list = _parse_sources(args.sources)
    multi_view_enabled = args.multi_view and len(source_list) > 1

    # Legacy variables for tracking (maintaining original behavior)
    last_status_text = "Normal"
    foundry_response = None

    # FPS calculation helpers (legacy)
    prev_time = time.perf_counter()
    fps = 0.0

    # Legacy latency / detection statistics
    total_frames = 0
    missed_detection_frames = 0
    fall_event_latencies = []
    current_fall_start = None
    last_latency = None

    # Legacy time-series metric buffer
    metrics_buffer = deque(maxlen=60)
    last_buffer_time = prev_time

    try:
        while True:
            # ------------------------------------------------------------------
            # Frame acquisition (using refactored service)
            # ------------------------------------------------------------------
            success, frame = camera_service.read_frame()
            if not success:
                break

            # Time measurement for FPS (legacy calculation)
            current_time = time.perf_counter()
            fps = 1.0 / (current_time - prev_time) if current_time != prev_time else fps
            prev_time = current_time

            # Initialize per-frame display metrics (legacy structure)
            trunk_angle_value = None
            nsar_value = None
            theta_u_value = None
            theta_d_value = None
            status_text = "Normal"
            abnormal_sec = 0.0

            # Statistics bookkeeping (legacy)
            total_frames += 1

            # Detect objects using refactored service
            person_boxes, other_objects = object_detector.detect_objects(frame)

            # Draw non-person objects using refactored service
            object_detector.draw_objects(frame, other_objects)

            # --------- Pose & fall detection for person instances --------- #
            for person_idx, box in enumerate(person_boxes, start=1):
                # Draw person boxes using refactored service
                object_detector.draw_person_boxes(frame, [box])

                # Analyze pose using refactored service (only for first person)
                if person_idx == 1:
                    pose_metrics = pose_analyzer.analyze_person(frame, box, fps)
                    
                    # Extract metrics for legacy compatibility
                    trunk_angle_value = pose_metrics.trunk_angle
                    nsar_value = pose_metrics.nsar
                    theta_u_value = pose_metrics.theta_u
                    theta_d_value = pose_metrics.theta_d
                    status_text = pose_metrics.status
                    abnormal_sec = pose_metrics.abnormal_sec

            # Legacy fall alert display logic (maintaining original behavior)
            if status_text == "Fall Detected":
                # Show trunk-angle-based fall alert if triggered
                if trunk_angle_value and trunk_angle_value > TRUNK_ANGLE_FALL_THRESHOLD_DEG:
                    cv2.putText(frame, f"Trunk Angle > {TRUNK_ANGLE_FALL_THRESHOLD_DEG:.1f}° Fall Detected", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Show NSAR fall alert if triggered
                if nsar_value and pose_analyzer.nsar_baseline and nsar_value < pose_analyzer.nsar_baseline * 0.7:
                    cv2.putText(frame, "NSAR drops > 30% Fall Detected", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Show mid-plumb fall alert if triggered
                if pose_analyzer.plumb_fall_counter >= pose_analyzer.plumb_fall_threshold_frames:
                    cv2.putText(frame, "Mid-Plumb Angles Fall Detected", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Missed detection tracking (legacy)
            if not person_boxes:
                missed_detection_frames += 1

            # ------------------------------------------------------------------
            # Latency calculation: record the first frame where warning begins,
            # finalise when fall is confirmed.
            # ------------------------------------------------------------------
            if status_text == "Warning" and current_fall_start is None:
                current_fall_start = current_time

            if status_text == "Fall Detected" and current_fall_start is not None:
                latency = current_time - current_fall_start
                fall_event_latencies.append(latency)
                last_latency = latency
                current_fall_start = None

            # ----------- External alerts on fall -----------
            if status_text == "Fall Detected" and last_status_text != "Fall Detected":
                # Use refactored alert service
                metrics_dict = {
                    "trunk_angle": trunk_angle_value,
                    "nsar": nsar_value,
                    "theta_d": theta_d_value
                }
                foundry_response = alert_service.check_and_send_alerts(status_text, metrics_dict)

            last_status_text = status_text

            # Display the annotated frame
            # ----------- Persistent HUD (top-left) ----------- #
            base_x, base_y, line_h = 10, 20, 15
            font, scale, thick, color = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, (255, 255, 255)

            cv2.putText(frame, f"FPS: {fps:.1f}", (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
            base_y += line_h

            if trunk_angle_value is not None:
                cv2.putText(frame, f"Trunk Angle: {trunk_angle_value:.1f}", (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
                base_y += line_h

            if nsar_value is not None:
                cv2.putText(frame, f"NSAR: {nsar_value:.2f}", (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
                base_y += line_h

            if theta_u_value is not None and theta_d_value is not None:
                cv2.putText(frame, f"θu: {theta_u_value:.0f}  θd: {theta_d_value:.0f}", (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
                base_y += line_h

            cv2.putText(frame, f"Status: {status_text}", (base_x, base_y), font, scale, (0, 0, 255) if status_text != "Normal" else color, thick, cv2.LINE_AA)
            base_y += line_h

            if abnormal_sec > 0:
                cv2.putText(frame, f"Abnormal for: {abnormal_sec:.1f}s", (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
                base_y += line_h

            # Latency / detection metrics (legacy)
            miss_rate = (missed_detection_frames / total_frames * 100) if total_frames else 0.0
            cv2.putText(frame, f"Miss Rate: {miss_rate:.1f}%", (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
            base_y += line_h

            if last_latency is not None:
                cv2.putText(frame, f"Last Latency: {last_latency:.2f}s", (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
                base_y += line_h

            if fall_event_latencies:
                avg_latency = sum(fall_event_latencies) / len(fall_event_latencies)
                cv2.putText(frame, f"Avg Latency: {avg_latency:.2f}s", (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)

            # Display Azure Foundry response if available
            if foundry_response:
                base_y += line_h
                wrapped = foundry_response.split("\n")
                for line in wrapped:
                    cv2.putText(frame, line, (base_x, base_y), font, scale, (0, 255, 255), thick, cv2.LINE_AA)
                    base_y += line_h

            # ----------- Metric buffering (once per second) ----------- #
            if current_time - last_buffer_time >= 1.0:
                metrics_buffer.append({
                    "timestamp": current_time,
                    "trunk_angle": trunk_angle_value,
                    "nsar": nsar_value,
                    "theta_u": theta_u_value,
                    "theta_d": theta_d_value,
                })
                last_buffer_time = current_time
            # ---------------------------------------------------------- #

            # Display frame using refactored service
            display_service.display_frame(frame)

            # Exit on 'q' key press
            if display_service.check_exit():
                break
    finally:
        # Release all camera resources
        camera_service.release()
        pose_analyzer.close()
        display_service.cleanup()


if __name__ == "__main__":
    main() 