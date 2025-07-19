from __future__ import annotations
"""Streaming helper for Flask – provides `stream_frames_flask` generator.

This module reproduces the lightweight fall-detection pipeline from
`webcam_feed.py`, but isolates all Flask-specific streaming utilities so the
original script remains CLI-focused.
"""

from typing import Generator
import time
import math

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
from ultralytics import YOLO  # type: ignore

# Re-use landmark and class definitions from the main script
from webcam_feed import (
    INDOOR_CLASSES,
    ESSENTIAL_LANDMARK_IDS,
    ESSENTIAL_CONNECTIONS,
)

mp_pose = mp.solutions.pose  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# One-time initialisation ----------------------------------------------------
# ---------------------------------------------------------------------------

_stream_inited = False


def _init_streaming_resources(pose_complexity: int = 1):  # noqa: D401
    """Initialise global model / pose instances used during streaming."""
    global _stream_model, _stream_pose, _stream_allowed_classes, _stream_inited

    if _stream_inited:
        return

    _stream_model = YOLO("yolov8n.pt")  # type: ignore[attr-defined]
    _stream_model.fuse()

    _stream_allowed_classes = {
        cls for cls in INDOOR_CLASSES if cls in _stream_model.names.values()
    }

    _stream_pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=pose_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    _stream_inited = True


# ---------------------------------------------------------------------------
# Frame-level processing -----------------------------------------------------
# ---------------------------------------------------------------------------

def _annotate_frame(frame, fps: float | None = None):  # noqa: C901
    """Return *frame* annotated with detections & pose landmarks."""

    global _stream_model, _stream_pose, _stream_allowed_classes
    model = _stream_model
    pose = _stream_pose
    allowed_classes = _stream_allowed_classes

    conf_threshold = 0.25
    results = model(frame, conf=conf_threshold, verbose=False, show=False)[0]

    persons_boxes = []
    for det_box in results.boxes:
        cls_id = int(det_box.cls[0])
        class_name = model.names[cls_id]
        conf = float(det_box.conf[0])
        if conf < conf_threshold or class_name not in allowed_classes:
            continue

        x1, y1, x2, y2 = map(int, det_box.xyxy[0])
        if class_name == "person":
            persons_boxes.append(det_box)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    for box in persons_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(roi_rgb)
        landmarks = pose_results.pose_landmarks

        is_fall = False
        if landmarks:
            lm = landmarks.landmark
            if lm[11].visibility > 0.3 and lm[23].visibility > 0.3:
                dx = lm[11].x - lm[23].x
                dy = lm[11].y - lm[23].y
                norm = (dx * dx + dy * dy) ** 0.5
                if norm > 0:
                    cos_theta = (-dy) / norm
                    cos_theta = max(-1.0, min(1.0, cos_theta))
                    angle_deg = math.degrees(math.acos(cos_theta))
                    if angle_deg > 45:
                        is_fall = True

            # Landmark dots
            for idx in ESSENTIAL_LANDMARK_IDS:
                px = x1 + int(lm[idx].x * (x2 - x1))
                py = y1 + int(lm[idx].y * (y2 - y1))
                cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)

            # Skeleton lines
            for idx1, idx2 in ESSENTIAL_CONNECTIONS:
                if lm[idx1].visibility < 0.3 or lm[idx2].visibility < 0.3:
                    continue
                x_a = x1 + int(lm[idx1].x * (x2 - x1))
                y_a = y1 + int(lm[idx1].y * (y2 - y1))
                x_b = x1 + int(lm[idx2].x * (x2 - x1))
                y_b = y1 + int(lm[idx2].y * (y2 - y1))
                cv2.line(frame, (x_a, y_a), (x_b, y_b), (0, 255, 255), 2)
        else:
            width, height = x2 - x1, y2 - y1
            is_fall = width > height * 1.2

        color = (0, 0, 255) if is_fall else (0, 255, 0)
        label = "FALL" if is_fall else "person"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# ---------------------------------------------------------------------------
# Public generator -----------------------------------------------------------
# ---------------------------------------------------------------------------

def stream_frames_flask() -> Generator[bytes, None, None]:
    """Yield MJPEG frames suitable for a Flask Response."""

    _init_streaming_resources()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for streaming.")

    prev = time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()
            fps_val = 1.0 / (now - prev) if now != prev else 0.0
            prev = now

            frame = _annotate_frame(frame, fps=fps_val)

            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
    finally:
        cap.release() 