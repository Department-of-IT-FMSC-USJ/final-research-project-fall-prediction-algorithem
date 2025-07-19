from __future__ import annotations
"""Streaming helper for Flask – provides `stream_frames_flask` generator.

This module reproduces the lightweight fall-detection pipeline from
`webcam_feed.py`, but isolates all Flask-specific streaming utilities so the
original script remains CLI-focused.
"""

from typing import Dict, Any, Generator, List
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

# ------------- Live statistics shared with Flask ------------- #
_latest_stats: Dict[str, Any] = {
    "fps": None,
    "trunk_angle": None,
    "nsar": None,
    "theta_u": None,
    "theta_d": None,
    "fall_detected": False,
}

def get_latest_stats() -> Dict[str, Any]:  # noqa: D401
    """Return a shallow copy of the most recently computed frame statistics."""
    return _latest_stats.copy()

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
    global _latest_stats

    trunk_angle_value: float | None = None  # stats for person #1
    nsar_value: float | None = None
    theta_u_value: float | None = None
    theta_d_value: float | None = None
    frame_fall_detected = False

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

    for person_idx, box in enumerate(persons_boxes, start=1):
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

                    # cache first computed trunk angle for stats
                    if person_idx == 1 and trunk_angle_value is None:
                        trunk_angle_value = angle_deg

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

        # Accumulate fall flag for this frame
        if is_fall:
            frame_fall_detected = True

        # ---- Additional metrics for person #1 ---- #
        if landmarks and person_idx == 1:
            # Mid-plumb angles (θu / θd)
            req_indices = [0, 11, 12, 23, 24, 27, 28]
            if all(lm[i].visibility > 0.3 for i in req_indices):
                w_roi, h_roi = roi.shape[1], roi.shape[0]
                px = lambda idx: (
                    x1 + int(lm[idx].x * (x2 - x1)),
                    y1 + int(lm[idx].y * (y2 - y1)),
                )
                l_sh, r_sh = px(11), px(12)
                l_hip, r_hip = px(23), px(24)
                l_ank, r_ank = px(27), px(28)

                sh_mid = ((l_sh[0] + r_sh[0]) * 0.5, (l_sh[1] + r_sh[1]) * 0.5)
                hip_mid = ((l_hip[0] + r_hip[0]) * 0.5, (l_hip[1] + r_hip[1]) * 0.5)
                ank_mid = ((l_ank[0] + r_ank[0]) * 0.5, (l_ank[1] + r_ank[1]) * 0.5)

                dx_u, dy_u = hip_mid[0] - sh_mid[0], hip_mid[1] - sh_mid[1]
                dx_d, dy_d = ank_mid[0] - hip_mid[0], ank_mid[1] - hip_mid[1]
                norm_u = (dx_u**2 + dy_u**2) ** 0.5
                norm_d = (dx_d**2 + dy_d**2) ** 0.5
                if norm_u > 0 and norm_d > 0:
                    theta_u_value = math.degrees(math.acos(max(-1.0, min(1.0, (-dy_u) / norm_u))))
                    theta_d_value = math.degrees(math.acos(max(-1.0, min(1.0, (-dy_d) / norm_d))))

            # NSAR
            req_indices2 = [0, 11, 12, 27, 28]
            if all(lm[i].visibility > 0.3 for i in req_indices2):
                w_roi, h_roi = roi.shape[1], roi.shape[0]
                nose_px = (x1 + int(lm[0].x * (x2 - x1)), y1 + int(lm[0].y * (y2 - y1)))
                l_sh_px = (x1 + int(lm[11].x * (x2 - x1)), y1 + int(lm[11].y * (y2 - y1)))
                r_sh_px = (x1 + int(lm[12].x * (x2 - x1)), y1 + int(lm[12].y * (y2 - y1)))
                l_ank_px = (x1 + int(lm[27].x * (x2 - x1)), y1 + int(lm[27].y * (y2 - y1)))
                r_ank_px = (x1 + int(lm[28].x * (x2 - x1)), y1 + int(lm[28].y * (y2 - y1)))

                width_px = abs(l_sh_px[0] - r_sh_px[0])
                ankle_mid_y = (l_ank_px[1] + r_ank_px[1]) * 0.5
                height_px = abs(nose_px[1] - ankle_mid_y)
                if height_px > 0:
                    nsar_value = width_px / height_px

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ------------- update live stats shared with Flask ------------- #
    if fps is not None:
        _latest_stats["fps"] = fps
    if trunk_angle_value is not None:
        _latest_stats["trunk_angle"] = trunk_angle_value
    if nsar_value is not None:
        _latest_stats["nsar"] = nsar_value
    if theta_u_value is not None:
        _latest_stats["theta_u"] = theta_u_value
    if theta_d_value is not None:
        _latest_stats["theta_d"] = theta_d_value
    _latest_stats["fall_detected"] = frame_fall_detected

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