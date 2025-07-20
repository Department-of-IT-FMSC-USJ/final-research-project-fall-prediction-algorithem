from __future__ import annotations
"""Streaming helper for Flask – provides `stream_frames_flask` generator.

This module reproduces the lightweight fall-detection pipeline from
`webcam_feed.py`, but isolates all Flask-specific streaming utilities so the
original script remains CLI-focused.
"""

from typing import Dict, Any, Generator, List, Deque
import time
import math
import os
from collections import deque

from dotenv import load_dotenv  # type: ignore
from telegram_sender import _send as telegram_send  # type: ignore
from azure_foundry_predict import predict as foundry_predict  # type: ignore

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
from ultralytics import YOLO  # type: ignore

# Re-use landmark and class definitions from the main script
from webcam_feed import (
    INDOOR_CLASSES,
    ESSENTIAL_LANDMARK_IDS,
    ESSENTIAL_CONNECTIONS,
    ROI_MARGIN,
    TRUNK_ANGLE_FALL_THRESHOLD_DEG,
)

# Load .env early (Telegram & Azure credentials)
load_dotenv()

# ---------------- Pose / smoothing helpers ---------------- #
POSE_PERSISTENCE_FRAMES = 5
SMOOTH_WINDOW = 8
BASELINE_SAMPLE_SIZE = 30

# Runtime state (globals to persist across frames)
last_pose_landmarks = None  # cached landmarks
missing_pose_count = 0

trunk_angle_hist: deque[float] = deque(maxlen=SMOOTH_WINDOW)
nsar_hist: deque[float] = deque(maxlen=SMOOTH_WINDOW)

nsar_baseline_samples: deque[float] = deque(maxlen=BASELINE_SAMPLE_SIZE)
nsar_baseline: float | None = None

plumb_fall_counter = 0
PLUMB_FALL_THRESHOLD_FRAMES = 60

last_status_text = "Normal"
foundry_response_global: str | None = None

# ---------------- Time-series buffer ----------------#
metrics_buffer: Deque[dict] = deque(maxlen=300)  # ~5 minutes at 1 Hz
last_buffer_time = 0.0

def get_metrics_history() -> list[dict]:
    """Return a snapshot list of buffered metrics (oldest→newest)."""
    return list(metrics_buffer)

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
    global _latest_stats, last_pose_landmarks, missing_pose_count
    global nsar_baseline, plumb_fall_counter, last_status_text, foundry_response_global

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

    plumb_condition_met = False

    for person_idx, box in enumerate(persons_boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ---------- Expand ROI by margin ---------- #
        h_frame, w_frame, _ = frame.shape
        box_w, box_h = x2 - x1, y2 - y1
        margin_x = int(box_w * ROI_MARGIN)
        margin_y = int(box_h * ROI_MARGIN)
        x1_e = max(0, x1 - margin_x)
        y1_e = max(0, y1 - margin_y)
        x2_e = min(w_frame, x2 + margin_x)
        y2_e = min(h_frame, y2 + margin_y)

        roi = frame[y1_e:y2_e, x1_e:x2_e]
        if roi.size == 0:
            continue

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(roi_rgb)

        # ---------- Landmark persistence ---------- #
        landmarks = pose_results.pose_landmarks
        if landmarks is None and last_pose_landmarks is not None and missing_pose_count < POSE_PERSISTENCE_FRAMES:
            missing_pose_count += 1
            landmarks = last_pose_landmarks
        elif landmarks is not None:
            last_pose_landmarks = landmarks
            missing_pose_count = 0

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
                    # ---- Trunk angle logic ---- #
                    if person_idx == 1:
                        trunk_angle_hist.append(angle_deg)
                        trunk_angle_smooth = sum(trunk_angle_hist) / len(trunk_angle_hist)
                        trunk_angle_value = trunk_angle_smooth

                        if trunk_angle_smooth > TRUNK_ANGLE_FALL_THRESHOLD_DEG:
                            is_fall = True

                    if angle_deg > 45 and person_idx != 1:
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

                    # Baseline establishment & fall rule
                    if nsar_baseline is None:
                        nsar_baseline_samples.append(nsar_value)
                        if len(nsar_baseline_samples) >= BASELINE_SAMPLE_SIZE:
                            sorted_samples = sorted(nsar_baseline_samples)
                            mid = len(sorted_samples) // 2
                            nsar_baseline = (
                                sorted_samples[mid] if len(sorted_samples) % 2 == 1 else (sorted_samples[mid - 1] + sorted_samples[mid]) * 0.5
                            )
                    else:
                        nsar_hist.append(nsar_value)
                        nsar_smooth = sum(nsar_hist) / len(nsar_hist)
                        nsar_value = nsar_smooth
                        if nsar_smooth < nsar_baseline * 0.7:
                            is_fall = True

    # ---------- Plumb-angle combo rule ---------- #
    if theta_u_value is not None and theta_d_value is not None:
        if theta_u_value > 45 and theta_d_value > 60:
            plumb_condition_met = True

    # Accumulate fall flag for this frame
    if is_fall:
        frame_fall_detected = True

    # ---------- External alerts on transition ---------- #
    status_text = "Fall Detected" if frame_fall_detected else "Normal"
    if status_text == "Fall Detected" and last_status_text != "Fall Detected":
        prompt = (
            f"Metrics: trunk_angle={trunk_angle_value}, "
            f"nsar={nsar_value}, theta_d={theta_d_value}. "
            "Predict fall risk."
        )
        print("[AzureFoundry] Prompt:", prompt)
        try:
            foundry_response_global = foundry_predict(prompt)
            print("[Azure Foundry]", foundry_response_global)
        except Exception as e:
            print(f"[AzureFoundry] Failed: {e}")

        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            try:
                msg = "Fall detected!"
                if foundry_response_global:
                    msg += f"\n\nPrediction:\n{foundry_response_global}"
                telegram_send(token, int(chat_id), msg)
            except Exception as e:
                print(f"[Telegram] Failed: {e}")

    last_status_text = status_text

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ------------- update live stats shared with Flask ------------- #
    if fps is not None:
        _latest_stats["fps"] = fps
    if trunk_angle_value is not None:
        _latest_stats["trunk_angle"] = trunk_angle_value
    # Keep previous value if metric not available this frame
    if nsar_value is not None:
        _latest_stats["nsar"] = nsar_value
    if theta_u_value is not None:
        _latest_stats["theta_u"] = theta_u_value
    if theta_d_value is not None:
        _latest_stats["theta_d"] = theta_d_value
    _latest_stats["fall_detected"] = frame_fall_detected
    if foundry_response_global is not None:
        _latest_stats["prediction"] = foundry_response_global

    # Display foundry response on stream (small overlay)
    if foundry_response_global:
        y = 40
        for line in foundry_response_global.split("\n"):
            cv2.putText(frame, line[:60], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y += 15

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

            # -------- Time-series sampler (~1 Hz) --------
            global last_buffer_time
            if now - last_buffer_time >= 1.0:
                metrics_buffer.append({
                    "ts": now,
                    "trunk_angle": _latest_stats.get("trunk_angle"),
                    "nsar": _latest_stats.get("nsar"),
                    "theta_u": _latest_stats.get("theta_u"),
                    "theta_d": _latest_stats.get("theta_d"),
                    "fall": _latest_stats.get("fall_detected"),
                })
                last_buffer_time = now

            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
    finally:
        cap.release() 