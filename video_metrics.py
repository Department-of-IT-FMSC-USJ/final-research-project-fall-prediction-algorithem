# video_metrics.py
"""Offline fall-detection metrics extraction.

This script reuses the lightweight YOLOv8 + MediaPipe pipeline from
`webcam_feed.py` but operates **offline** on a provided video file instead of a
live webcam.  For each frame it records the following metrics (or ``null`` when
unavailable):

* ``trunk_angle`` – degrees between trunk vector and the vertical
* ``nsar`` – Normalised Shoulder-to-Ankle Ratio
* ``theta_u`` – upper mid-plumb angle (shoulder–hip)
* ``theta_d`` – lower mid-plumb angle (hip–ankle)
* ``fall_detected`` – boolean flag indicating a fall event on that frame

The collected list is saved as JSON to ``metrics_json/<video-name>_metrics.json``
or a custom path supplied via ``--output``.

Usage:
    python video_metrics.py --input path/to/video.mp4 [--output out.json] [--show]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import List, Union, Dict, Any

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
from ultralytics import YOLO  # type: ignore

# -----------------------------------------------------------------------------
# Re-use constants from the live pipeline to stay perfectly in sync
# -----------------------------------------------------------------------------
from webcam_feed import (
    ROI_MARGIN,
    TRUNK_ANGLE_FALL_THRESHOLD_DEG,
    ESSENTIAL_LANDMARK_IDS,
)  # noqa: E402

# Only the ID set is required for the offline analyser; the drawing connection
# list is omitted because we do not need to render skeletons when ``--show`` is
# disabled.  Import lazily if the user requests visualisation.

mp_pose = mp.solutions.pose  # type: ignore[attr-defined]

# ---------------- Smoothing / baseline parameters ----------------------------
SMOOTH_WINDOW = 8          # moving-average window for angles & NSAR
BASELINE_SAMPLE_SIZE = 30  # frames used to estimate the standing NSAR baseline
PLUMB_FALL_THRESHOLD_FRAMES = 60  # ≈ 2 s at 30 FPS


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:  # noqa: D401
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
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _ensure_output_path(input_path: Path, custom_out: str | None) -> Path:
    if custom_out:
        return Path(custom_out).expanduser().resolve()
    out_dir = Path("metrics_json")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{input_path.stem}_metrics.json"


# -----------------------------------------------------------------------------
# Core analysis routine
# -----------------------------------------------------------------------------

def _process_video(
    src_path: Path,
    out_path: Path,
    pose_complexity: int = 1,
    show: bool = False,
) -> None:
    # ---------------- Model initialisation ----------------------------------
    model = YOLO("yolov8n.pt")
    model.fuse()

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=pose_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # Filter to COCO classes we annotate in the live feed (optional rendering)
    allowed_classes = {cls for cls in {
        "person",
        "chair",
        "couch",
        "bed",
        "dining table",
        "tv",
        "bottle",
        "laptop",
        "book",
        "cell phone",
        "remote",
        "keyboard",
        "mouse",
        "potted plant",
    } if cls in model.names.values()}

    # ---------------- Video IO ----------------------------------------------
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file '{src_path}'.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

    # ---------------- Runtime helpers & buffers -----------------------------
    trunk_angle_hist: deque[float] = deque(maxlen=SMOOTH_WINDOW)
    nsar_hist: deque[float] = deque(maxlen=SMOOTH_WINDOW)
    nsar_baseline_samples: deque[float] = deque(maxlen=BASELINE_SAMPLE_SIZE)
    nsar_baseline: float | None = None

    plumb_fall_counter = 0

    metrics: List[Dict[str, Any]] = []

    frame_idx = 0
    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # EOF

            frame_idx += 1
            current_time = time.perf_counter()
            # ------------------------------------------------------------------
            # Object detection (YOLOv8)
            # ------------------------------------------------------------------
            results = model(frame, conf=0.25, verbose=False, show=False)[0]

            # Gather person detections
            persons_boxes = [box for box in results.boxes if model.names[int(box.cls[0])] == "person"]

            # Default per-frame values
            trunk_angle_value: float | None = None
            nsar_value: float | None = None
            theta_u_value: float | None = None
            theta_d_value: float | None = None
            fall_detected = False

            if persons_boxes:
                # Analyse **first** detected person (closest proxy to live pipeline)
                box = persons_boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Enlarge bounding box to reduce limb truncation
                h_frame, w_frame, _ = frame.shape
                margin_x = int((x2 - x1) * ROI_MARGIN)
                margin_y = int((y2 - y1) * ROI_MARGIN)
                x1_e = max(0, x1 - margin_x)
                y1_e = max(0, y1 - margin_y)
                x2_e = min(w_frame, x2 + margin_x)
                y2_e = min(h_frame, y2 + margin_y)
                roi = frame[y1_e:y2_e, x1_e:x2_e]

                if roi.size:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(roi_rgb)
                    landmarks = pose_results.pose_landmarks

                    if landmarks:
                        lm = landmarks.landmark
                        w_roi, h_roi = x2_e - x1_e, y2_e - y1_e

                        # -------- Trunk angle (left shoulder–left hip) ---------
                        if lm[11].visibility > 0.3 and lm[23].visibility > 0.3:
                            dx = lm[11].x - lm[23].x
                            dy = lm[11].y - lm[23].y
                            norm = math.hypot(dx, dy)
                            if norm > 0:
                                angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, (-dy) / norm))))
                                trunk_angle_hist.append(angle_deg)
                                trunk_angle_value = sum(trunk_angle_hist) / len(trunk_angle_hist)

                        # -------- Mid-plumb angles (θᵤ / θ_d) ------------------
                        req_indices = [0, 11, 12, 23, 24, 27, 28]
                        if all(lm[i].visibility > 0.3 for i in req_indices):
                            # Helper lambda: return full-frame px coordinates
                            px = lambda idx: (
                                x1_e + int(lm[idx].x * w_roi),
                                y1_e + int(lm[idx].y * h_roi),
                            )
                            l_sh, r_sh = px(11), px(12)
                            l_hip, r_hip = px(23), px(24)
                            l_ank, r_ank = px(27), px(28)

                            sh_mid = ((l_sh[0] + r_sh[0]) * 0.5, (l_sh[1] + r_sh[1]) * 0.5)
                            hip_mid = ((l_hip[0] + r_hip[0]) * 0.5, (l_hip[1] + r_hip[1]) * 0.5)
                            ank_mid = ((l_ank[0] + r_ank[0]) * 0.5, (l_ank[1] + r_ank[1]) * 0.5)

                            # Upper segment
                            dx_u, dy_u = hip_mid[0] - sh_mid[0], hip_mid[1] - sh_mid[1]
                            norm_u = math.hypot(dx_u, dy_u)
                            # Lower segment
                            dx_d, dy_d = ank_mid[0] - hip_mid[0], ank_mid[1] - hip_mid[1]
                            norm_d = math.hypot(dx_d, dy_d)

                            if norm_u > 0 and norm_d > 0:
                                theta_u = math.degrees(math.acos(max(-1.0, min(1.0, (-dy_u) / norm_u))))
                                theta_d = math.degrees(math.acos(max(-1.0, min(1.0, (-dy_d) / norm_d))))
                                theta_u_value, theta_d_value = theta_u, theta_d

                                if theta_u > 45 and theta_d > 60:
                                    plumb_fall_counter += 1
                                else:
                                    plumb_fall_counter = 0

                        # -------- NSAR ---------------------------------------
                        req_indices = [0, 11, 12, 27, 28]
                        if all(lm[i].visibility > 0.3 for i in req_indices):
                            nose_px = (x1_e + int(lm[0].x * w_roi), y1_e + int(lm[0].y * h_roi))
                            l_sh_px = (x1_e + int(lm[11].x * w_roi), y1_e + int(lm[11].y * h_roi))
                            r_sh_px = (x1_e + int(lm[12].x * w_roi), y1_e + int(lm[12].y * h_roi))
                            l_ank_px = (x1_e + int(lm[27].x * w_roi), y1_e + int(lm[27].y * h_roi))
                            r_ank_px = (x1_e + int(lm[28].x * w_roi), y1_e + int(lm[28].y * h_roi))

                            width_px = abs(l_sh_px[0] - r_sh_px[0])
                            ankle_mid_y = (l_ank_px[1] + r_ank_px[1]) * 0.5
                            height_px = abs(nose_px[1] - ankle_mid_y)
                            if height_px > 0:
                                nsar = width_px / height_px
                                nsar_hist.append(nsar)
                                nsar_smooth = sum(nsar_hist) / len(nsar_hist)
                                nsar_value = nsar_smooth

                                if nsar_baseline is None:
                                    nsar_baseline_samples.append(nsar_smooth)
                                    if len(nsar_baseline_samples) >= BASELINE_SAMPLE_SIZE:
                                        sorted_samples = sorted(nsar_baseline_samples)
                                        mid = len(sorted_samples) // 2
                                        nsar_baseline = (
                                            sorted_samples[mid]
                                            if len(sorted_samples) % 2 == 1
                                            else (sorted_samples[mid - 1] + sorted_samples[mid]) * 0.5
                                        )
                                elif nsar_smooth < nsar_baseline * 0.7:
                                    fall_detected = True

                        # -------- Trunk-angle condition ----------------------
                        if trunk_angle_value is not None and trunk_angle_value > TRUNK_ANGLE_FALL_THRESHOLD_DEG:
                            fall_detected = True

                        # -------- Mid-plumb persistence ----------------------
                        if plumb_fall_counter >= PLUMB_FALL_THRESHOLD_FRAMES:
                            fall_detected = True

            # Append per-frame metrics ---------------------------------------
            metrics.append(
                {
                    "trunk_angle": trunk_angle_value,
                    "nsar": nsar_value,
                    "theta_u": theta_u_value,
                    "theta_d": theta_d_value,
                    "fall_detected": fall_detected,
                }
            )

            # Optional preview -----------------------------------------------
            if show:
                # Only import drawing utilities when needed to avoid overhead
                from webcam_feed import _LANDMARK_SPEC, _CONNECTION_SPEC, ESSENTIAL_CONNECTIONS  # noqa: E402

                display_frame = frame.copy()
                # Draw detections for allowed classes (cyan / green / red)
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    if class_name not in allowed_classes:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 0, 255) if class_name == "person" else (255, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                cv2.imshow("Preview – press 'q' to abort", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Aborted by user – partial metrics will be saved.")
                    break

            # Progress indicator (optional)
            if total_frames and frame_idx % 30 == 0:
                pct = frame_idx / total_frames * 100
                print(f"Processed {frame_idx}/{total_frames} frames ({pct:.1f}%)", end="\r")

            # Minimal sleep to keep UI responsive when showing frames
            if show:
                time.sleep(0.001)

    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        pose.close()

    # ---------------- Save metrics ------------------------------------------
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {len(metrics)} frame metrics to '{out_path}'.")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    args = _parse_args()
    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input video '{src}' not found.")

    out = _ensure_output_path(src, args.output)
    _process_video(src, out, pose_complexity=args.pose_complexity, show=args.show)


if __name__ == "__main__":
    main() 