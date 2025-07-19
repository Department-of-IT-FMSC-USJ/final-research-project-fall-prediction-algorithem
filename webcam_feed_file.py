import os
import math
import argparse
from typing import List, Dict, Any, Union

import cv2
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore
from ultralytics import YOLO  # type: ignore
import mediapipe as mp  # type: ignore

# ----------------------------- Constants ----------------------------- #
ROI_MARGIN = 0.25  # Expand person box to include limbs
CONF_THRESHOLD = 0.25

# MediaPipe helpers
mp_pose = mp.solutions.pose  # type: ignore[attr-defined]

# Landmarks of interest (indices follow MediaPipe Pose spec)
ESSENTIAL_LANDMARK_IDS = {
    0,   # Nose
    11, 12,  # Shoulders (L/R)
    23, 24,  # Hips (L/R)
    27, 28,  # Ankles (L/R)
}


# --------------------------- Metric helpers --------------------------- #


def _compute_metrics(lm: List[Any],
                     visibility: List[float]) -> Dict[str, Union[float, None]]:
    """Compute posture metrics for a single person (same implementation as in extract_video_metrics.py)."""
    metrics: Dict[str, Union[float, None]] = {
        "trunk_angle": None,
        "nsar": None,
        "theta_u": None,
        "theta_d": None,
    }

    def vis_ok(idx: int) -> bool:
        return visibility[idx] > 0.3

    # -------------------- Trunk angle -------------------- #
    if vis_ok(11) and vis_ok(23):
        dx = lm[11].x - lm[23].x
        dy = lm[11].y - lm[23].y
        norm = math.hypot(dx, dy)
        if norm > 0:
            cos_theta = (-dy) / norm
            cos_theta = max(-1.0, min(1.0, cos_theta))
            metrics["trunk_angle"] = math.degrees(math.acos(cos_theta))

    # -------------------- θu / θd (mid-plumb) -------------------- #
    req_mid = [11, 12, 23, 24, 27, 28]
    if all(vis_ok(i) for i in req_mid):
        sh_mid_x = (lm[11].x + lm[12].x) * 0.5
        sh_mid_y = (lm[11].y + lm[12].y) * 0.5
        hip_mid_x = (lm[23].x + lm[24].x) * 0.5
        hip_mid_y = (lm[23].y + lm[24].y) * 0.5
        ank_mid_x = (lm[27].x + lm[28].x) * 0.5
        ank_mid_y = (lm[27].y + lm[28].y) * 0.5

        dx_u = hip_mid_x - sh_mid_x
        dy_u = hip_mid_y - sh_mid_y
        dx_d = ank_mid_x - hip_mid_x
        dy_d = ank_mid_y - hip_mid_y

        norm_u = math.hypot(dx_u, dy_u)
        norm_d = math.hypot(dx_d, dy_d)
        if norm_u > 0 and norm_d > 0:
            cos_u = (-dy_u) / norm_u
            cos_d = (-dy_d) / norm_d
            cos_u = max(-1.0, min(1.0, cos_u))
            cos_d = max(-1.0, min(1.0, cos_d))
            metrics["theta_u"] = math.degrees(math.acos(cos_u))
            metrics["theta_d"] = math.degrees(math.acos(cos_d))

    # -------------------- NSAR -------------------- #
    req_nsar = [0, 11, 12, 27, 28]
    if all(vis_ok(i) for i in req_nsar):
        width = abs(lm[11].x - lm[12].x)
        ankle_mid_y = (lm[27].y + lm[28].y) * 0.5
        height = abs(lm[0].y - ankle_mid_y)
        if height > 0:
            metrics["nsar"] = width / height

    return metrics


# --------------------------- Fall Detector --------------------------- #


class FallDetector:
    """Encapsulates fall-detection logic for processing a *single* video file.

    The implementation re-uses the posture metric computation from
    `extract_video_metrics.py` and follows the same fall-detection criteria:

    1. Trunk angle > 45°
    2. Upper & lower mid-plumb angles (θu, θd) > (45°, 60°)
    3. NSAR drops more than 30 % below the baseline (median of first 30 samples)
    """

    def __init__(self, pose_complexity: int = 1):
        self.model = YOLO("yolov8n.pt")
        self.model.fuse()

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=pose_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    # ------------------------------------------------------------------
    def detect(
        self,
        video_path: str,
        show_progress: bool = True,
        collect_metrics: bool = False,
    ) -> Union[bool, List[Dict[str, Any]]]:
        """Process *video_path*.

        Args:
            video_path: path to input video.
            show_progress: whether to display a tqdm progress bar.
            collect_metrics: if **True**, returns a list with per-frame metric
                dictionaries (each containing trunk_angle, nsar, theta_u, theta_d,
                fall_detected).  If **False** (default), stops at the first fall
                trigger and returns a simple boolean.

        Returns:
            • bool – "fall detected?" when *collect_metrics* is False.
            • List[Dict[str, Any]] – per-frame metrics when *collect_metrics* is True.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file '{video_path}'.")

        baseline_nsar: Union[float, None] = None
        baseline_samples: List[float] = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        pbar = tqdm(
            total=frame_count,
            disable=not show_progress,
            desc=os.path.basename(video_path),
            unit="frame",
        )

        fall_detected = False
        metrics_per_frame: List[Dict[str, Any]] = [] if collect_metrics else []  # keep type predictable

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO inference – keep first person only
                results = self.model(frame, conf=CONF_THRESHOLD, verbose=False, show=False)[0]
                person_box = None
                for det_box in results.boxes:
                    if int(det_box.cls[0]) == 0:  # 'person' class
                        person_box = det_box
                        break

                if person_box is not None:
                    x1, y1, x2, y2 = map(int, person_box.xyxy[0])
                    h_frame, w_frame, _ = frame.shape
                    box_w, box_h = x2 - x1, y2 - y1
                    margin_x = int(box_w * ROI_MARGIN)
                    margin_y = int(box_h * ROI_MARGIN)
                    x1_e = max(0, x1 - margin_x)
                    y1_e = max(0, y1 - margin_y)
                    x2_e = min(w_frame, x2 + margin_x)
                    y2_e = min(h_frame, y2 + margin_y)
                    roi = frame[y1_e:y2_e, x1_e:x2_e]

                    if roi.size > 0:
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        pose_results = self.pose.process(roi_rgb)
                        landmarks = pose_results.pose_landmarks
                        if landmarks:
                            lm_list = landmarks.landmark
                            vis_list = [l.visibility for l in lm_list]
                            metrics = _compute_metrics(lm_list, vis_list)
                        else:
                            metrics = {"trunk_angle": None, "nsar": None, "theta_u": None, "theta_d": None}
                    else:
                        metrics = {"trunk_angle": None, "nsar": None, "theta_u": None, "theta_d": None}
                else:
                    metrics = {"trunk_angle": None, "nsar": None, "theta_u": None, "theta_d": None}

                # ---------------- Fall detection logic ---------------- #
                if metrics["nsar"] is not None:
                    nsar_val = metrics["nsar"]
                    if baseline_nsar is None:
                        baseline_samples.append(nsar_val)  # type: ignore[arg-type]
                        if len(baseline_samples) >= 30:
                            sorted_samples = sorted(baseline_samples)
                            mid = len(sorted_samples) // 2
                            baseline_nsar = (
                                sorted_samples[mid]
                                if len(sorted_samples) % 2 == 1
                                else (sorted_samples[mid - 1] + sorted_samples[mid]) * 0.5
                            )

                fall_flag = False

                if (
                    metrics["trunk_angle"] is not None and metrics["trunk_angle"] > 45
                ) or (
                    metrics["theta_u"] is not None
                    and metrics["theta_d"] is not None
                    and metrics["theta_u"] > 45
                    and metrics["theta_d"] > 60
                ) or (
                    baseline_nsar is not None
                    and metrics["nsar"] is not None
                    and metrics["nsar"] < baseline_nsar * 0.7
                ):
                    fall_flag = True

                # Record per-frame metrics if requested
                if collect_metrics:
                    rec = metrics.copy()
                    rec["fall_detected"] = fall_flag
                    metrics_per_frame.append(rec)

                if fall_flag:
                    fall_detected = True
                    if not collect_metrics:
                        break  # early exit when not collecting full timeline

                pbar.update(1)
        finally:
            pbar.close()
            cap.release()

        if collect_metrics:
            return metrics_per_frame
        return fall_detected

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release MediaPipe resources."""
        self.pose.close()

    # Support "with" context manager for automatic cleanup
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # don't suppress exceptions


# --------------------------- CLI entry-point --------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect falls in a video file using the YOLOv8 + MediaPipe pipeline.")
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("--pose-complexity", type=int, choices=[0, 1, 2], default=1,
                        help="MediaPipe Pose complexity (0=lite, 1=full, 2=heavy).")
    args = parser.parse_args()

    with FallDetector(pose_complexity=args.pose_complexity) as detector:
        fall = detector.detect(args.video)

    print("Fall Detected" if fall else "No Fall Detected")


if __name__ == "__main__":
    main() 