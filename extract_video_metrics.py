import os
import json
import math
import argparse
from typing import List, Dict, Any, Union

import cv2
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore

# The script still constructs mp_pose for completeness (even if unused later)
import mediapipe as mp  # type: ignore

# YOLOv8 and MediaPipe imports
from webcam_feed_file import FallDetector

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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract trunk angle, NSAR, and mid-plumb angles for every frame in a directory of videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    default_dir = r"C:\\Users\\kasun\\OneDrive - Faculty of Management Studies and Commerce\\BIS\\Research Docs\\ML video sampels\\Coffee_room_01\\Coffee_room_01\\Videos"

    parser.add_argument("--video-dir", type=str, default=default_dir, help="Directory containing test videos.")
    parser.add_argument("--output-dir", type=str, default="metrics_json", help="Directory to write JSON metric files.")
    parser.add_argument("--max-files", type=int, default=5, help="Maximum number of video files to process.")
    parser.add_argument("--pose-complexity", type=int, choices=[0, 1, 2], default=1, help="MediaPipe Pose complexity level.")
    return parser.parse_args()


# --------------------------- Metric helpers --------------------------- #

def _compute_metrics(lm: List[Any],
                     visibility: List[float]) -> Dict[str, Union[float, None]]:
    """Compute posture metrics for a single person.

    Args:
        lm: List of landmarks (length 33) in **normalized** coordinates.
        visibility: Visibility scores for each landmark (length 33).

    Returns:
        Dict with keys trunk_angle, nsar, theta_u, theta_d – values may be None.
    """
    # Initialise all metrics as None; compute only if landmarks available
    metrics: Dict[str, Union[float, None]] = {
        "trunk_angle": None,
        "nsar": None,
        "theta_u": None,
        "theta_d": None,
    }

    # Helper to check landmark visibility > 0.3
    def vis_ok(idx: int) -> bool:
        return visibility[idx] > 0.3

    # -------------------- Trunk angle -------------------- #
    if vis_ok(11) and vis_ok(23):
        dx = lm[11].x - lm[23].x
        dy = lm[11].y - lm[23].y
        norm = math.hypot(dx, dy)
        if norm > 0:
            cos_theta = (-dy) / norm  # vertical reference
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

        # Upper vector (shoulder-hip)
        dx_u = hip_mid_x - sh_mid_x
        dy_u = hip_mid_y - sh_mid_y
        # Lower vector (hip-ankle)
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


# --------------------------- Main routine --------------------------- #

def main() -> None:
    args = _parse_args()

    video_dir = os.path.abspath(args.video_dir)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Collect video files (common extensions)
    valid_ext = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
              if os.path.splitext(f)[1].lower() in valid_ext]
    # Limit to the first N files as requested
    videos = sorted(videos)[:args.max_files]

    if not videos:
        raise RuntimeError(f"No video files found in {video_dir}.")

    print(f"Found {len(videos)} videos in {video_dir}. Processing…")

    # Instantiate the reusable detector once to avoid loading the model repeatedly
    detector = FallDetector(pose_complexity=args.pose_complexity)

    try:
        for vid_path in videos:
            vid_name = os.path.splitext(os.path.basename(vid_path))[0]

            # Collect per-frame metrics via the detector
            metrics_per_frame = detector.detect(vid_path, collect_metrics=True)  # type: ignore[assignment]

            # Type narrowing for static checkers
            assert isinstance(metrics_per_frame, list)

            # Save JSON exactly like the original implementation
            out_path = os.path.join(out_dir, f"{vid_name}_metrics.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metrics_per_frame, f, indent=2)

            print(f"Saved {len(metrics_per_frame)} frame metrics to {out_path}.")
    finally:
        detector.close()


if __name__ == "__main__":
    main() 