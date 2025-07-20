"""Pose analysis and fall detection using MediaPipe."""

import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class PoseMetrics:
    """Container for pose analysis metrics."""
    trunk_angle: Optional[float] = None
    nsar: Optional[float] = None
    theta_u: Optional[float] = None
    theta_d: Optional[float] = None
    status: str = "Normal"
    abnormal_sec: float = 0.0

class PoseAnalyzer:
    """Handles pose estimation and fall detection analysis."""
    
    # Essential landmark IDs for pose analysis
    ESSENTIAL_LANDMARK_IDS = {
        0,   # Nose
        2, 5,  # Eyes (L/R)
        7, 8,  # Ears (L/R)
        9, 10,  # Mouth corners (L/R)
        11, 12,  # Shoulders (L/R)
        13, 14,  # Elbows (L/R)
        15, 16,  # Wrists (L/R)
        23, 24,  # Hips (L/R)
        25, 26,  # Knees (L/R)
        27, 28,  # Ankles (L/R)
        31       # Left foot index (toe)
    }
    
    ESSENTIAL_CONNECTIONS = [
        (11, 12),               # Shoulders
        (11, 13), (13, 15),     # Left arm
        (12, 14), (14, 16),     # Right arm
        (11, 23), (12, 24),     # Torso sides
        (23, 24),               # Hip line
        (23, 25), (25, 27), (27, 31),  # Left leg / foot
        (24, 26), (26, 28)             # Right leg
    ]
    
    def __init__(self, pose_complexity: int = 1, roi_margin: float = 0.25):
        """Initialize pose analyzer with MediaPipe pose estimation.
        
        Args:
            pose_complexity: MediaPipe pose complexity (0=lite, 1=full, 2=heavy)
            roi_margin: Margin around person bounding box for pose estimation
        """
        self.roi_margin = roi_margin
        self.trunk_angle_fall_threshold = math.degrees(0.59)  # ≈ 33.8°
        
        # Initialize MediaPipe Pose
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=pose_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Pose persistence & smoothing
        self.pose_persistence_frames = 5
        self.last_pose_landmarks = None
        self.missing_pose_count = 0
        
        # Smoothing windows
        self.smooth_window = 8
        self.trunk_angle_hist = deque(maxlen=self.smooth_window)
        self.nsar_hist = deque(maxlen=self.smooth_window)
        
        # NSAR baseline
        self.baseline_sample_size = 30
        self.nsar_baseline_samples = deque(maxlen=self.baseline_sample_size)
        self.nsar_baseline = None
        
        # Plumb line fall detection
        self.plumb_fall_counter = 0
        self.plumb_fall_threshold_frames = 60  # ~2 seconds at 30 FPS
        
        # Drawing specs
        self.landmark_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=2
        )
        self.connection_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 255), thickness=2, circle_radius=2
        )
    
    def analyze_person(self, frame: np.ndarray, person_box, fps: float) -> PoseMetrics:
        """Analyze pose for a single person and detect falls.
        
        Args:
            frame: Input frame
            person_box: Person bounding box from YOLO
            fps: Current FPS for timing calculations
            
        Returns:
            PoseMetrics object with analysis results
        """
        x1, y1, x2, y2 = map(int, person_box.xyxy[0])
        
        # Calculate ROI with margin
        h_frame, w_frame, _ = frame.shape
        box_w = x2 - x1
        box_h = y2 - y1
        margin_x = int(box_w * self.roi_margin)
        margin_y = int(box_h * self.roi_margin)
        
        x1_e = max(0, x1 - margin_x)
        y1_e = max(0, y1 - margin_y)
        x2_e = min(w_frame, x2 + margin_x)
        y2_e = min(h_frame, y2 + margin_y)
        
        roi = frame[y1_e:y2_e, x1_e:x2_e]
        if roi.size == 0:
            return PoseMetrics()
        
        # Process pose estimation
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(roi_rgb)
        
        # Handle pose persistence
        landmarks = pose_results.pose_landmarks
        if landmarks is None and self.last_pose_landmarks is not None and self.missing_pose_count < self.pose_persistence_frames:
            self.missing_pose_count += 1
            landmarks = self.last_pose_landmarks
        elif landmarks is not None:
            self.last_pose_landmarks = landmarks
            self.missing_pose_count = 0
        
        if not landmarks:
            return PoseMetrics()
        
        # Draw pose landmarks
        self._draw_pose_landmarks(roi, landmarks)
        
        # Calculate metrics
        metrics = PoseMetrics()
        h, w, _ = roi.shape
        lm = landmarks.landmark
        
        # Trunk angle analysis
        metrics.trunk_angle = self._calculate_trunk_angle(lm, w, h)
        
        # NSAR analysis
        metrics.nsar = self._calculate_nsar(lm, w, h, x1_e, y1_e)
        
        # Mid-plumb line angles
        theta_u, theta_d = self._calculate_plumb_angles(lm, w, h, x1_e, y1_e)
        metrics.theta_u = theta_u
        metrics.theta_d = theta_d
        
        # Determine status
        metrics.status, metrics.abnormal_sec = self._determine_status(
            metrics, fps
        )
        
        return metrics
    
    def _draw_pose_landmarks(self, roi: np.ndarray, landmarks) -> None:
        """Draw pose landmarks and connections on ROI."""
        h, w, _ = roi.shape
        lm = landmarks.landmark
        
        # Draw connections first
        for idx1, idx2 in self.ESSENTIAL_CONNECTIONS:
            if idx1 in self.ESSENTIAL_LANDMARK_IDS and idx2 in self.ESSENTIAL_LANDMARK_IDS:
                x1, y1 = int(lm[idx1].x * w), int(lm[idx1].y * h)
                x2, y2 = int(lm[idx2].x * w), int(lm[idx2].y * h)
                cv2.line(roi, (x1, y1), (x2, y2), 
                        self.connection_spec.color, self.connection_spec.thickness)
        
        # Draw landmark points
        for idx in self.ESSENTIAL_LANDMARK_IDS:
            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(roi, (x, y), self.landmark_spec.circle_radius, 
                      self.landmark_spec.color, cv2.FILLED)
        
        # Draw trunk line (left shoulder to left hip)
        ls = lm[11]  # left shoulder
        lh = lm[23]  # left hip
        
        if ls.visibility > 0.3 and lh.visibility > 0.3:
            x_ls, y_ls = int(ls.x * w), int(ls.y * h)
            x_lh, y_lh = int(lh.x * w), int(lh.y * h)
            cv2.line(roi, (x_lh, y_lh), (x_ls, y_ls), (255, 0, 0), 2)
    
    def _calculate_trunk_angle(self, lm, w: int, h: int) -> Optional[float]:
        """Calculate trunk angle (left shoulder to left hip vs vertical)."""
        ls = lm[11]  # left shoulder
        lh = lm[23]  # left hip
        
        if ls.visibility <= 0.3 or lh.visibility <= 0.3:
            return None
        
        dx = ls.x - lh.x
        dy = ls.y - lh.y
        norm = math.hypot(dx, dy)
        
        if norm <= 0:
            return None
        
        # Dot product with vertical unit vector (0, -1)
        cos_theta = (-dy) / norm
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_deg = math.degrees(math.acos(cos_theta))
        
        # Note: Drawing is handled in _draw_pose_landmarks method
        # We don't draw here to avoid the undefined 'roi' variable
        
        # Smooth the angle
        self.trunk_angle_hist.append(angle_deg)
        return sum(self.trunk_angle_hist) / len(self.trunk_angle_hist)
    
    def _calculate_nsar(self, lm, w: int, h: int, x1_e: int, y1_e: int) -> Optional[float]:
        """Calculate Normalized Shoulder-to-Ankle Ratio."""
        req_indices = [0, 11, 12, 27, 28]
        if not all(lm[i].visibility > 0.3 for i in req_indices):
            return None
        
        # Pixel positions in full frame
        nose_px = (x1_e + int(lm[0].x * w), y1_e + int(lm[0].y * h))
        l_sh_px = (x1_e + int(lm[11].x * w), y1_e + int(lm[11].y * h))
        r_sh_px = (x1_e + int(lm[12].x * w), y1_e + int(lm[12].y * h))
        l_ank_px = (x1_e + int(lm[27].x * w), y1_e + int(lm[27].y * h))
        r_ank_px = (x1_e + int(lm[28].x * w), y1_e + int(lm[28].y * h))
        
        width_px = abs(l_sh_px[0] - r_sh_px[0])
        ankle_mid_y_px = (l_ank_px[1] + r_ank_px[1]) * 0.5
        height_px = abs(nose_px[1] - ankle_mid_y_px)
        
        if height_px <= 0:
            return None
        
        nsar = width_px / height_px
        
        # Smooth NSAR
        self.nsar_hist.append(nsar)
        nsar_smooth = sum(self.nsar_hist) / len(self.nsar_hist)
        
        # Update baseline
        if self.nsar_baseline is None:
            self.nsar_baseline_samples.append(nsar_smooth)
            if len(self.nsar_baseline_samples) >= self.baseline_sample_size:
                sorted_samples = sorted(self.nsar_baseline_samples)
                mid = len(sorted_samples) // 2
                self.nsar_baseline = (
                    sorted_samples[mid]
                    if len(sorted_samples) % 2 == 1
                    else (sorted_samples[mid - 1] + sorted_samples[mid]) * 0.5
                )
        
        return nsar_smooth
    
    def _calculate_plumb_angles(self, lm, w: int, h: int, x1_e: int, y1_e: int) -> Tuple[Optional[float], Optional[float]]:
        """Calculate mid-plumb line angles (θᵤ / θ_d)."""
        req_indices = [0, 11, 12, 23, 24, 27, 28]
        if not all(lm[i].visibility > 0.3 for i in req_indices):
            return None, None
        
        # Convert to full-frame pixel coordinates
        def px(idx: int) -> tuple[int, int]:
            return (
                x1_e + int(lm[idx].x * w),
                y1_e + int(lm[idx].y * h),
            )
        
        l_sh_px = px(11)
        r_sh_px = px(12)
        l_hip_px = px(23)
        r_hip_px = px(24)
        l_ank_px = px(27)
        r_ank_px = px(28)
        
        # Midpoints
        sh_mid = ((l_sh_px[0] + r_sh_px[0]) * 0.5, (l_sh_px[1] + r_sh_px[1]) * 0.5)
        hip_mid = ((l_hip_px[0] + r_hip_px[0]) * 0.5, (l_hip_px[1] + r_hip_px[1]) * 0.5)
        ank_mid = ((l_ank_px[0] + r_ank_px[0]) * 0.5, (l_ank_px[1] + r_ank_px[1]) * 0.5)
        
        # Upper segment vector
        dx_u = hip_mid[0] - sh_mid[0]
        dy_u = hip_mid[1] - sh_mid[1]
        norm_u = math.hypot(dx_u, dy_u)
        
        # Lower segment vector
        dx_d = ank_mid[0] - hip_mid[0]
        dy_d = ank_mid[1] - hip_mid[1]
        norm_d = math.hypot(dx_d, dy_d)
        
        if norm_u <= 0 or norm_d <= 0:
            return None, None
        
        cos_u = (-dy_u) / norm_u
        cos_d = (-dy_d) / norm_d
        cos_u = max(-1.0, min(1.0, cos_u))
        cos_d = max(-1.0, min(1.0, cos_d))
        
        theta_u = math.degrees(math.acos(cos_u))
        theta_d = math.degrees(math.acos(cos_d))
        
        return theta_u, theta_d
    
    def _determine_status(self, metrics: PoseMetrics, fps: float) -> Tuple[str, float]:
        """Determine fall status based on metrics."""
        status = "Normal"
        abnormal_sec = 0.0
        
        # Check trunk angle threshold
        if (metrics.trunk_angle is not None and 
            metrics.trunk_angle > self.trunk_angle_fall_threshold):
            status = "Fall Detected"
        
        # Check NSAR baseline
        if (metrics.nsar is not None and self.nsar_baseline is not None and
            metrics.nsar < self.nsar_baseline * 0.7):
            status = "Fall Detected"
        
        # Check plumb line angles
        plumb_condition_met = (
            metrics.theta_u is not None and metrics.theta_d is not None and
            metrics.theta_u > 45 and metrics.theta_d > 60
        )
        
        if plumb_condition_met:
            self.plumb_fall_counter += 1
            abnormal_sec = self.plumb_fall_counter / fps if fps > 0 else 0.0
            
            if self.plumb_fall_counter >= self.plumb_fall_threshold_frames:
                status = "Fall Detected"
            elif status == "Normal":
                status = "Warning"
        else:
            self.plumb_fall_counter = 0
        
        return status, abnormal_sec
    
    def close(self) -> None:
        """Clean up MediaPipe pose resources."""
        self.pose.close() 