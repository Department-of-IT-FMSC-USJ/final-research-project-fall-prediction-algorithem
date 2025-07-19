"""Webcam Person Detection (YOLOv8)

Live webcam feed with real-time person detection powered by the lightweight
YOLOv8n model from the *ultralytics* package. A bounding box and confidence
score are drawn around each detected person. Press the 'q' key to quit.
"""

import cv2
# The ultralytics package is optional at lint time; add type ignore for static checkers.
# YOLOv8 for person detection
from ultralytics import YOLO  # type: ignore

# MediaPipe modules
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
import math
import time
import ctypes
from collections import deque  # Added for buffering time-series metrics

mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore[attr-defined]
mp_pose = mp.solutions.pose  # type: ignore[attr-defined]

# ----------------- Pose landmark subset ----------------- #
# Keep 20 key landmarks for a compact yet informative skeleton.
# Indices follow MediaPipe Pose (see docs).
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

# Basic skeleton connections among the selected landmarks
ESSENTIAL_CONNECTIONS = [
    (11, 12),               # Shoulders
    (11, 13), (13, 15),     # Left arm
    (12, 14), (14, 16),     # Right arm
    (11, 23), (12, 24),     # Torso sides
    (23, 24),               # Hip line
    (23, 25), (25, 27), (27, 31),  # Left leg / foot
    (24, 26), (26, 28)             # Right leg
]

# Drawing specs
_LANDMARK_SPEC = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
_CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# ---------------- Indoor object classes (COCO) ---------------- #
# Used for generic object detection overlay alongside fall detection
INDOOR_CLASSES = {
    "person",
    "chair",
    "couch",          # sofa
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
}


def main() -> None:
    """Start the webcam feed and display frames until the user presses 'q'."""
    # 0 is typically the default webcam. Change the index if you have multiple cameras.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Make sure a camera is connected and not used by another application.")

    # Load a lightweight YOLOv8 model (nano version) trained on COCO
    model = YOLO("yolov8n.pt")

    # Allowed object classes for overlay (filter to those present in model)
    allowed_classes = {cls for cls in INDOOR_CLASSES if cls in model.names.values()}

    # Confidence threshold for displaying detections
    conf_threshold = 0.25

    # Initialize MediaPipe Pose once (low-complexity model for speed)
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=0,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    # Baseline NSAR (Normalized Shoulder-to-Ankle Ratio) value; set after first valid measurement
    nsar_baseline = None

    # Counter for consecutive frames where both upper and lower plumb angles exceed thresholds
    plumb_fall_counter = 0
    PLUMB_FALL_THRESHOLD_FRAMES = 60  # ~2 seconds at 30 FPS

    window_name = "Webcam Live Feed – Person Detection (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Make the window fullscreen for better visibility
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Screen resolution (for resizing frames to fullscreen)
    screen_w = ctypes.windll.user32.GetSystemMetrics(0)
    screen_h = ctypes.windll.user32.GetSystemMetrics(1)

    # FPS calculation helpers
    prev_time = time.perf_counter()
    fps = 0.0

    # ---------------- Time-series metric buffer ---------------- #
    metrics_buffer = deque(maxlen=60)  # store at most 60 sampled frames (~1 minute)
    last_buffer_time = prev_time  # timestamp of last sample

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                break

            # Time measurement for FPS
            current_time = time.perf_counter()
            fps = 1.0 / (current_time - prev_time) if current_time != prev_time else fps
            prev_time = current_time

            # Initialize per-frame display metrics
            trunk_angle_value = None  # degrees
            nsar_value = None         # ratio
            theta_u_value = None      # degrees
            theta_d_value = None      # degrees
            status_text = "Normal"
            abnormal_sec = 0.0

            # Reset per-frame plumb condition flag
            plumb_condition_met = False

            # Run YOLO inference (all classes) without spawning its own window
            results = model(frame, conf=conf_threshold, verbose=False, show=False)[0]

            # ---------------- Object detection overlay ---------------- #
            persons_boxes = []  # collect person detections for pose processing

            for det_box in results.boxes:
                cls_id = int(det_box.cls[0])
                class_name = model.names[cls_id]
                conf = float(det_box.conf[0])

                # Filter by confidence and allowed classes
                if conf < conf_threshold or class_name not in allowed_classes:
                    continue

                x1, y1, x2, y2 = map(int, det_box.xyxy[0])

                if class_name == "person":
                    # Queue for fall-detection processing later
                    persons_boxes.append(det_box)
                else:
                    # Draw non-person object bounding box (cyan)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 255, 0), 1, cv2.LINE_AA)

            # --------- Pose & fall detection for person instances --------- #
            for person_idx, box in enumerate(persons_boxes, start=1):
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"person {person_idx}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1, cv2.LINE_AA)

                # ------- MediaPipe Pose on the ROI ------- #
                roi = frame[y1:y2, x1:x2]
                # Skip if ROI is empty
                if roi.size == 0:
                    continue

                # MediaPipe expects RGB images
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(roi_rgb)

                if pose_results.pose_landmarks:
                    # Draw only selected landmarks
                    h, w, _ = roi.shape
                    lm = pose_results.pose_landmarks.landmark

                    # Draw connections first (for underlay)
                    for idx1, idx2 in ESSENTIAL_CONNECTIONS:
                        if idx1 in ESSENTIAL_LANDMARK_IDS and idx2 in ESSENTIAL_LANDMARK_IDS:
                            x1, y1 = int(lm[idx1].x * w), int(lm[idx1].y * h)
                            x2, y2 = int(lm[idx2].x * w), int(lm[idx2].y * h)
                            cv2.line(roi, (x1, y1), (x2, y2), _CONNECTION_SPEC.color, _CONNECTION_SPEC.thickness)

                    # Draw landmark points
                    for idx in ESSENTIAL_LANDMARK_IDS:
                        x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                        cv2.circle(roi, (x, y), _LANDMARK_SPEC.circle_radius, _LANDMARK_SPEC.color, cv2.FILLED)

                    # -------- Trunk angle (left shoulder–left hip vs vertical) -------- #
                    # Landmark indices: 11 = left shoulder, 23 = left hip
                    ls = lm[11]
                    lh = lm[23]

                    # Only compute if visibility is reasonable (>0.3)
                    if ls.visibility > 0.3 and lh.visibility > 0.3:
                        dx = ls.x - lh.x
                        dy = ls.y - lh.y
                        norm = math.hypot(dx, dy)
                        if norm > 0:
                            # Dot product with vertical unit vector (0, -1)
                            cos_theta = (-dy) / norm  # negative because image y-axis is downwards
                            cos_theta = max(-1.0, min(1.0, cos_theta))  # clamp
                            angle_deg = math.degrees(math.acos(cos_theta))

                            # Convert normalized coords to pixels for drawing
                            x_ls, y_ls = int(ls.x * w), int(ls.y * h)
                            x_lh, y_lh = int(lh.x * w), int(lh.y * h)

                            # Draw the trunk line
                            cv2.line(roi, (x_lh, y_lh), (x_ls, y_ls), (255, 0, 0), 2)

                            # Store angle for global display (first person only)
                            if person_idx == 1:
                                trunk_angle_value = angle_deg

                            # Trigger fall alert if trunk angle exceeds 45° (first person only)
                            if person_idx == 1 and angle_deg > 45:
                                status_text = "Fall Detected"
                                trunk_fall_alert_display = True

                    # -------- Mid-Plumb Line Angles (θᵤ / θ_d) -------- #
                    if person_idx == 1:
                        req_indices = [0, 11, 12, 23, 24, 27, 28]
                        if all(lm[i].visibility > 0.3 for i in req_indices):
                            # Normalized coords
                            l_sh = lm[11]
                            r_sh = lm[12]
                            l_hip = lm[23]
                            r_hip = lm[24]
                            l_ank = lm[27]
                            r_ank = lm[28]

                            # Midpoints in normalized space
                            sh_mid_x = (l_sh.x + r_sh.x) * 0.5
                            sh_mid_y = (l_sh.y + r_sh.y) * 0.5

                            hip_mid_x = (l_hip.x + r_hip.x) * 0.5
                            hip_mid_y = (l_hip.y + r_hip.y) * 0.5

                            ank_mid_x = (l_ank.x + r_ank.x) * 0.5
                            ank_mid_y = (l_ank.y + r_ank.y) * 0.5

                            # Upper segment vector (hip midpoint - shoulder midpoint)
                            dx_u = hip_mid_x - sh_mid_x
                            dy_u = hip_mid_y - sh_mid_y
                            norm_u = math.hypot(dx_u, dy_u)

                            # Lower segment vector (ankle midpoint - hip midpoint)
                            dx_d = ank_mid_x - hip_mid_x
                            dy_d = ank_mid_y - hip_mid_y
                            norm_d = math.hypot(dx_d, dy_d)

                            if norm_u > 0 and norm_d > 0:
                                cos_u = (-dy_u) / norm_u  # vertical unit vector (0,-1)
                                cos_d = (-dy_d) / norm_d
                                cos_u = max(-1.0, min(1.0, cos_u))
                                cos_d = max(-1.0, min(1.0, cos_d))
                                theta_u = math.degrees(math.acos(cos_u))
                                theta_d = math.degrees(math.acos(cos_d))

                                # Check thresholds
                                if theta_u > 45 and theta_d > 60:
                                    plumb_condition_met = True
                                    theta_u_value = theta_u
                                    theta_d_value = theta_d

                                # Optionally display angles (commented for clarity)
                                cv2.putText(frame, f"θu:{theta_u:.0f} θd:{theta_d:.0f}", (10, 120),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1, cv2.LINE_AA)

                    # -------- NSAR (Normalized Shoulder-to-Ankle Ratio) -------- #
                    if person_idx == 1:
                        # Required landmark indices for NSAR computation
                        req_indices = [0, 11, 12, 27, 28]
                        if all(lm[i].visibility > 0.3 for i in req_indices):
                            nose = lm[0]
                            left_shoulder = lm[11]
                            right_shoulder = lm[12]
                            left_ankle = lm[27]
                            right_ankle = lm[28]

                            # Convert normalized coordinates to pixel distances
                            width_px = abs(left_shoulder.x - right_shoulder.x) * w
                            ankle_mid_y = (left_ankle.y + right_ankle.y) / 2
                            height_px = abs(nose.y - ankle_mid_y) * h

                            if height_px > 0:
                                nsar = width_px / height_px
                                nsar_value = nsar

                                # Initialize baseline using first valid ratio
                                if nsar_baseline is None:
                                    nsar_baseline = nsar
                                else:
                                    # Trigger fall warning if NSAR drops more than 30% from baseline
                                    if nsar < nsar_baseline * 0.7:
                                        status_text = "Fall Detected"
                                        nsar_message_display = "NSAR drops > 30% Fall Detected"

            # (Old trunk_angle_display block removed; handled by unified HUD now)

            # Show fall detection warning if triggered this frame
            if 'nsar_message_display' in locals():
                cv2.putText(frame, nsar_message_display, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                del nsar_message_display

            # Show trunk-angle-based fall alert if triggered this frame
            if 'trunk_fall_alert_display' in locals():
                cv2.putText(frame, "Trunk Angle > 45 % Fall Detected", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                del trunk_fall_alert_display

            # After processing persons, update plumb fall counter
            if plumb_condition_met:
                plumb_fall_counter += 1
            else:
                plumb_fall_counter = 0

            # If plumb condition persists but not yet fall threshold, mark as warning
            if plumb_condition_met and plumb_fall_counter < PLUMB_FALL_THRESHOLD_FRAMES and status_text == "Normal":
                status_text = "Warning"

            # Display mid-plumb fall alert if counter exceeds threshold
            if plumb_fall_counter >= PLUMB_FALL_THRESHOLD_FRAMES:
                status_text = "Fall Detected"
                cv2.putText(frame, "Mid-Plumb Angles Fall Detected", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Calculate abnormal duration in seconds if abnormal condition persists
            if plumb_condition_met:
                abnormal_sec = plumb_fall_counter / fps if fps > 0 else 0.0

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

            # Resize to fullscreen resolution for display
            display_frame = cv2.resize(frame, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_name, display_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release everything when finished
        cap.release()
        cv2.destroyAllWindows()
        pose.close()


if __name__ == "__main__":
    main() 