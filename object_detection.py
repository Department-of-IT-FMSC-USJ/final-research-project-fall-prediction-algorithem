"""YOLOv8 Object Detection (Common Indoor Objects)

This script uses a pretrained YOLOv8 model (trained on the COCO dataset) to detect
common indoor objects such as 'person', 'chair', 'bed', 'tv', 'bottle', 'laptop',
and others in real time. It works with either a webcam feed or a video file.

Usage (command line):
    python object_detection.py                 # Uses default webcam (index 0)
    python object_detection.py --source 1      # Use webcam with index 1
    python object_detection.py --source path/to/video.mp4

Press the 'q' key in the display window to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Set

import cv2  # type: ignore
from ultralytics import YOLO  # type: ignore


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source. Use an integer for webcam index (e.g., '0') or a path to a video file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to a YOLOv8 weights file trained on COCO (e.g., yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for displaying bounding boxes.",
    )
    return parser.parse_args()


def _open_source(src: str | int) -> cv2.VideoCapture:
    """Open a webcam or video file source."""
    cap = cv2.VideoCapture(src)  # type: ignore[arg-type]
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source '{src}'.")
    return cap


def _get_allowed_classes(model: YOLO) -> Set[str]:
    """Return a set of COCO class names typically found indoors.

    The returned set is used to filter detections to improve readability, but the
    list can be modified as needed.
    """
    indoor_classes = {
        "person",
        "chair",
        "couch",  # "sofa" in some label sets – YOLOv8 uses "couch"
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
    # Verify that the requested classes actually exist in model.names; fall back
    # to the intersection to avoid KeyError for custom models.
    return {cls for cls in indoor_classes if cls in model.names.values()}


def _draw_detections(
    frame,
    results,
    allowed_classes: Set[str],
    class_names,
    conf_threshold: float,
) -> None:  # type: ignore[override]
    """Draw bounding boxes and labels for allowed detections on the frame."""
    boxes = results.boxes
    # Iterate through detections; each io element corresponds to a single prediction
    for xyxy, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        class_name = class_names[int(cls_id)]

        # Filter by allowed classes and confidence
        if allowed_classes and class_name not in allowed_classes:
            continue
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def main() -> None:  # noqa: D401
    """Run the object detection loop."""
    args = _parse_args()

    # Determine the appropriate video source type (int for webcam index, str for path)
    if args.source.isdigit():
        source: int | str = int(args.source)
    else:
        source_str: str = args.source  # Explicit str for type checker
        # Expand relative path for clarity
        source_path = Path(source_str).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Video file '{source_str}' not found.")
        source = str(source_path)

    # Load model (COCO-pretrained weights by default)
    model = YOLO(args.model)
    model.fuse()  # Performance optimisation (merges Conv & BN layers)

    # Allowed classes for display (editable)
    allowed_classes = _get_allowed_classes(model)

    cap = _open_source(source)

    window_name = "YOLOv8 Object Detection (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video / camera failure

            # Run inference (results list has length 1)
            results = model(frame, conf=args.conf, verbose=False, show=False)[0]

            # Draw detections
            _draw_detections(frame, results, allowed_classes, model.names, args.conf)

            # FPS calculation
            current_time = time.perf_counter()
            fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
            prev_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 