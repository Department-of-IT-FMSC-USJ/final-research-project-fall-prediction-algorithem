"""Object detection using YOLOv8."""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Set, Tuple, Optional

class ObjectDetector:
    """Handles object detection using YOLOv8."""
    
    # Indoor object classes for detection
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
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        """Initialize YOLO object detector.
        
        Args:
            model_path: Path to YOLO model file
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.model.fuse()  # fuse Conv+BN for faster inference
        self.conf_threshold = conf_threshold
        
        # Filter allowed classes to those present in the model
        self.allowed_classes = {
            cls for cls in self.INDOOR_CLASSES 
            if cls in self.model.names.values()
        }
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List, List]:
        """Detect objects in frame and separate person detections.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (person_boxes, other_objects) where each is a list of detections
        """
        # Run YOLO inference
        results = self.model(
            frame, 
            conf=self.conf_threshold, 
            verbose=False, 
            show=False
        )[0]
        
        person_boxes = []
        other_objects = []
        
        for det_box in results.boxes:
            cls_id = int(det_box.cls[0])
            class_name = self.model.names[cls_id]
            conf = float(det_box.conf[0])
            
            # Filter by confidence and allowed classes
            if conf < self.conf_threshold or class_name not in self.allowed_classes:
                continue
            
            if class_name == "person":
                person_boxes.append(det_box)
            else:
                other_objects.append((det_box, class_name, conf))
        
        return person_boxes, other_objects
    
    def draw_objects(self, frame: np.ndarray, other_objects: List) -> None:
        """Draw bounding boxes for non-person objects.
        
        Args:
            frame: Frame to draw on
            other_objects: List of (det_box, class_name, conf) tuples
        """
        for det_box, class_name, conf in other_objects:
            x1, y1, x2, y2 = map(int, det_box.xyxy[0])
            
            # Draw bounding box (cyan)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                frame, label, (x1, max(y1 - 10, 0)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA
            )
    
    def draw_person_boxes(self, frame: np.ndarray, person_boxes: List) -> None:
        """Draw bounding boxes for person detections.
        
        Args:
            frame: Frame to draw on
            person_boxes: List of person detection boxes
        """
        for person_idx, box in enumerate(person_boxes, start=1):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {person_idx}"
            cv2.putText(
                frame, label, (x1, max(y1 - 10, 0)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA
            ) 