"""Camera service for handling video capture."""

import cv2
import numpy as np
from typing import List, Union, Tuple, Optional

class CameraService:
    """Handles camera initialization and frame capture."""
    
    def __init__(self, sources: str = "0", multi_view: bool = False):
        """Initialize camera service.
        
        Args:
            sources: Comma-separated list of video sources
            multi_view: Whether to enable multi-camera processing
        """
        self.sources = self._parse_sources(sources)
        self.multi_view = multi_view and len(self.sources) > 1
        self.caps = []
        self._initialize_cameras()
    
    def _parse_sources(self, sources: str) -> List[Union[int, str]]:
        """Convert a comma-separated source string to a list of typed sources.
        
        Args:
            sources: Comma-separated string of sources
            
        Returns:
            List of source identifiers
        """
        result: List[Union[int, str]] = []
        for token in sources.split(","):
            token = token.strip()
            if not token:
                continue
            result.append(int(token) if token.isdigit() else token)
        return result or [0]
    
    def _initialize_cameras(self) -> None:
        """Initialize camera captures."""
        if self.multi_view:
            for src in self.sources:
                cap_stream = cv2.VideoCapture(src)
                if not cap_stream.isOpened():
                    raise RuntimeError(f"Could not open video source '{src}'.")
                self.caps.append(cap_stream)
                # Minimize internal buffer to reduce latency
                cap_stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.sources[0])
            if not cap.isOpened():
                raise RuntimeError(
                    "Could not open webcam. Make sure a camera is connected and not "
                    "used by another application."
                )
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.caps = [cap]
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera(s).
        
        Returns:
            Tuple of (success, frame) where frame is None if unsuccessful
        """
        if self.multi_view:
            frames: List[np.ndarray] = []
            valid = True
            
            for cap_stream in self.caps:
                ret_i, frame_i = cap_stream.read()
                if not ret_i:
                    valid = False
                    break
                frames.append(frame_i)
            
            if not valid or not frames:
                print("Failed to grab frame from one of the cameras.")
                return False, None
            
            # For now, use the first frame (placeholder for future fusion)
            frame = frames[0]
        else:
            ret, frame = self.caps[0].read()
            if not ret:
                print("Failed to grab frame from webcam.")
                return False, None
        
        return True, frame
    
    def fuse_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Placeholder multi-view fusion strategy.
        
        Args:
            frames: List of frames from multiple cameras
            
        Returns:
            Fused frame
        """
        # For the initial implementation, simply returns the first frame.
        # This stub exists so that future work can combine detections from 
        # multiple viewpoints to mitigate occlusions and varying perspectives.
        return frames[0]
    
    def release(self) -> None:
        """Release all camera resources."""
        for cam in self.caps:
            cam.release() 