"""Statistics service for tracking performance metrics."""

import time
from typing import List, Dict, Optional

class StatisticsService:
    """Handles statistics tracking and latency calculations."""
    
    def __init__(self):
        """Initialize statistics service."""
        self.total_frames = 0
        self.missed_detection_frames = 0
        self.fall_event_latencies: List[float] = []
        self.current_fall_start: Optional[float] = None
        self.last_latency: Optional[float] = None
    
    def update_frame_count(self, has_persons: bool) -> None:
        """Update frame statistics.
        
        Args:
            has_persons: Whether persons were detected in the frame
        """
        self.total_frames += 1
        if not has_persons:
            self.missed_detection_frames += 1
    
    def update_latency_tracking(self, status: str, current_time: float) -> None:
        """Update latency tracking for fall detection.
        
        Args:
            status: Current fall detection status
            current_time: Current timestamp
        """
        # Record the first frame where warning begins
        if status == "Warning" and self.current_fall_start is None:
            self.current_fall_start = current_time
        
        # Finalize when fall is confirmed
        if status == "Fall Detected" and self.current_fall_start is not None:
            latency = current_time - self.current_fall_start
            self.fall_event_latencies.append(latency)
            self.last_latency = latency
            self.current_fall_start = None
    
    def get_statistics(self) -> Dict:
        """Get current statistics.
        
        Returns:
            Dictionary containing all current statistics
        """
        miss_rate = (self.missed_detection_frames / self.total_frames * 100) if self.total_frames else 0.0
        
        stats = {
            "miss_rate": miss_rate,
            "last_latency": self.last_latency,
        }
        
        if self.fall_event_latencies:
            stats["avg_latency"] = sum(self.fall_event_latencies) / len(self.fall_event_latencies)
        
        return stats
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.total_frames = 0
        self.missed_detection_frames = 0
        self.fall_event_latencies.clear()
        self.current_fall_start = None
        self.last_latency = None 