"""Display service for rendering UI and HUD elements."""

import cv2
import ctypes
import numpy as np
from typing import Optional, List
from collections import deque

class DisplayService:
    """Handles all display and UI rendering."""
    
    def __init__(self, window_name: str = "Webcam Live Feed – Person Detection (press 'q' to quit)"):
        """Initialize display service.
        
        Args:
            window_name: Name of the display window
        """
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Make the window fullscreen for better visibility
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Get screen resolution
        self.screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        
        # FPS calculation
        self.prev_time = None
        self.fps = 0.0
        
        # Metrics buffer for time-series data
        self.metrics_buffer = deque(maxlen=60)  # store at most 60 sampled frames (~1 minute)
        self.last_buffer_time = None
    
    def update_fps(self, current_time: float) -> float:
        """Update and return current FPS.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Current FPS value
        """
        if self.prev_time is not None:
            self.fps = 1.0 / (current_time - self.prev_time) if current_time != self.prev_time else self.fps
        self.prev_time = current_time
        return self.fps
    
    def buffer_metrics(self, current_time: float, metrics: dict) -> None:
        """Buffer metrics for time-series analysis.
        
        Args:
            current_time: Current timestamp
            metrics: Dictionary containing pose metrics
        """
        if self.last_buffer_time is None or current_time - self.last_buffer_time >= 1.0:
            self.metrics_buffer.append({
                "timestamp": current_time,
                "trunk_angle": metrics.get("trunk_angle"),
                "nsar": metrics.get("nsar"),
                "theta_u": metrics.get("theta_u"),
                "theta_d": metrics.get("theta_d"),
            })
            self.last_buffer_time = current_time
    
    def draw_hud(self, frame: np.ndarray, metrics: dict, stats: dict) -> None:
        """Draw heads-up display with metrics and statistics.
        
        Args:
            frame: Frame to draw on
            metrics: Dictionary containing pose metrics
            stats: Dictionary containing statistics
        """
        base_x, base_y, line_h = 10, 20, 15
        font, scale, thick, color = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, (255, 255, 255)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (base_x, base_y), 
                   font, scale, color, thick, cv2.LINE_AA)
        base_y += line_h
        
        # Trunk angle
        if metrics.get("trunk_angle") is not None:
            cv2.putText(frame, f"Trunk Angle: {metrics['trunk_angle']:.1f}", 
                       (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
            base_y += line_h
        
        # NSAR
        if metrics.get("nsar") is not None:
            cv2.putText(frame, f"NSAR: {metrics['nsar']:.2f}", 
                       (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
            base_y += line_h
        
        # Plumb angles
        if metrics.get("theta_u") is not None and metrics.get("theta_d") is not None:
            cv2.putText(frame, f"θu: {metrics['theta_u']:.0f}  θd: {metrics['theta_d']:.0f}", 
                       (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
            base_y += line_h
        
        # Status
        status = metrics.get("status", "Normal")
        status_color = (0, 0, 255) if status != "Normal" else color
        cv2.putText(frame, f"Status: {status}", (base_x, base_y), 
                   font, scale, status_color, thick, cv2.LINE_AA)
        base_y += line_h
        
        # Abnormal duration
        if metrics.get("abnormal_sec", 0) > 0:
            cv2.putText(frame, f"Abnormal for: {metrics['abnormal_sec']:.1f}s", 
                       (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
            base_y += line_h
        
        # Detection statistics
        miss_rate = stats.get("miss_rate", 0.0)
        cv2.putText(frame, f"Miss Rate: {miss_rate:.1f}%", 
                   (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
        base_y += line_h
        
        # Latency statistics
        if stats.get("last_latency") is not None:
            cv2.putText(frame, f"Last Latency: {stats['last_latency']:.2f}s", 
                       (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
            base_y += line_h
        
        if stats.get("avg_latency") is not None:
            cv2.putText(frame, f"Avg Latency: {stats['avg_latency']:.2f}s", 
                       (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
            base_y += line_h
    
    def draw_fall_alerts(self, frame: np.ndarray, metrics: dict) -> None:
        """Draw fall detection alerts on frame.
        
        Args:
            frame: Frame to draw on
            metrics: Dictionary containing pose metrics
        """
        base_y = 60
        
        # NSAR fall alert
        if metrics.get("nsar_message_display"):
            cv2.putText(frame, metrics["nsar_message_display"], (10, base_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            base_y += 30
        
        # Trunk angle fall alert
        if metrics.get("trunk_fall_alert_display"):
            cv2.putText(frame, f"Trunk Angle > {metrics.get('trunk_angle_threshold', 33.8):.1f}° Fall Detected", 
                       (10, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            base_y += 30
        
        # Mid-plumb fall alert
        if metrics.get("plumb_fall_detected"):
            cv2.putText(frame, "Mid-Plumb Angles Fall Detected", (10, base_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    def draw_foundry_response(self, frame: np.ndarray, foundry_response: Optional[str]) -> None:
        """Draw Azure Foundry response on frame.
        
        Args:
            frame: Frame to draw on
            foundry_response: Optional Azure Foundry response text
        """
        if not foundry_response:
            return
        
        base_x, base_y, line_h = 10, 200, 15
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        color = (0, 255, 255)  # Yellow color for Foundry response
        
        wrapped = foundry_response.split("\n")
        for line in wrapped:
            cv2.putText(frame, line, (base_x, base_y), font, scale, color, thick, cv2.LINE_AA)
            base_y += line_h
    
    def display_frame(self, frame: np.ndarray) -> None:
        """Display the frame in fullscreen.
        
        Args:
            frame: Frame to display
        """
        # Resize to fullscreen resolution
        display_frame = cv2.resize(frame, (self.screen_w, self.screen_h), 
                                 interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.window_name, display_frame)
    
    def check_exit(self) -> bool:
        """Check if user pressed 'q' to exit.
        
        Returns:
            True if user wants to exit, False otherwise
        """
        return cv2.waitKey(1) & 0xFF == ord('q')
    
    def cleanup(self) -> None:
        """Clean up display resources."""
        cv2.destroyAllWindows() 