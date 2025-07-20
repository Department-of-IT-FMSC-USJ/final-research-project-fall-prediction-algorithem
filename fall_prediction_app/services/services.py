"""
Service layer for Fall Prediction System.
Following the Single Responsibility Principle by separating business logic from web layer.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os


@dataclass
class FallMetrics:
    """Data class for fall detection metrics."""
    fps: Optional[float] = None
    trunk_angle: Optional[float] = None
    nsar: Optional[float] = None
    theta_u: Optional[float] = None
    theta_d: Optional[float] = None
    fall_detected: bool = False
    prediction: Optional[str] = None
    timestamp: Optional[datetime] = None


class MetricsService:
    """Service for handling metrics data."""
    
    def __init__(self, history_size: int = 120):
        self.history_size = history_size
        self._metrics_history: list[FallMetrics] = []
    
    def add_metrics(self, metrics: FallMetrics) -> None:
        """Add new metrics to history."""
        metrics.timestamp = datetime.now()
        self._metrics_history.append(metrics)
        
        # Keep only the last N entries
        if len(self._metrics_history) > self.history_size:
            self._metrics_history = self._metrics_history[-self.history_size:]
    
    def get_latest_metrics(self) -> Optional[FallMetrics]:
        """Get the most recent metrics."""
        return self._metrics_history[-1] if self._metrics_history else None
    
    def get_metrics_history(self) -> list[Dict[str, Any]]:
        """Get metrics history for charts."""
        return [
            {
                "ts": int(metric.timestamp.timestamp()) if metric.timestamp else 0,
                "fps": metric.fps,
                "trunk_angle": metric.trunk_angle,
                "nsar": metric.nsar,
                "theta_u": metric.theta_u,
                "theta_d": metric.theta_d,
                "fall_detected": metric.fall_detected,
                "prediction": metric.prediction,
            }
            for metric in self._metrics_history
        ]
    
    def clear_history(self) -> None:
        """Clear metrics history."""
        self._metrics_history.clear()


class SettingsService:
    """Service for handling application settings."""
    
    def __init__(self, settings_file: str = "settings.json"):
        self.settings_file = settings_file
        self._settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file or return defaults."""
        default_settings = {
            "camera_enabled": True,
            "share_analytics": True,
            "show_personal_data": False,
            "telegram_token": "",
            "telegram_chat_id": "",
            "telegram_phone": "",
        }
        
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    return {**default_settings, **json.load(f)}
            except (json.JSONDecodeError, IOError):
                pass
        
        return default_settings
    
    def _save_settings(self) -> None:
        """Save settings to file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self._settings, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save settings: {e}")
    
    def get_settings(self) -> Dict[str, Any]:
        """Get all current settings."""
        return self._settings.copy()
    
    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """Update settings with validation and persistence."""
        for key, value in new_settings.items():
            if key in self._settings:
                if key in {"camera_enabled", "share_analytics", "show_personal_data"}:
                    self._settings[key] = self._parse_bool(value)
                else:
                    self._settings[key] = str(value)
        
        self._save_settings()
    
    @staticmethod
    def _parse_bool(value: str | bool | None) -> bool:
        """Parse boolean values safely."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).lower() in {"1", "true", "yes", "on"}


class NotificationService:
    """Service for handling notifications (Telegram, etc.)."""
    
    def __init__(self, settings_service: SettingsService):
        self.settings_service = settings_service
    
    def send_fall_alert(self, metrics: FallMetrics) -> bool:
        """Send fall detection alert."""
        settings = self.settings_service.get_settings()
        
        if not settings.get("telegram_token") or not settings.get("telegram_chat_id"):
            return False
        
        # TODO: Implement Telegram notification
        # This would integrate with the existing telegram_sender.py
        message = f"🚨 Fall Detected!\n"
        message += f"Time: {metrics.timestamp}\n"
        message += f"Trunk Angle: {metrics.trunk_angle:.1f}°\n"
        message += f"NSAR: {metrics.nsar:.3f}\n"
        message += f"Prediction: {metrics.prediction}"
        
        print(f"Would send notification: {message}")
        return True
    
    def send_daily_report(self, metrics_summary: Dict[str, Any]) -> bool:
        """Send daily analytics report."""
        settings = self.settings_service.get_settings()
        
        if not settings.get("share_analytics"):
            return False
        
        # TODO: Implement daily report
        print(f"Would send daily report: {metrics_summary}")
        return True 