"""
Configuration settings for the Fall Prediction System.
Following the Single Responsibility Principle by separating configuration from application logic.
"""

from typing import Dict, Any


class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = "your-secret-key-here"
    DEBUG = False
    TESTING = False
    
    # Default application settings
    DEFAULT_SETTINGS = {
        "camera_enabled": True,
        "share_analytics": True,
        "show_personal_data": False,
        "telegram_token": "",
        "telegram_chat_id": "",
        "telegram_phone": "",
    }
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 5050
    
    # Video stream settings
    VIDEO_FPS = 30
    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480
    
    # Analytics settings
    METRICS_HISTORY_SIZE = 120  # 2 minutes at 1 FPS
    CHART_UPDATE_INTERVAL = 1000  # milliseconds


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SECRET_KEY = "dev-secret-key"


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = "prod-secret-key-change-this"


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SECRET_KEY = "test-secret-key"


# Configuration mapping
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config(config_name: str = "default") -> Config:
    """Get configuration class by name."""
    return config.get(config_name, config["default"]) 