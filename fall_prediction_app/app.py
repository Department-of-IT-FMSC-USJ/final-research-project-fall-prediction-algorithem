from __future__ import annotations

import argparse
import sys
import os

# Add parent directory to path to import fall_stream
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from flask import Flask, Response, render_template, jsonify, request
from fall_stream import stream_frames_flask, get_latest_stats, get_metrics_history

from config import get_config
from services import SettingsService, MetricsService, NotificationService


class FallPredictionApp:
    """Main application class following Single Responsibility Principle."""
    
    def __init__(self, config_name: str = "default"):
        self.config = get_config(config_name)
        self.app = Flask(__name__)
        self.app.config.from_object(self.config)
        
        # Initialize services
        self.settings_service = SettingsService()
        self.metrics_service = MetricsService(history_size=self.config.METRICS_HISTORY_SIZE)
        self.notification_service = NotificationService(self.settings_service)
        
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup all application routes."""
        # Main pages
        self.app.route("/")(self.index)
        self.app.route("/analytics")(self.analytics)
        self.app.route("/settings")(self.settings_page)
        
        # API endpoints
        self.app.route("/video_feed")(self.video_feed)
        self.app.route("/stats")(self.stats)
        self.app.route("/timeline")(self.timeline)
        self.app.route("/api/settings", methods=["GET", "POST"])(self.api_settings)
    
    def index(self):
        """Render the main dashboard page."""
        return render_template("index.html")
    
    def analytics(self):
        """Render the analytics page."""
        return render_template("analytics.html")
    
    def settings_page(self):
        """Render the settings page."""
        return render_template("settings.html")
    
    def video_feed(self) -> Response:
        """Stream video feed."""
        return Response(
            stream_frames_flask(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    
    def stats(self) -> Response:
        """Return latest statistics as JSON."""
        # For now, use the existing fall_stream functions
        # TODO: Integrate with MetricsService
        return jsonify(get_latest_stats())
    
    def timeline(self) -> Response:
        """Return metric history for charts."""
        # For now, use the existing fall_stream functions
        # TODO: Integrate with MetricsService
        return jsonify(get_metrics_history())
    
    def api_settings(self) -> Response:
        """Handle settings API requests."""
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            self.settings_service.update_settings(data)
            return jsonify({"status": "ok", **self.settings_service.get_settings()})
        return jsonify(self.settings_service.get_settings())
    
    def run(self, host: str = None, port: int = None, debug: bool = None) -> None:
        """Run the Flask application."""
        host = host or self.config.HOST
        port = port or self.config.PORT
        debug = debug if debug is not None else self.config.DEBUG
        
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
        except OSError as e:
            print(f"[ERROR] Failed to bind to port {port}")
            print(f" » {e}")
            print("Hint: Another process may be using the port or your firewall/antivirus is blocking it.")
            print("Try a different port, e.g.: python app.py --port 6060")


def main():
    """Main entry point following Single Responsibility Principle."""
    parser = argparse.ArgumentParser(description="Run Fall Prediction Flask dashboard")
    parser.add_argument("--host", help="Host to bind (default from config)")
    parser.add_argument("--port", type=int, help="Port to bind (default from config)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", default="default", choices=["development", "production", "testing"], 
                       help="Configuration to use")
    args = parser.parse_args()

    app = FallPredictionApp(config_name=args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main() 