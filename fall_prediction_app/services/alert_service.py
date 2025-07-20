"""Alert service for fall detection notifications."""

import os
from typing import Optional
from telegram_sender import _send as telegram_send
from azure_foundry_predict import predict as foundry_predict

class AlertService:
    """Handles external alerts for fall detection."""
    
    def __init__(self):
        """Initialize alert service with environment variables."""
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.last_status = "Normal"
    
    def check_and_send_alerts(self, status: str, metrics: dict) -> Optional[str]:
        """Check if alerts should be sent and send them.
        
        Args:
            status: Current fall detection status
            metrics: Dictionary containing pose metrics
            
        Returns:
            Azure Foundry response if available, None otherwise
        """
        # Only send alerts when status changes from normal to fall detected
        if status == "Fall Detected" and self.last_status != "Fall Detected":
            return self._send_fall_alerts(metrics)
        
        self.last_status = status
        return None
    
    def _send_fall_alerts(self, metrics: dict) -> Optional[str]:
        """Send fall detection alerts to all configured services.
        
        Args:
            metrics: Dictionary containing pose metrics
            
        Returns:
            Azure Foundry response if available, None otherwise
        """
        foundry_response = None
        
        # Azure AI Foundry prediction
        prompt = self._build_foundry_prompt(metrics)
        print("[AzureFoundry] Prompt:", prompt)
        
        try:
            foundry_response = foundry_predict(prompt)
            print("[Azure Foundry]", foundry_response)
        except Exception as e:
            print(f"[AzureFoundry] Failed to get prediction: {e}")
        
        # Telegram alert
        self._send_telegram_alert(foundry_response)
        
        return foundry_response
    
    def _build_foundry_prompt(self, metrics: dict) -> str:
        """Build prompt for Azure Foundry prediction.
        
        Args:
            metrics: Dictionary containing pose metrics
            
        Returns:
            Formatted prompt string
        """
        return (
            f"Metrics: trunk_angle={metrics.get('trunk_angle')}, "
            f"nsar={metrics.get('nsar')}, theta_d={metrics.get('theta_d')}. "
            "Predict fall risk."
        )
    
    def _send_telegram_alert(self, foundry_response: Optional[str]) -> None:
        """Send Telegram alert with optional Foundry response.
        
        Args:
            foundry_response: Optional Azure Foundry prediction response
        """
        if not self.telegram_token or not self.telegram_chat_id:
            print("[Telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping alert.")
            return
        
        try:
            message = "Fall Risk Predicted!"
            if foundry_response:
                message += f"\n\nPrediction:\n{foundry_response}"
            
            telegram_send(self.telegram_token, int(self.telegram_chat_id), message)
        except Exception as e:
            print(f"[Telegram] Failed to send alert: {e}") 