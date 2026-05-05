"""
Alert generation and notification system for drowsiness detection.

This module provides:
- AlertSystem: Generates and manages driver drowsiness alerts
- Multi-modal notifications (visual, audio, SMS)
- Alert logging and statistics
- Alert export for analysis

Alert channels:
- Visual: Console/terminal notifications
- Audio: System beep/sound alerts
- SMS: SMS notifications (requires service)
- File: Log file with timestamps and details
"""
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Generate and manage driver drowsiness alerts.
    
    This system processes inference results and generates multi-channel
    notifications when drowsiness is detected. It handles:
    - Alert generation from inference results
    - Multi-modal notifications (visual, audio, file, SMS)
    - Alert history and deduplication
    - Statistics and reporting
    
    Attributes:
        log_file: Path to alert log file
        enable_audio: Whether to enable audio alerts
        enable_visual: Whether to enable visual alerts
        enable_sms: Whether to enable SMS alerts
        sms_phone: Phone number for SMS alerts
        alert_count: Total alerts generated
        alert_history: List of all alerts generated
    """
    
    def __init__(self,
                 log_file: Optional[Path] = None,
                 enable_audio: bool = True,
                 enable_visual: bool = True,
                 enable_sms: bool = False,
                 sms_phone: Optional[str] = None):
        """
        Initialize the alert system with notification channels.
        
        Args:
            log_file: Path to log file for alerts (optional)
            enable_audio: Enable audio alerts (beep/sound)
            enable_visual: Enable visual alerts (console)
            enable_sms: Enable SMS alerts
            sms_phone: Phone number for SMS alerts
        """
        self.log_file = log_file
        self.enable_audio = enable_audio
        self.enable_visual = enable_visual
        self.enable_sms = enable_sms
        self.sms_phone = sms_phone
        
        # Alert tracking
        self.alert_count = 0          # Total alerts generated
        self.alert_history = []       # List of all alerts
    
    def process_inference(self, inference_result: Dict) -> Optional[Dict]:
        """
        Process inference result and generate alerts if needed
        
        Args:
            inference_result: Result from RealtimeInference.predict()
            
        Returns:
            Alert dictionary if triggered, None otherwise
        """
        if not inference_result.get('should_alert'):
            return None
        
        # Generate alert
        alert = {
            'timestamp': datetime.now().isoformat(),
            'class_id': inference_result['class_id'],
            'class_name': inference_result['class_name'],
            'confidence': inference_result['confidence'],
            'temporal_score': inference_result['temporal_score'],
            'reason': inference_result['reason'],
            'alert_number': self.alert_count + 1,
        }
        
        # Execute alert actions
        self._execute_alert(alert)
        
        # Log alert
        self.alert_count += 1
        self.alert_history.append(alert)
        
        return alert
    
    def _execute_alert(self, alert: Dict):
        """Execute all enabled alert actions"""
        if self.enable_visual:
            self._alert_visual(alert)
        
        if self.enable_audio:
            self._alert_audio(alert)
        
        if self.enable_sms:
            self._alert_sms(alert)
        
        # Always log
        self._log_alert(alert)
    
    def _alert_visual(self, alert: Dict):
        """Visual alert (console/display)"""
        message = self._format_alert_message(alert)
        logger.warning(f"\n{'='*60}\n{message}\n{'='*60}\n")
        print(f"\n{'🚨 '*10}")
        print(f"DROWSINESS ALERT #{alert['alert_number']}")
        print(f"{'🚨 '*10}\n")
        print(message)
    
    def _alert_audio(self, alert: Dict):
        """Audio alert (beep/sound)"""
        try:
            # Try to use system beep
            import winsound
            # Beep parameters: frequency (Hz), duration (ms)
            winsound.Beep(1000, 500)  # 1000 Hz for 500 ms
            winsound.Beep(1000, 500)  # Second beep
        except ImportError:
            logger.warning("winsound module not available for audio alerts")
        except Exception as e:
            logger.error(f"Error playing audio alert: {e}")
    
    def _alert_sms(self, alert: Dict):
        """SMS alert (placeholder)"""
        if not self.sms_phone:
            logger.warning("SMS phone number not configured")
            return
        
        message = self._format_alert_message(alert)
        logger.info(f"Would send SMS to {self.sms_phone}: {message}")
        # TODO: Integrate with SMS provider (Twilio, AWS SNS, etc.)
    
    def _log_alert(self, alert: Dict):
        """Log alert to file"""
        message = self._format_alert_message(alert)
        logger.error(f"ALERT: {message}")
        
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(f"{alert['timestamp']} - {message}\n")
            except Exception as e:
                logger.error(f"Error writing alert to log file: {e}")
    
    @staticmethod
    def _format_alert_message(alert: Dict) -> str:
        """Format alert message"""
        return (
            f"Drowsiness Alert #{alert['alert_number']}\n"
            f"Status: {alert['class_name']}\n"
            f"Confidence: {alert['confidence']:.2%}\n"
            f"Temporal Score: {alert['temporal_score']:.2%}\n"
            f"Reason: {alert['reason']}\n"
            f"Time: {alert['timestamp']}"
        )
    
    def get_statistics(self) -> Dict:
        """Get alert statistics"""
        return {
            'total_alerts': self.alert_count,
            'alerts_by_class': self._count_alerts_by_class(),
            'alert_history': self.alert_history,
        }
    
    def _count_alerts_by_class(self) -> Dict:
        """Count alerts by drowsiness class"""
        counts = {
            'Awake': 0,
            'Drowsy': 0,
            'Asleep': 0,
        }
        
        for alert in self.alert_history:
            class_name = alert['class_name']
            if class_name in counts:
                counts[class_name] += 1
        
        return counts
    
    def export_alerts(self, filepath: Path):
        """Export alert history to JSON"""
        import json
        try:
            with open(filepath, 'w') as f:
                json.dump(self.alert_history, f, indent=2)
            logger.info(f"Exported {len(self.alert_history)} alerts to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting alerts: {e}")
