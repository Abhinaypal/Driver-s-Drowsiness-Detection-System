"""
Alert generation and notification system for drowsiness detection.

This module provides:
- AlertSystem: Generates and manages driver drowsiness alerts
- Multi-modal notifications (visual, audio, SMS)
- Alert logging and statistics
- Alert export for analysis

Alert channels:
- Visual: Console/terminal notifications
- Audio: Multiple audio alert types (beep, voice, alarm, music) with intensity levels
- SMS: SMS notifications (requires service)
- File: Log file with timestamps and details

Audio Alert Types:
- Drowsy: Soft beep → Medium alarm → Loud voice alert (escalates with intensity)
- Asleep: Loud voice alert → Emergency alarm (critical safety)
"""
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

# Import audio manager
try:
    from src.audio import AudioManager
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Generate and manage driver drowsiness alerts with multi-modal notifications.
    
    This system processes inference results and generates multi-channel
    notifications when drowsiness is detected. It handles:
    - Alert generation from inference results
    - Multi-modal notifications (visual, audio, file, SMS)
    - Alert history and deduplication
    - Statistics and reporting
    - Audio alert escalation (intensity levels 1-3)
    
    Attributes:
        log_file: Path to alert log file
        enable_audio: Whether to enable audio alerts
        enable_visual: Whether to enable visual alerts
        enable_sms: Whether to enable SMS alerts
        audio_manager: AudioManager instance for audio playback
        sms_phone: Phone number for SMS alerts
        alert_count: Total alerts generated
        alert_history: List of all alerts generated
        last_class: Last detected class for intensity escalation
    """
    
    def __init__(self,
                 log_file: Optional[Path] = None,
                 enable_audio: bool = True,
                 enable_visual: bool = True,
                 enable_sms: bool = False,
                 audio_dir: str = "audio_files",
                 sms_phone: Optional[str] = None):
        """
        Initialize the alert system with notification channels.
        
        Args:
            log_file: Path to log file for alerts (optional)
            enable_audio: Enable audio alerts (requires audio files)
            enable_visual: Enable visual alerts (console)
            enable_sms: Enable SMS alerts
            audio_dir: Directory containing audio files for alerts
            sms_phone: Phone number for SMS alerts
        """
        self.log_file = log_file
        self.enable_audio = enable_audio
        self.enable_visual = enable_visual
        self.enable_sms = enable_sms
        self.sms_phone = sms_phone
        
        # Initialize audio manager
        if self.enable_audio and AUDIO_AVAILABLE:
            try:
                self.audio_manager = AudioManager(audio_dir=audio_dir)
                self.audio_manager.set_volume(0.8)  # 80% volume
                logger.info("✓ AudioManager initialized")
            except Exception as e:
                logger.warning(f"Could not initialize AudioManager: {e}")
                self.audio_manager = None
        else:
            self.audio_manager = None
        
        # Alert tracking
        self.alert_count = 0          # Total alerts generated
        self.alert_history = []       # List of all alerts
        self.last_class = None        # For intensity escalation
        self.consecutive_alerts = {}  # Count consecutive alerts per class
    
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
        """
        Audio alert with intensity escalation.
        
        Strategy:
        - First drowsy alert: Soft beep (intensity 1)
        - Repeated drowsy: Medium alarm (intensity 2)
        - Sustained drowsy: Loud voice alert (intensity 3)
        - Asleep: Emergency alarm (maximum intensity)
        """
        if not self.audio_manager:
            return
        
        try:
            class_name = alert['class_name']
            class_id = alert['class_id']
            
            # Calculate intensity based on consecutive alerts
            if class_name != self.last_class:
                # New class detected, reset counter
                self.consecutive_alerts[class_name] = 1
                self.last_class = class_name
                intensity = 1
            else:
                # Same class, escalate intensity
                self.consecutive_alerts[class_name] = self.consecutive_alerts.get(class_name, 0) + 1
                intensity = min(3, self.consecutive_alerts[class_name])
            
            # Select alert based on drowsiness level
            if class_id == 1:  # Drowsy
                self.audio_manager.play_alert('drowsy', intensity=intensity, wait=False)
                logger.info(f"Audio alert: Drowsy (intensity={intensity})")
            
            elif class_id == 2:  # Asleep
                self.audio_manager.play_alert('asleep', intensity=3, wait=False)
                logger.info(f"Audio alert: Asleep (EMERGENCY)")
            
        except Exception as e:
            logger.error(f"Error playing audio alert: {e}")
            # Fallback to system beep
            self._system_beep_fallback(alert)
    
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
    
    def _system_beep_fallback(self, alert: Dict):
        """Fallback system beep when audio files unavailable."""
        try:
            import winsound
            # Escalate beep frequency with intensity
            class_id = alert['class_id']
            intensity = self.consecutive_alerts.get(alert['class_name'], 1)
            
            freq = 800 + class_id * 300 + intensity * 200
            duration = 300 + intensity * 200
            
            winsound.Beep(freq, duration)
        except ImportError:
            logger.debug("winsound not available for fallback beep")
        except Exception as e:
            logger.debug(f"Error with fallback beep: {e}")
    
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
    
    def generate_audio_files(self, output_dir: str = "audio_files"):
        """
        Generate synthetic audio files for testing.
        
        Args:
            output_dir: Directory to save audio files
        """
        if not AUDIO_AVAILABLE:
            logger.warning("Cannot generate audio files: audio module not available")
            return False
        
        try:
            from src.audio import create_synthetic_alerts
            create_synthetic_alerts(output_dir)
            self.audio_manager = AudioManager(audio_dir=output_dir)
            logger.info(f"✓ Generated audio files in {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Error generating audio files: {e}")
            return False
