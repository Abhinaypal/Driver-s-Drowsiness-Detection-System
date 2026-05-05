"""
Audio management system for drowsiness alerts.

Handles audio playback across platforms (Windows, macOS, Linux).
Supports multiple audio types: beeps, voice, alarms, music.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, List
import threading
import time

logger = logging.getLogger(__name__)


class AudioManager:
    """
    Cross-platform audio playback for drowsiness alerts.
    
    Features:
    - Supports multiple audio file formats (WAV, MP3, OGG)
    - Cross-platform (Windows, macOS, Linux)
    - Volume control
    - Threading for non-blocking playback
    - Alert intensity levels (1-3)
    - Multiple alert types
    """
    
    def __init__(self, audio_dir: str = "audio_files"):
        """
        Initialize AudioManager.
        
        Args:
            audio_dir: Directory containing audio files
        """
        self.audio_dir = Path(audio_dir)
        self.platform = sys.platform
        self.volume = 0.8
        self.playback_thread = None
        self.is_playing = False
        
        # Initialize playback backend
        self.backend = self._init_backend()
        
        logger.info(f"AudioManager initialized on {self.platform} using {self.backend}")
    
    def _init_backend(self) -> str:
        """
        Initialize appropriate audio backend for platform.
        
        Returns:
            Name of the audio backend being used
        """
        try:
            # Try pygame first (most reliable cross-platform)
            import pygame
            pygame.mixer.init()
            return "pygame"
        except Exception as e:
            logger.debug(f"pygame not available: {e}")
        
        try:
            # Try simpleaudio (lightweight)
            import simpleaudio
            return "simpleaudio"
        except Exception as e:
            logger.debug(f"simpleaudio not available: {e}")
        
        try:
            # Try pydub (most flexible)
            import pydub.playback
            return "pydub"
        except Exception as e:
            logger.debug(f"pydub not available: {e}")
        
        # Fallback to OS-specific commands
        if self.platform == "win32":
            return "winsound"
        elif self.platform == "darwin":
            return "afplay"
        else:
            return "aplay"
    
    def set_volume(self, volume: float):
        """
        Set playback volume (0.0 - 1.0).
        
        Args:
            volume: Volume level (0.0 = silent, 1.0 = max)
        """
        self.volume = max(0.0, min(1.0, volume))
        
        if self.backend == "pygame":
            import pygame
            pygame.mixer.music.set_volume(self.volume)
    
    def play_alert(self, alert_type: str, intensity: int = 1, wait: bool = False):
        """
        Play an alert sound.
        
        Args:
            alert_type: Type of alert ('drowsy', 'asleep', 'beep', 'voice', 'alarm', 'music')
            intensity: Alert intensity level (1=soft, 2=medium, 3=loud)
            wait: If True, block until audio finishes
        """
        try:
            # Get audio file path
            audio_file = self._get_audio_file(alert_type, intensity)
            
            if not audio_file or not audio_file.exists():
                logger.warning(f"Audio file not found: {audio_file}")
                # Fallback to system beep
                self._system_beep(intensity)
                return
            
            # Log alert
            logger.info(f"Playing alert: {alert_type} (intensity={intensity}) from {audio_file}")
            
            # Play audio
            if wait:
                self._play_blocking(audio_file)
            else:
                self._play_threaded(audio_file)
        
        except Exception as e:
            logger.error(f"Error playing alert: {e}")
            self._system_beep(intensity)
    
    def _get_audio_file(self, alert_type: str, intensity: int) -> Optional[Path]:
        """
        Get audio file path based on alert type and intensity.
        
        Args:
            alert_type: Type of alert
            intensity: Alert intensity (1-3)
        
        Returns:
            Path to audio file, or None if not found
        """
        # Map alert types to directories
        alert_map = {
            'beep': 'beeps/beep_soft_drowsy.wav',
            'beep_asleep': 'beeps/beep_alert_asleep.wav',
            'drowsy_soft': 'beeps/beep_soft_drowsy.wav',
            'drowsy_medium': 'alarms/alarm_drowsy_level2.wav',
            'drowsy_loud': 'voice/voice_drowsy_strong.wav',
            'asleep': 'voice/voice_asleep.wav',
            'asleep_emergency': 'alarms/alarm_asleep_emergency.wav',
            'voice': 'voice/voice_drowsy.wav',
            'alarm_level1': 'alarms/alarm_drowsy_level1.wav',
            'alarm_level2': 'alarms/alarm_drowsy_level2.wav',
            'alarm_level3': 'alarms/alarm_drowsy_level3.wav',
            'music': 'music/music_alert_upbeat.wav',
        }
        
        # Map intensity levels
        if alert_type == 'drowsy':
            file_key = ['drowsy_soft', 'drowsy_medium', 'drowsy_loud'][min(intensity - 1, 2)]
            audio_file = alert_map.get(file_key)
        elif alert_type == 'asleep':
            audio_file = alert_map.get('asleep_emergency' if intensity >= 3 else 'asleep')
        else:
            audio_file = alert_map.get(alert_type)
        
        if audio_file:
            return self.audio_dir / audio_file
        
        return None
    
    def _play_blocking(self, audio_file: Path):
        """Play audio and block until finished."""
        try:
            if self.backend == "pygame":
                import pygame
                sound = pygame.mixer.Sound(str(audio_file))
                sound.set_volume(self.volume)
                sound.play()
                # Wait for playback to finish
                while pygame.mixer.get_busy():
                    time.sleep(0.1)
            
            elif self.backend == "simpleaudio":
                import simpleaudio
                wave_obj = simpleaudio.WaveObject.from_wave_file(str(audio_file))
                play_obj = wave_obj.play()
                play_obj.wait_done()
            
            elif self.backend == "pydub":
                from pydub import AudioSegment
                from pydub.playback import play
                sound = AudioSegment.from_wav(str(audio_file))
                play(sound)
            
            elif self.backend == "winsound":
                import winsound
                winsound.PlaySound(str(audio_file), winsound.SND_FILENAME)
            
            else:
                self._play_with_os_command(audio_file)
        
        except Exception as e:
            logger.error(f"Error playing audio with {self.backend}: {e}")
    
    def _play_threaded(self, audio_file: Path):
        """Play audio in background thread (non-blocking)."""
        if self.is_playing:
            logger.debug("Audio already playing, skipping")
            return
        
        def play_audio():
            self.is_playing = True
            try:
                self._play_blocking(audio_file)
            finally:
                self.is_playing = False
        
        thread = threading.Thread(target=play_audio, daemon=True)
        thread.start()
        self.playback_thread = thread
    
    def _play_with_os_command(self, audio_file: Path):
        """Fallback: play audio using OS command."""
        try:
            if self.platform == "darwin":
                os.system(f"afplay '{audio_file}'")
            else:  # Linux
                os.system(f"play '{audio_file}' 2>/dev/null")
        except Exception as e:
            logger.error(f"Error using OS command: {e}")
    
    def _system_beep(self, intensity: int = 1):
        """
        Fallback system beep when audio files unavailable.
        
        Args:
            intensity: Beep intensity (1-3) - affects frequency
        """
        try:
            if self.backend == "winsound" or self.platform == "win32":
                import winsound
                # Intensity affects frequency and duration
                freq = 440 * intensity  # Hz
                duration = 200 * intensity  # ms
                winsound.Beep(freq, duration)
            else:
                # Unix-like system
                print('\a', end='', flush=True)  # ASCII bell
                if intensity > 1:
                    time.sleep(0.2)
                    print('\a', end='', flush=True)
                if intensity > 2:
                    time.sleep(0.2)
                    print('\a', end='', flush=True)
        except Exception as e:
            logger.debug(f"System beep failed: {e}")
    
    def stop_playback(self):
        """Stop current audio playback."""
        try:
            if self.backend == "pygame":
                import pygame
                pygame.mixer.stop()
            self.is_playing = False
            logger.info("Audio playback stopped")
        except Exception as e:
            logger.debug(f"Error stopping playback: {e}")
    
    def list_available_alerts(self) -> Dict[str, List[str]]:
        """
        List all available audio alert files.
        
        Returns:
            Dictionary of alert types and their audio files
        """
        alerts = {}
        
        if not self.audio_dir.exists():
            logger.warning(f"Audio directory not found: {self.audio_dir}")
            return alerts
        
        for subdir in self.audio_dir.iterdir():
            if subdir.is_dir():
                files = [f.name for f in subdir.glob("*.wav") if f.is_file()]
                if files:
                    alerts[subdir.name] = files
        
        return alerts
    
    def test_all_alerts(self):
        """Test all available alerts (demonstration)."""
        logger.info("Testing all available audio alerts...")
        
        alerts_available = self.list_available_alerts()
        
        if not alerts_available:
            logger.warning("No audio files found")
            return
        
        for alert_type, files in alerts_available.items():
            logger.info(f"\nTesting {alert_type}:")
            for i, file in enumerate(files[:1], 1):  # Test first file of each type
                logger.info(f"  Playing: {file}")
                self.play_alert(alert_type, intensity=1, wait=True)
                time.sleep(0.5)


def create_synthetic_alerts(output_dir: str = "audio_files"):
    """
    Create synthetic alert audio files using pydub.
    
    This generates beep, alarm, and voice samples for testing.
    
    Args:
        output_dir: Directory to save audio files
    """
    try:
        from pydub.generators import Sine, WhiteNoise
        from pydub.utils import mediainfo
    except ImportError:
        logger.warning("pydub not available. Install with: pip install pydub")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Creating synthetic audio files in {output_path}")
    
    # Beep sounds
    beep_dir = output_path / "beeps"
    beep_dir.mkdir(exist_ok=True)
    
    # Soft beep (1000 Hz, 500ms)
    beep_soft = Sine(1000).to_audio_segment(duration=500)
    beep_soft.export(str(beep_dir / "beep_soft_drowsy.wav"), format="wav")
    logger.info("✓ Created: beep_soft_drowsy.wav")
    
    # Alert beep (1500 Hz, 1000ms)
    beep_alert = Sine(1500).to_audio_segment(duration=1000)
    beep_alert.export(str(beep_dir / "beep_alert_asleep.wav"), format="wav")
    logger.info("✓ Created: beep_alert_asleep.wav")
    
    # Alarm sounds
    alarm_dir = output_path / "alarms"
    alarm_dir.mkdir(exist_ok=True)
    
    # Progressive alarms with multiple tones
    for level in range(1, 4):
        freq = 800 + level * 200
        duration = 500 + level * 200
        alarm = Sine(freq).to_audio_segment(duration=duration)
        alarm.export(str(alarm_dir / f"alarm_drowsy_level{level}.wav"), format="wav")
        logger.info(f"✓ Created: alarm_drowsy_level{level}.wav")
    
    # Emergency alarm
    emergency = Sine(2000).to_audio_segment(duration=2000)
    emergency.export(str(alarm_dir / "alarm_asleep_emergency.wav"), format="wav")
    logger.info("✓ Created: alarm_asleep_emergency.wav")
    
    # Voice alert placeholders (with tone)
    voice_dir = output_path / "voice"
    voice_dir.mkdir(exist_ok=True)
    
    for voice_type, freq in [("voice_drowsy.wav", 800), ("voice_asleep.wav", 1200)]:
        voice = Sine(freq).to_audio_segment(duration=2000)
        voice.export(str(voice_dir / voice_type), format="wav")
        logger.info(f"✓ Created: {voice_type}")
    
    # Music alert placeholder
    music_dir = output_path / "music"
    music_dir.mkdir(exist_ok=True)
    
    music = Sine(1200).to_audio_segment(duration=3000)
    music.export(str(music_dir / "music_alert_upbeat.wav"), format="wav")
    logger.info("✓ Created: music_alert_upbeat.wav")
    
    logger.info("\n✓ All synthetic audio files created successfully!")
    logger.info("Note: These are placeholder files. Replace with high-quality audio for production.")


if __name__ == "__main__":
    # Test audio manager
    logging.basicConfig(level=logging.INFO)
    
    print("Testing AudioManager...")
    audio_mgr = AudioManager()
    
    # Create synthetic alerts if available
    try:
        create_synthetic_alerts()
    except Exception as e:
        print(f"Could not create synthetic alerts: {e}")
    
    # Test playback
    audio_mgr.test_all_alerts()
