"""
Demonstration of the Audio Alert System.

Shows:
1. Audio file generation
2. Alert system initialization
3. Different alert types and intensities
4. Real-time drowsiness detection with audio alerts
5. Alert statistics and logging
"""
import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import LABELS_DIR, IMAGES_DIR, LOGS_DIR
from src.data import AnnotationLoader, DatasetBuilder
from src.preprocessing import ImageProcessor, FeatureExtractor
from src.inference import RuleBasedClassifier, RealtimeInference, AlertSystem
from src.audio import AudioManager, create_synthetic_alerts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_audio_generation():
    """Demonstrate synthetic audio file generation."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: AUDIO FILE GENERATION")
    logger.info("=" * 80)
    
    logger.info("\nGenerating synthetic audio files for testing...")
    logger.info("(In production, use high-quality professional audio files)")
    
    try:
        create_synthetic_alerts("audio_files")
        logger.info("✓ Audio files generated successfully")
        
        # List generated files
        audio_mgr = AudioManager()
        available = audio_mgr.list_available_alerts()
        
        logger.info(f"\nAvailable audio alerts:")
        for alert_type, files in available.items():
            logger.info(f"  {alert_type}/:")
            for file in files:
                logger.info(f"    - {file}")
        
        return audio_mgr
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return None


def demonstrate_alert_system():
    """Demonstrate alert system with audio playback."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: ALERT SYSTEM INITIALIZATION")
    logger.info("=" * 80)
    
    # Create alert system
    alert_system = AlertSystem(
        log_file=LOGS_DIR / "alerts.log",
        enable_audio=True,
        enable_visual=True,
        enable_sms=False,
        audio_dir="audio_files"
    )
    
    logger.info("✓ Alert system initialized")
    logger.info(f"  - Audio alerts: {'Enabled' if alert_system.audio_manager else 'Disabled'}")
    logger.info(f"  - Visual alerts: Enabled")
    logger.info(f"  - Log file: {alert_system.log_file}")
    
    return alert_system


def demonstrate_alert_types(alert_system: AlertSystem):
    """Demonstrate different alert types."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: ALERT TYPE DEMONSTRATION")
    logger.info("=" * 80)
    
    if not alert_system.audio_manager:
        logger.warning("Audio manager not available, skipping audio demos")
        return
    
    # Test different alert types
    demo_alerts = [
        ('Soft Beep', 'beep', 1),
        ('Medium Alarm', 'drowsy', 2),
        ('Loud Voice Alert', 'drowsy', 3),
        ('Emergency Alarm', 'asleep', 3),
    ]
    
    logger.info("\nPlaying different alert types...")
    logger.info("Listen to the variations in intensity and urgency:\n")
    
    for name, alert_type, intensity in demo_alerts:
        logger.info(f"► Playing: {name} (type={alert_type}, intensity={intensity})")
        
        # Simulate alert
        test_alert = {
            'alert_number': 1,
            'timestamp': time.time(),
            'class_name': 'Drowsy' if 'drowsy' in alert_type else 'Asleep',
            'class_id': 1 if 'drowsy' in alert_type else 2,
            'confidence': 0.85,
            'temporal_score': 0.75,
            'reason': f"Test: {name}"
        }
        
        alert_system._alert_audio(test_alert)
        time.sleep(2.5)  # Wait between alerts


def demonstrate_intensity_escalation(alert_system: AlertSystem):
    """Demonstrate alert intensity escalation."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: INTENSITY ESCALATION DEMONSTRATION")
    logger.info("=" * 80)
    
    logger.info("\nSimulating sustained drowsiness with escalating alerts...")
    logger.info("Notice how alerts get progressively more urgent:\n")
    
    # Simulate consecutive drowsy detections
    for detection_num in range(1, 4):
        logger.info(f"[Detection {detection_num}/3] Driver appears drowsy...")
        
        # Create alert
        test_alert = {
            'alert_number': detection_num,
            'timestamp': time.time(),
            'class_name': 'Drowsy',
            'class_id': 1,
            'confidence': 0.70 + detection_num * 0.05,
            'temporal_score': 0.60 + detection_num * 0.10,
            'reason': f"Sustained drowsiness detected (frame {detection_num})"
        }
        
        # Visual alert
        if alert_system.enable_visual:
            logger.warning(f"  ⚠️  ALERT #{detection_num}: Drowsiness detected!")
        
        # Audio alert (escalates in intensity)
        alert_system._alert_audio(test_alert)
        
        logger.info(f"  Intensity level: {min(3, detection_num)}/3")
        time.sleep(1.5)


def demonstrate_emergency_alert(alert_system: AlertSystem):
    """Demonstrate emergency alert for asleep state."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: EMERGENCY ALERT (ASLEEP STATE)")
    logger.info("=" * 80)
    
    logger.info("\n⛔ CRITICAL: Driver is asleep! Maximum urgency alert:\n")
    
    # Emergency alert
    emergency_alert = {
        'alert_number': 99,
        'timestamp': time.time(),
        'class_name': 'Asleep',
        'class_id': 2,
        'confidence': 0.98,
        'temporal_score': 0.95,
        'reason': 'EMERGENCY: Driver is asleep!'
    }
    
    # Visual alert
    if alert_system.enable_visual:
        logger.error("\n" + "=" * 80)
        logger.error("🚨 EMERGENCY ALERT 🚨")
        logger.error("DRIVER IS ASLEEP - IMMEDIATE ACTION REQUIRED")
        logger.error("=" * 80)
    
    # Audio alert (maximum intensity)
    alert_system._alert_audio(emergency_alert)


def demonstrate_with_real_data(alert_system: AlertSystem):
    """Demonstrate alerts with real dataset."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: REAL-TIME DETECTION WITH AUDIO ALERTS")
    logger.info("=" * 80)
    
    try:
        # Load data
        logger.info("\nLoading dataset...")
        loader = AnnotationLoader(LABELS_DIR)
        builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR, loader)
        dataset = builder.get_dataset()
        
        if not dataset:
            logger.warning("No dataset samples available")
            return
        
        # Initialize inference
        classifier = RuleBasedClassifier()
        inference_engine = RealtimeInference(classifier, sequence_length=3)
        
        logger.info(f"Loaded {len(dataset)} samples. Processing with alert system...\n")
        
        # Process samples
        num_samples = min(5, len(dataset))
        alert_count = 0
        
        for idx in range(num_samples):
            sample = dataset[idx]
            
            # Extract features
            attributes = sample['attributes']
            features = FeatureExtractor.extract_all_features(attributes)
            
            # Get inference
            result = inference_engine.predict(features)
            
            logger.info(f"\nSample {idx + 1}: {Path(sample['image_path']).name}")
            logger.info(f"  Predicted: {result['class_name']} ({result['confidence']:.0%})")
            logger.info(f"  True class: {sample['class_name']}")
            
            # Generate alert if needed
            if result.get('should_alert'):
                logger.info(f"  → ⚠️  Alert triggered!")
                
                # Create alert
                alert = {
                    'alert_number': alert_count + 1,
                    'timestamp': time.time(),
                    'class_name': result['class_name'],
                    'class_id': result['class_id'],
                    'confidence': result['confidence'],
                    'temporal_score': result.get('temporal_score', 0.0),
                    'reason': 'Drowsiness detected via inference engine'
                }
                
                # Execute alert
                alert_system._alert_visual(alert)
                alert_system._alert_audio(alert)
                alert_count += 1
            
            time.sleep(0.5)
        
        # Report
        logger.info(f"\n✓ Processed {num_samples} samples, {alert_count} alerts triggered")
        
    except Exception as e:
        logger.error(f"Error in real-time demo: {e}")


def print_configuration():
    """Print alert system configuration."""
    logger.info("\n" + "=" * 80)
    logger.info("AUDIO ALERT SYSTEM CONFIGURATION")
    logger.info("=" * 80)
    
    config_info = """
Alert Types:
  • Beep (Soft): Single warning beep (1000 Hz, 500ms)
  • Beep (Alert): Louder alert beep (1500 Hz, 1000ms)
  • Alarm (Level 1): Mild progressive alarm
  • Alarm (Level 2): Medium progressive alarm
  • Alarm (Level 3): Loud progressive alarm
  • Voice (Drowsy): "Wake up, you're drowsy!"
  • Voice (Asleep): "DANGER! You're asleep!"
  • Emergency: Maximum intensity siren

Intensity Levels:
  Level 1: Soft, non-intrusive (first detection)
  Level 2: Medium, noticeable (repeated detection)
  Level 3: Loud, urgent (sustained pattern)

Alert Directory Structure:
  audio_files/
  ├── beeps/           (warning beeps)
  ├── voice/           (voice alerts)
  ├── alarms/          (progressive alarms)
  └── music/           (music alerts)

Configuration in src/config.py:
  ALERT_CONFIG = {
      'audio_enabled': True,
      'alert_type': 'multi',
      'alert_volume': 0.8,  # 0-1.0
      'alert_directory': 'audio_files/',
  }

Platform Support:
  • Windows: winsound + pygame/pydub
  • macOS: afplay + pygame/pydub
  • Linux: aplay + pygame/pydub
    """
    
    logger.info(config_info)


def main():
    """Run full demonstration."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  AUDIO ALERT SYSTEM FOR DRIVER DROWSINESS DETECTION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Print configuration
    print_configuration()
    
    # Step 1: Generate audio files
    audio_mgr = demonstrate_audio_generation()
    
    # Step 2: Initialize alert system
    alert_system = demonstrate_alert_system()
    
    # Step 3: Demonstrate alert types
    if audio_mgr:
        demonstrate_alert_types(alert_system)
    
    # Step 4: Demonstrate intensity escalation
    if audio_mgr:
        demonstrate_intensity_escalation(alert_system)
    
    # Step 5: Demonstrate emergency alert
    if audio_mgr:
        demonstrate_emergency_alert(alert_system)
    
    # Step 6: Real-time with alerts
    demonstrate_with_real_data(alert_system)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ALERT SYSTEM SUMMARY")
    logger.info("=" * 80)
    
    stats = alert_system.get_statistics()
    logger.info(f"\nTotal alerts triggered: {stats['total_alerts']}")
    logger.info(f"Alert breakdown: {stats['alerts_by_class']}")
    
    logger.info("\n✓ Demonstration complete!")
    logger.info("Check 'logs/alerts.log' for full alert history")
    
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("""
1. Replace synthetic audio files with high-quality production audio:
   - Record professional voice alerts
   - Use royalty-free alarm sounds
   - Customize for your use case

2. Configure alert preferences in src/config.py:
   - Set alert volume level
   - Choose preferred alert types
   - Customize alert thresholds

3. Integrate with real camera feed:
   - Use OpenCV for video capture
   - Process frames in real-time
   - Trigger alerts based on detected drowsiness

4. Deploy on vehicle hardware:
   - Test on embedded systems (Raspberry Pi, Jetson)
   - Ensure reliable audio output
   - Monitor system performance

5. Customize alerts for your vehicle:
   - Add vehicle-specific sounds
   - Integrate with car's audio system
   - Add SMS/notification options
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
