# Audio Alert System Guide

## Overview

The Audio Alert System provides a sophisticated, multi-modal alert mechanism for driver drowsiness detection. When the system detects that a driver is drowsy or asleep, it can trigger:

- 🔊 **Audio alerts** (escalating intensity)
- 👁️ **Visual alerts** (console/display)
- 📝 **Logging** (permanent record)
- 📱 **SMS alerts** (optional)

---

## Alert Types

### 1. **Beep Alerts**
Soft, non-intrusive warning beeps

- **Beep Soft** (1000 Hz): First warning, minimal disruption
- **Beep Alert** (1500 Hz): Louder, more attention-getting

**Use case:** Light drowsiness or initial warnings

```
audio_files/beeps/
├── beep_soft_drowsy.wav    (500ms)
└── beep_alert_asleep.wav   (1000ms)
```

### 2. **Progressive Alarms**
Escalating alarm sounds with increasing urgency

- **Level 1** (800 Hz): Mild alarm, easy to wake to
- **Level 2** (1000 Hz): Medium intensity
- **Level 3** (1200 Hz): Loud, urgent
- **Emergency** (2000 Hz): Maximum urgency (ASLEEP)

**Use case:** Sustained drowsiness requiring stronger response

```
audio_files/alarms/
├── alarm_drowsy_level1.wav    (700ms)
├── alarm_drowsy_level2.wav    (900ms)
├── alarm_drowsy_level3.wav    (1100ms)
└── alarm_asleep_emergency.wav (2000ms)
```

### 3. **Voice Alerts**
Pre-recorded or synthesized voice messages

- **"Wake up, you're drowsy!"** - Clear warning message
- **"DANGER! You're asleep!"** - Critical emergency message
- **"Good. You're alert."** - Positive reinforcement (optional)

**Use case:** Clear, unambiguous communication

```
audio_files/voice/
├── voice_drowsy.wav        (2000ms)
├── voice_drowsy_strong.wav (2500ms)
└── voice_asleep.wav        (3000ms)
```

### 4. **Music Alerts**
Engaging, attention-grabbing music

- **Upbeat Music**: Energetic to wake driver
- **Siren/Rousing**: Police siren, march music
- **Shock Factor**: Unexpected sound to jolt awake

**Use case:** Maximum impact, memorable alert

```
audio_files/music/
├── music_alert_upbeat.wav (3000ms)
└── music_alert_siren.wav  (2500ms)
```

---

## Intensity Escalation

The system automatically escalates alert intensity based on repeated detections:

```
FIRST DETECTION:
┌─────────────────────────────┐
│  Soft Beep (Intensity=1)    │  🔊 (low volume)
│  1000 Hz, 500ms             │
│  Non-intrusive warning      │
└─────────────────────────────┘
         ↓ (if drowsiness continues)

REPEATED DETECTION:
┌─────────────────────────────┐
│  Medium Alarm (Intensity=2) │  🔊🔊 (medium volume)
│  1000 Hz, 900ms             │
│  Noticeable urgency         │
└─────────────────────────────┘
         ↓ (if drowsiness persists)

SUSTAINED PATTERN:
┌─────────────────────────────┐
│  Loud Voice (Intensity=3)   │  🔊🔊🔊 (high volume)
│  Voice message, 2500ms      │
│  Strong warning             │
└─────────────────────────────┘
         ↓ (if ASLEEP detected)

CRITICAL STATE:
┌─────────────────────────────┐
│  Emergency Alarm (Max)      │  🚨🚨🚨 (maximum)
│  2000 Hz siren, 2000ms      │
│  IMMEDIATE ACTION NEEDED    │
└─────────────────────────────┘
```

---

## Configuration

### Enable/Disable Audio Alerts

In `src/config.py`:

```python
ALERT_CONFIG = {
    'audio_enabled': True,        # Set to False to disable
    'alert_volume': 0.8,          # 0.0 (silent) to 1.0 (max)
    'audio_directory': 'audio_files/',
}
```

### Customize Alert Sounds

Map specific sounds to alert situations:

```python
ALERT_CONFIG = {
    'drowsy_sounds': {
        1: 'beeps/beep_soft_drowsy.wav',        # Level 1
        2: 'alarms/alarm_drowsy_level2.wav',    # Level 2
        3: 'voice/voice_drowsy_strong.wav',     # Level 3
    },
    'asleep_sounds': {
        3: 'alarms/alarm_asleep_emergency.wav', # Emergency
    },
}
```

### Volume Control

```python
# In your code:
audio_mgr = AudioManager()
audio_mgr.set_volume(0.5)  # 50% volume
audio_mgr.play_alert('drowsy', intensity=2)
```

---

## Using Audio Alerts

### Basic Usage

```python
from src.inference import AlertSystem
from src.audio import AudioManager

# Initialize
alert_system = AlertSystem(
    enable_audio=True,
    audio_dir='audio_files'
)

# Simulate drowsiness detection
features = {
    'perclos': 0.7,
    'eye_state': 'Drowsy',
    'head_pose': {'p': -5.0, 'y': 0.0, 'r': 0.0}
}

# Create inference result
result = {
    'class_id': 1,
    'class_name': 'Drowsy',
    'confidence': 0.85,
    'temporal_score': 0.75,
    'should_alert': True,
    'reason': 'High PERCLOS detected'
}

# Alert (with audio)
alert = alert_system.process_inference(result)
```

### With Real-Time Detection

```python
from src.inference import RuleBasedClassifier, RealtimeInference, AlertSystem
from src.preprocessing import FeatureExtractor

# Setup
classifier = RuleBasedClassifier()
inference = RealtimeInference(classifier)
alerts = AlertSystem(enable_audio=True)

# Process frame
attributes = {
    'eye_state': 'Drowsy',
    'perclos': 0.45,
    'head_pose': {'p': -3.0, 'y': 1.0, 'r': 0.0},
    'zone': 'Zone_On_Road'
}

features = FeatureExtractor.extract_all_features(attributes)
result = inference.predict(features)

# Automatic alert with audio
alert = alerts.process_inference(result)
```

---

## Audio Generation

### Synthetic Generation (for Testing)

```python
from src.audio import create_synthetic_alerts

# Generate synthetic audio files
create_synthetic_alerts('audio_files')
```

This creates placeholder audio files with different tones and durations.

### Text-to-Speech (for Voice Alerts)

```python
# Using gTTS library
from gtts import gTTS

text = "Wake up! You're getting drowsy!"
tts = gTTS(text=text, lang='en', slow=False)
tts.save('audio_files/voice/voice_drowsy.wav')
```

### Recording Custom Audio

1. Record high-quality audio files (WAV format)
2. Place in appropriate subdirectory
3. Update configuration in `src/config.py`

**Recommended tools:**
- Audacity (free, open-source)
- Adobe Audition (professional)
- GarageBand (macOS)

---

## Platform Support

### Windows
- ✅ Uses `winsound` for system beeps
- ✅ Uses `pygame` or `pydub` for audio files
- Default beep: Alert alert (440 Hz base frequency)

### macOS
- ✅ Uses `afplay` command
- ✅ Uses `pygame` or `pydub`
- Excellent audio support

### Linux
- ✅ Uses `aplay` or `play` command
- ✅ Uses `pygame` or `pydub`
- Requires ALSA or PulseAudio

### Fallback
If audio libraries unavailable:
- System beep via `print('\a')`
- Frequency and duration simulated via timing

---

## Alert Statistics

Monitor alert activity:

```python
stats = alert_system.get_statistics()

print(f"Total alerts: {stats['total_alerts']}")
print(f"By class: {stats['alerts_by_class']}")

# Output:
# Total alerts: 15
# By class: {'Awake': 0, 'Drowsy': 12, 'Asleep': 3}
```

---

## Logging and Export

### File Logging

All alerts automatically logged to:
```
logs/alerts.log
logs/dms.log
```

Format:
```
2026-05-05 14:32:15,123 - ALERT: Drowsiness Alert #1
Status: Drowsy/Microsleep
Confidence: 85%
Temporal Score: 75%
Reason: Sustained PERCLOS > 0.2
Time: 2026-05-05T14:32:15.123456
```

### Export to JSON

```python
alert_system.export_alerts(Path('alerts_report.json'))
```

---

## Testing and Debugging

### Test Audio Playback

```bash
python -c "
from src.audio import AudioManager
audio = AudioManager()
audio.test_all_alerts()
"
```

### Test Alert System

```bash
python demo_alert_system.py
```

Demonstrates:
- Audio file generation
- Alert system initialization
- Different alert types
- Intensity escalation
- Emergency alerts
- Real-time detection

### Check Available Audio Files

```python
from src.audio import AudioManager

audio = AudioManager()
alerts = audio.list_available_alerts()

for alert_type, files in alerts.items():
    print(f"{alert_type}: {files}")
```

---

## Best Practices

✅ **DO:**
- Use high-quality audio files (44.1kHz, 16-bit WAV)
- Test on target hardware
- Keep alert sounds concise (<3 seconds)
- Start with volume at 70-80%
- Include multiple alert types for variety
- Log all alerts for analysis

❌ **DON'T:**
- Use system beeps for critical alerts
- Make alerts too soft (driver might miss)
- Make alerts too long (disrupts driving)
- Use jarring sounds that startle driver
- Forget to test on embedded systems
- Disable audio alerts in production

---

## Troubleshooting

### No Audio Output

1. Check if audio is enabled:
   ```python
   alert_system.audio_manager  # Should not be None
   ```

2. Verify audio files exist:
   ```python
   ls audio_files/beeps/
   ls audio_files/voice/
   ```

3. Check volume setting:
   ```python
   audio_mgr.set_volume(1.0)  # Maximum volume
   ```

### Audio Cuts Off

- Ensure audio file duration is sufficient
- Check for threading issues (use `wait=True`)
- Verify audio format compatibility

### Latency Issues

- Pre-load common alerts on startup
- Use smaller audio files
- Disable visual alerts if needed
- Profile with `cProfile`

---

## Advanced Features

### Custom Audio Selection

```python
def get_custom_alert_sound(drowsiness_level, time_of_day):
    """Select audio based on context"""
    if time_of_day == 'night':
        # Louder alerts at night
        return 'alarms/alarm_level3.wav'
    elif drowsiness_level == 'high':
        return 'voice/voice_asleep.wav'
    else:
        return 'beeps/beep_soft.wav'
```

### Alert Chaining

```python
# Play multiple alerts in sequence
alerts = [
    ('beep', 1),
    ('alarm', 2),
    ('voice', 3)
]

for alert_type, intensity in alerts:
    audio_mgr.play_alert(alert_type, intensity=intensity, wait=True)
    time.sleep(0.5)
```

### Adaptive Alerts

```python
# Adapt alert based on driver history
if driver.alert_response_time < 2.0:
    # Driver responds quickly, use subtle alerts
    intensity = 1
else:
    # Driver responds slowly, use loud alerts
    intensity = 3
```

---

## License & Attribution

Audio files created using:
- `pydub` (MIT License)
- `pygame` (LGPL)
- `gTTS` (MIT License)

For production, use professionally-recorded or royalty-free audio from:
- Freesound.org
- Zapsplat.com
- BBC Sound Library
- Your own professional recordings

---

## References

- [WAV Audio Format Specs](https://en.wikipedia.org/wiki/WAV)
- [Audio Alert UX Best Practices](https://www.nngroup.com/articles/audio-usability/)
- [Beeping Guide](https://www.johndcook.com/blog/2016/02/23/keyboard-click-frequency/)
- [Driver Alert Sounds Research](https://www.nhtsa.gov/)

