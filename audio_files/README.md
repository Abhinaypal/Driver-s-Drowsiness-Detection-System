# Audio Files Directory

This directory contains all audio alert files for the drowsiness detection system.

## Directory Structure

### `/beeps/`
Single warning beep sounds (short, non-intrusive)
- `beep_soft_drowsy.wav` - Soft single beep for drowsiness warning
- `beep_alert_asleep.wav` - Louder beep for asleep alert
- `beep_system.wav` - System notification beep

### `/voice/`
Voice alerts (TTS or recorded)
- `voice_drowsy.wav` - "Hey! You're getting drowsy. Stay alert!"
- `voice_drowsy_strong.wav` - "WAKE UP! You're drowsy!"
- `voice_asleep.wav` - "DANGER! You're asleep!"
- `voice_awake.wav` - "Good. You're alert."

### `/alarms/`
Progressive alarm sounds (escalating intensity)
- `alarm_drowsy_level1.wav` - Mild progressive alarm
- `alarm_drowsy_level2.wav` - Medium progressive alarm
- `alarm_drowsy_level3.wav` - Loud progressive alarm
- `alarm_asleep_emergency.wav` - Emergency alarm for asleep state

### `/music/`
Music/song alerts (engaging alerts)
- `music_alert_upbeat.wav` - Energetic music to wake up
- `music_alert_rousing.wav` - Rousing march-like music
- `music_alert_siren.wav` - Police siren (attention grabber)

## Audio Specifications

All audio files should ideally be:
- **Format:** WAV (uncompressed, highest quality)
- **Sample Rate:** 44.1 kHz or 48 kHz
- **Channels:** Mono or Stereo
- **Bit Depth:** 16-bit
- **Duration:** 
  - Beeps: 0.5-1.0 seconds
  - Voice alerts: 2-4 seconds
  - Alarms: 1-2 seconds
  - Music: 2-5 seconds

## Usage

The system automatically selects appropriate audio based on:
1. **Alert Level:** Drowsy vs Asleep
2. **Intensity Progression:** Escalates from soft to loud
3. **User Preference:** Configurable in `ALERT_CONFIG`

## Audio Generation

You can generate audio files using:
1. **Python `pydub`**: Generate synthetic beeps/alerts
2. **Text-to-Speech (TTS):** Use `gTTS` or `pyttsx3` for voice alerts
3. **Free online tools:** Freesound.org, Zapsplat
4. **Your own recordings:** Record professional alerts

### Generate Beep Example (Python)

```python
from pydub 
import AudioSegment
from pydub.generators import Sine
from pydub.utils import mediainfo

# Generate 1000Hz sine wave for 0.5 seconds
beep = Sine(1000).to_audio_segment(duration=500)
beep.export("beep_soft_drowsy.wav", format="wav")
```

### Generate Voice Alert (Python)

```python
from gtts import gTTS

# Generate voice alert
text = "Hey! You're getting drowsy. Stay alert!"
tts = gTTS(text=text, lang='en', slow=False)
tts.save("voice_drowsy.wav")
```

## Configuration

In `src/config.py`, set your preferred alerts:

```python
ALERT_CONFIG = {
    'audio_enabled': True,
    'alert_type': 'multi',  # 'beep', 'voice', 'alarm', 'music', 'multi'
    'alert_volume': 0.8,    # 0.0 - 1.0
    'alert_directory': 'audio_files/',
    'drowsy_sounds': {
        'level_1': 'beeps/beep_soft_drowsy.wav',
        'level_2': 'alarms/alarm_drowsy_level2.wav',
        'level_3': 'voice/voice_drowsy_strong.wav',
    },
    'asleep_sounds': {
        'emergency': 'alarms/alarm_asleep_emergency.wav',
        'backup': 'voice/voice_asleep.wav',
    }
}
```

## Customization

### Add Custom Audio Files

1. Create your audio files (WAV format recommended)
2. Place in appropriate subdirectory
3. Update `ALERT_CONFIG` in `src/config.py`
4. System will automatically use them

### Disable Audio

```python
# In src/config.py
ALERT_CONFIG = {
    'audio_enabled': False,  # Disable all audio alerts
}
```

### Volume Control

```python
# In src/config.py
ALERT_CONFIG = {
    'alert_volume': 0.5,  # 50% volume (0.0-1.0)
}
```

## Testing

Test audio playback:

```bash
python -c "
from src.audio import AudioManager
audio = AudioManager()
audio.play_alert('drowsy', intensity=2)
"
```

## Cross-Platform Compatibility

- **Windows:** Uses `winsound` (built-in)
- **macOS:** Uses `afplay` (built-in)
- **Linux:** Uses `play` from SoX or `aplay` (ALSA)
- **Fallback:** `pydub` with `pygame` or `simpleaudio`

See `src/audio/audio_manager.py` for implementation details.

