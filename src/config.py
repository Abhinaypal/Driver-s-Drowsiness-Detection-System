"""
Configuration for Driver Drowsiness Detection System
"""
from pathlib import Path
import os

# Project root - two levels up from src/config.py
PROJECT_ROOT = Path(__file__).parent.parent

# Dataset paths
VIDEO_DATASET_ROOT = PROJECT_ROOT / "video dataset"
OLD_IMAGE_DATASET_ROOT = PROJECT_ROOT / "Simuletic_DMS_Dataset"

DATASET_ROOT = VIDEO_DATASET_ROOT
VIDEOS_DIR = VIDEO_DATASET_ROOT
IMAGES_DIR = OLD_IMAGE_DATASET_ROOT / "images"
VIDEO_LABELS_DIR = VIDEO_DATASET_ROOT / "labels"
OLD_LABELS_DIR = OLD_IMAGE_DATASET_ROOT / "labels"

LABELS_DIRS = [path for path in [VIDEO_LABELS_DIR, OLD_LABELS_DIR] if path.exists()]
LABELS_DIR = LABELS_DIRS[0] if LABELS_DIRS else VIDEO_LABELS_DIR

# Output directories
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

# Create directories if they don't exist
CHECKPOINTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)
MANIFESTS_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "input_size": 224,
    "sequence_length": 10,  # Number of frames to consider
    "batch_size": 16,
    "num_classes": 3,  # Alert, Drowsy, Awake
    "learning_rate": 1e-3,
    "epochs": 100,
    "device": "cuda",  # or "cpu"
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    "resize_size": 224,
    "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet stats
    "normalize_std": [0.229, 0.224, 0.225],
    "frame_skip": 5,  # Process every 5th frame in video
}

# Inference configuration
INFERENCE_CONFIG = {
    "drowsiness_threshold": 0.6,
    "alert_threshold": 0.8,
    "confidence_threshold": 0.7,
    "alert_cooldown_seconds": 5,
    "frame_buffer_size": 30,
    "output_fps": 30,
}

# Drowsiness levels
CLASS_LABELS = {
    0: "Awake",
    1: "Drowsy/Microsleep",
    2: "Asleep",
}

# Data split ratios
DATA_SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

# Feature extraction config
FEATURE_CONFIG = {
    "extract_perclos": True,
    "extract_head_pose": True,
    "extract_eye_state": True,
    "perclos_threshold": 0.2,  # Closed eyes percentage threshold
}

# Alert configuration
ALERT_CONFIG = {
    # General settings
    "enable_audio": True,
    "enable_visual": True,
    "enable_sms": False,
    "log_to_file": True,
    "sms_phone": None,
    
    # Audio alert settings
    "audio_directory": str(PROJECT_ROOT / "audio_files"),
    "audio_enabled": True,
    "alert_type": "multi",  # 'beep', 'voice', 'alarm', 'music', 'multi'
    "alert_volume": 0.8,    # 0.0 (silent) to 1.0 (max)
    
    # Alert message
    "alert_message": "Driver Drowsiness Alert!",
    
    # Alert sounds by intensity level
    "drowsy_sounds": {
        1: "beeps/beep_soft_drowsy.wav",      # Soft beep (first detection)
        2: "alarms/alarm_drowsy_level2.wav",  # Medium alarm (repeated)
        3: "voice/voice_drowsy_strong.wav",   # Loud voice alert (sustained)
    },
    "asleep_sounds": {
        3: "alarms/alarm_asleep_emergency.wav",  # Emergency alarm (critical)
    },
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "log_file": LOGS_DIR / "dms.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
