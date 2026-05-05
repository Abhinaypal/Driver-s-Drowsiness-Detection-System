"""
Configuration for Driver Drowsiness Detection System
"""
from pathlib import Path
import os

# Project root - two levels up from src/config.py
PROJECT_ROOT = Path(__file__).parent.parent

# Dataset paths
DATASET_ROOT = PROJECT_ROOT / "Simuletic_DMS_Dataset"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

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
    "enable_audio": True,
    "enable_visual": True,
    "enable_sms": False,
    "sms_phone": None,
    "log_to_file": True,
    "alert_message": "Driver Drowsiness Alert!",
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "log_file": LOGS_DIR / "dms.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
