# Driver Drowsiness Detection System

A production-ready architecture for detecting driver drowsiness in real-time using computer vision and machine learning.

## 🏗️ Architecture Overview

```
Input (Camera/Video)
         ↓
    Data Pipeline (Load, validate, cache)
         ↓
    Preprocessing (Normalize, resize, augment)
         ↓
    Feature Extraction (PERCLOS, head pose, eye state)
         ↓
    ML Model (CNN/LSTM for classification)
         ↓
    Inference Engine (Real-time predictions, temporal reasoning)
         ↓
    Alert System (Notifications, logging)
         ↓
    Output (Alerts, dashboards, logs)
```

## 📁 Project Structure

```
Driver Drowsiness Detection system/
├── ARCHITECTURE.md                    # Detailed architecture documentation
├── generate_manifest.py               # Create flat CSV/JSON dataset manifests
├── main.py                           # Main demonstration script
├── predict_image.py                  # Single-image CNN inference
├── requirements.txt                  # Project dependencies
├── train_model.py                    # CNN training entrypoint
├── config.py                         # System configuration
│
├── Simuletic_DMS_Dataset/            # Dataset
│   ├── images/                       # Face images
│   ├── labels/                       # JSON annotations
│   └── visualizations/               # Visualization outputs
│
└── src/                              # Source code
    ├── config.py                     # Configuration
    │
    ├── data/                         # Data loading
    │   ├── dataset.py               # Dataset and annotation loaders
    │   ├── manifest.py              # Flat manifest generation/loading
    │   └── __init__.py
    │
    ├── preprocessing/                # Preprocessing & features
    │   ├── feature_extractor.py     # Image processor, feature extraction
    │   └── __init__.py
    │
    ├── models/                       # ML models
    │   ├── cnn.py                   # CNN image classifier
    │   └── __init__.py
    │
    ├── training/                     # Model training
    │   ├── image_dataset.py         # PyTorch Dataset/DataLoader adapter
    │   ├── trainer.py               # Training/evaluation loop
    │   └── __init__.py
    │
    └── inference/                    # Inference & alerts
        ├── classifier.py            # Classification and prediction
        ├── alert_system.py          # Alert generation
        └── __init__.py
```

## 🚀 Getting Started

### 1. Installation

```bash
# Clone or navigate to project directory
cd "Driver Drowsiness Detection system"

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `src/config.py` to customize:

```python
# Model parameters
MODEL_CONFIG = {
    "input_size": 224,
    "sequence_length": 10,
    "batch_size": 16,
}

# Inference thresholds
INFERENCE_CONFIG = {
    "drowsiness_threshold": 0.6,
    "alert_threshold": 0.8,
    "alert_cooldown_seconds": 5,
}
```

### 3. Run Demonstration

```bash
# Run the main demonstration
python main.py
```

This will:
- ✅ Load and explore the dataset
- ✅ Demonstrate preprocessing pipeline
- ✅ Show inference and classification
- ✅ Generate sample alerts

## 📊 Data Pipeline

### Loading Dataset

```python
from src.data import AnnotationLoader, DatasetBuilder
from src.config import LABELS_DIR, IMAGES_DIR

# Load annotations
loader = AnnotationLoader(LABELS_DIR)

# Build complete dataset
builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR, loader)
dataset = builder.get_dataset()

# Split for train/val/test
train, val, test = builder.split_dataset()
```

### Generate Manifest

```bash
python generate_manifest.py --format csv
```

This creates a reusable flat manifest in `manifests/dataset_manifest.csv` with:
`image_path`, `video_id`, `class_id`, `class_label`, `timestamp`, `perclos`,
`eye_state`, `zone`, and head pose columns.

### Dataset Statistics

```python
# Get statistics
stats = loader.get_statistics()
print(f"Total samples: {stats['total_samples']}")
print(f"Classes: {stats['classes']}")
```

## 🔧 Preprocessing

### Image Processing

```python
from src.preprocessing import ImageProcessor

processor = ImageProcessor(
    target_size=(224, 224),
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Preprocess single image
processed = processor.preprocess(image_path)
```

### Feature Extraction

```python
from src.preprocessing import FeatureExtractor, TemporalFeatureExtractor

# Extract static features from annotations
features = FeatureExtractor.extract_all_features(attributes)
# Returns: perclos, eye_state, head_pose, zone

# Extract temporal features
temporal = TemporalFeatureExtractor.extract_temporal_features(
    perclos_values=[0.2, 0.3, 0.4, 0.5],
    eye_states=[0, 0, 1, 1]
)
```

## 🎯 Classification & Inference

### Rule-Based Classifier (Quick Start)

```python
from src.inference import RuleBasedClassifier, RealtimeInference

# Initialize classifier
classifier = RuleBasedClassifier(
    perclos_drowsy_threshold=0.2,
    perclos_asleep_threshold=0.8
)

# Initialize inference engine
inference = RealtimeInference(
    classifier,
    sequence_length=10,
    alert_threshold=0.6,
    alert_cooldown_seconds=5
)

# Make prediction
result = inference.predict(features)
# Returns: class_id, confidence, temporal_score, should_alert
```

### Result Structure

```python
result = {
    'class_id': 0,                          # 0=awake, 1=drowsy, 2=asleep
    'class_name': 'Awake',
    'confidence': 0.95,                     # 0-1 confidence score
    'temporal_score': 0.3,                  # Based on frame buffer
    'should_alert': False,
    'timestamp': '2024-05-04T10:30:00',
    'reason': 'Driver is awake (confidence: 0.95)'
}
```

## 🚨 Alert System

```python
from src.inference import AlertSystem

alert_system = AlertSystem(
    log_file=Path('alerts.log'),
    enable_audio=True,
    enable_visual=True,
    enable_sms=False
)

# Process inference results
alert = alert_system.process_inference(result)

if alert:
    print(f"Alert triggered: {alert['reason']}")

# Get statistics
stats = alert_system.get_statistics()
print(f"Total alerts: {stats['total_alerts']}")
```

## 🔄 Complete Pipeline Example

```python
from pathlib import Path
from src.config import LABELS_DIR, IMAGES_DIR
from src.data import DatasetBuilder
from src.preprocessing import ImageProcessor, FeatureExtractor
from src.inference import RuleBasedClassifier, RealtimeInference, AlertSystem

# 1. Load data
builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR)
dataset = builder.get_dataset()

# 2. Initialize components
processor = ImageProcessor()
classifier = RuleBasedClassifier()
inference = RealtimeInference(classifier)
alerts = AlertSystem()

# 3. Process samples
for sample in dataset:
    # Preprocess image
    image = processor.preprocess(Path(sample['image_path']))
    
    # Extract features
    features = FeatureExtractor.extract_all_features(sample['attributes'])
    
    # Inference
    result = inference.predict(features)
    
    # Alert
    alert = alerts.process_inference(result)
    if alert:
        print(f"⚠️  {alert['reason']}")
```

## 📈 Model Training

Train the image-based CNN classifier:

```bash
python train_model.py --epochs 10 --batch-size 16 --device auto
```

This will load the dataset, convert images into normalized PyTorch tensors, train
`SimpleDrowsinessCNN`, save the best checkpoint to `checkpoints/drowsiness_cnn.pt`,
and print validation/test metrics.

Quick smoke test:

```bash
python train_model.py --epochs 1 --batch-size 8 --device cpu
```

Train from a generated manifest:

```bash
python train_model.py --manifest manifests/dataset_manifest.csv --epochs 10 --batch-size 16 --device auto
```

Run prediction on a single image with the saved checkpoint:

```bash
python predict_image.py Simuletic_DMS_Dataset/images/driver_no_sleep_f5.jpg --device cpu
```

## 🧪 Testing

Run unit tests:

```bash
python test_architecture.py
```

## 📊 Monitoring & Logging

Logs are stored in:
- `logs/dms.log` - Main application log
- `logs/alerts.log` - Alert log
- `predictions/` - Prediction outputs

View logs:

```bash
tail -f logs/dms.log
```

## 🔐 Production Considerations

### Deployment
- [ ] Docker containerization
- [ ] Model quantization (ONNX, TensorRT)
- [ ] Load testing and benchmarking
- [ ] Security and authentication

### Monitoring
- [ ] Performance metrics
- [ ] Model drift detection
- [ ] Alert escalation
- [ ] Dashboard integration

### Scalability
- [ ] Multi-GPU support
- [ ] Distributed inference
- [ ] Database for alert history
- [ ] API for integration

## 📝 Configuration Reference

See `src/config.py` for all configurable parameters:

```python
MODEL_CONFIG          # Model architecture parameters
PREPROCESSING_CONFIG  # Image normalization settings
INFERENCE_CONFIG      # Prediction thresholds
ALERT_CONFIG          # Alert channel settings
FEATURE_CONFIG        # Feature extraction settings
```

## 🎓 Learning Resources

1. **PERCLOS (Percentage of Eyelid Closure)**
   - Key metric for drowsiness detection
   - Higher values = more closed eyes = more drowsy

2. **Temporal Features**
   - Blink frequency patterns
   - Eye closure duration trends
   - State transition analysis

3. **Head Pose Estimation**
   - Pitch, yaw, roll angles
   - Nodding detection
   - Face orientation changes

## 🐛 Troubleshooting

### Image not loading
```
Check: IMAGES_DIR path and image file existence
```

### Module import errors
```bash
# Reinstall in development mode
pip install -e .
```

### Annotation mismatches
```python
# Verify annotation keys
from src.data import AnnotationLoader
loader = AnnotationLoader(LABELS_DIR)
print(loader.get_all_keys())
```

## 📞 Support

For issues or questions:
1. Check `ARCHITECTURE.md` for detailed design
2. Review example code in `main.py`
3. Check logs in `logs/dms.log`

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Simuletic DMS Dataset v5.1
- OpenCV for computer vision
- PyTorch/TensorFlow community
