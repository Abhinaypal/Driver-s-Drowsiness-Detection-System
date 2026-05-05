# Driver Drowsiness Detection System - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DROWSINESS DETECTION SYSTEM              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐  │
│  │   Input      │    │   Data       │   │  Preproc.   │  │
│  │  Pipeline    │───▶│  Loading     │──▶│  Features   │  │
│  │              │    │              │   │  Extraction │  │
│  └──────────────┘    └──────────────┘   └──────────────┘  │
│         △                                       │           │
│         │                                       ▼           │
│         │                              ┌──────────────────┐│
│         │                              │  ML Model        ││
│         │                              │  (CNN/RNN)       ││
│         │                              │  Drowsiness      ││
│         │                              │  Classifier      ││
│         │                              └──────────────────┘│
│         │                                       │           │
│         │                                       ▼           │
│  ┌──────────────┐                     ┌──────────────────┐│
│  │   Output     │◀────────────────────│ Inference        ││
│  │   Actions    │                     │ & Confidence     ││
│  │   Alerts     │                     │ Scoring          ││
│  └──────────────┘                     └──────────────────┘│
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Alert System (Audio, Visual, SMS, Dashboard)        │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Architecture Layers

### 1. **Input Layer**
- Camera feed capture (real-time video)
- Dataset loader (for training)
- Frame extraction and buffering

### 2. **Data Pipeline**
- **Loading**: JSON annotations + image loading
- **Validation**: Ensure data integrity
- **Caching**: In-memory cache for faster access
- **Augmentation**: Rotation, brightness, blur (for training)

### 3. **Preprocessing Layer**
- Image normalization (mean/std)
- Resizing to model input size (224x224)
- Face detection & ROI extraction
- Feature calculation:
  - PERCLOS (eye closure percentage)
  - Head pose angles
  - Eye aspect ratio

### 4. **Feature Extraction**
- Temporal features from frame sequences
- Statistical features (mean, std, trend of PERCLOS)
- Eye state transitions
- Blink frequency

### 5. **ML Model Layer**
- **Primary**: CNN for frame-level features
- **Secondary**: LSTM/RNN for temporal patterns
- **Hybrid**: CNN-LSTM for end-to-end learning
- Outputs: Drowsiness score (0-1)

### 6. **Inference Engine**
- Real-time prediction
- Confidence scoring
- Decision logic with thresholds
- State tracking (frame history)

### 7. **Alert System**
- Threshold-based alerts
- Multi-modal notifications (audio, visual)
- Alert deduplication
- Event logging

---

## Module Structure

```
src/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # Dataset loader
│   ├── annotations.py      # JSON annotation handler
│   └── loader.py           # DataLoader with caching
│
├── preprocessing/
│   ├── __init__.py
│   ├── image_processor.py  # Normalization, resizing
│   ├── face_detector.py    # Face detection & ROI
│   └── feature_extractor.py # PERCLOS, head pose, etc.
│
├── models/
│   ├── __init__.py
│   ├── cnn.py              # CNN backbone
│   ├── lstm.py             # LSTM temporal model
│   └── hybrid.py           # CNN-LSTM combined
│
├── inference/
│   ├── __init__.py
│   ├── predictor.py        # Real-time predictor
│   ├── classifier.py       # Drowsiness classifier
│   └── alert_system.py     # Alert generation
│
├── config.py               # Configuration (paths, params)
├── utils.py                # Utility functions
└── train.py                # Training script
```

---

## Data Flow

### Training Pipeline
```
Raw Videos
    ↓
Extract Frames + Annotations (JSON)
    ↓
[Dataset] Load images + labels
    ↓
[Preprocessing] Resize, normalize, augment
    ↓
[Feature Extraction] PERCLOS, head pose, eye state
    ↓
[Model] Train CNN-LSTM
    ↓
[Checkpoints] Save best model
    ↓
[Evaluation] Validate on test set
```

### Inference Pipeline
```
Camera Feed (Video Stream)
    ↓
[Frame Buffer] Maintain sequence (last N frames)
    ↓
[Face Detection] Extract face ROI
    ↓
[Preprocessing] Normalize, resize
    ↓
[Feature Extraction] Real-time features
    ↓
[Model Inference] Get drowsiness score
    ↓
[Alert Logic] Threshold check
    ↓
[Output] Alert or Log
```

---

## Key Components

### 1. Dataset Handler (`data/dataset.py`)
```
Responsibilities:
- Load JSON annotations
- Match images to labels
- Handle multiple datasets
- Data validation
- Batch creation
```

### 2. Preprocessing Pipeline (`preprocessing/`)
```
Responsibilities:
- Image normalization (ImageNet stats)
- Resizing to 224x224
- Face detection using OpenCV/MediaPipe
- Feature extraction:
  - PERCLOS calculation
  - Head pose estimation
  - Eye aspect ratio
```

### 3. Model Architecture (`models/`)
```
CNN-LSTM Hybrid:
- Input: 224x224x3 images + temporal sequence
- CNN: Extract spatial features (ResNet50 backbone)
- LSTM: Learn temporal patterns
- Output: Binary classification (drowsy/alert)
```

### 4. Inference Engine (`inference/`)
```
Responsibilities:
- Load trained model
- Pre-process frames
- Generate predictions
- Track state over time
- Generate alerts
```

### 5. Configuration (`config.py`)
```
Paths:
- dataset_root
- model_checkpoints
- output_dir

Model Parameters:
- input_size: 224x224
- sequence_length: 10 frames
- batch_size: 32
- learning_rate

Thresholds:
- drowsiness_threshold: 0.6
- alert_cooldown: 5 seconds
```

---

## Implementation Strategy

### Phase 1: Data Pipeline ✅ (Foundation)
- [ ] Load dataset from JSON
- [ ] Create DataLoader
- [ ] Implement caching

### Phase 2: Preprocessing ✅ (Feature Engineering)
- [ ] Image normalization
- [ ] Face detection
- [ ] PERCLOS extraction from metadata
- [ ] Augmentation strategies

### Phase 3: Model Development 🔧 (ML)
- [ ] CNN backbone (transfer learning)
- [ ] LSTM temporal model
- [ ] Hybrid CNN-LSTM
- [ ] Training loop

### Phase 4: Inference Engine 🔧 (Deployment)
- [ ] Real-time predictor
- [ ] Confidence scoring
- [ ] Alert system
- [ ] State tracking

### Phase 5: Optimization & Deployment
- [ ] Model quantization
- [ ] Edge deployment (TensorRT)
- [ ] Performance monitoring
- [ ] A/B testing

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Data** | Python, NumPy, Pandas, OpenCV |
| **ML Framework** | PyTorch / TensorFlow |
| **Face Detection** | MediaPipe / OpenCV DNN |
| **Preprocessing** | Albumentations, Torchvision |
| **Inference** | ONNX, TensorRT (optional) |
| **Deployment** | FastAPI, Docker |
| **Monitoring** | Prometheus, Grafana |

---

## Scalability Considerations

### Horizontal Scaling
- Multi-GPU training (DataParallel)
- Distributed inference (load balancing)
- Model serving (Triton, KServe)

### Vertical Scaling
- Larger models (ResNet152, ViT)
- Longer sequences (30+ frames)
- Ensemble methods

### Data Scaling
- 10x more training data
- Hard negative mining
- Synthetic data generation

---

## Testing Strategy

```
Unit Tests:
- Dataset loading
- Preprocessing functions
- Feature extraction

Integration Tests:
- Full pipeline (data → inference)
- Model serving
- Alert triggering

End-to-End Tests:
- Real video feeds
- Performance benchmarks
- Stress testing
```

---

## Next Steps
1. Review and approve architecture
2. Start with Phase 1: Data Pipeline
3. Create unit tests
4. Build prototypes incrementally
