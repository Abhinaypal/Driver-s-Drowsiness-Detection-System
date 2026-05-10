"""
Main entry point demonstrating the system architecture
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DATASET_ROOT, VIDEOS_DIR, IMAGES_DIR, LABELS_DIR, LABELS_DIRS, LOGS_DIR,
    MODEL_CONFIG, INFERENCE_CONFIG, LOGGING_CONFIG
)
from src.data import AnnotationLoader, DatasetBuilder
from src.preprocessing import ImageProcessor, FeatureExtractor, TemporalFeatureExtractor
from src.inference import RuleBasedClassifier, RealtimeInference, AlertSystem

# Setup logging
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def demonstrate_data_loading():
    """Demonstrate data loading and exploration"""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Loading and Exploration")
    logger.info("=" * 60)
    
    # Load annotations
    annotation_loader = AnnotationLoader(LABELS_DIRS, VIDEOS_DIR)
    logger.info(f"Available annotations: {annotation_loader.get_all_keys()}")
    
    # Get statistics
    stats = annotation_loader.get_statistics()
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total annotations: {stats['total_annotations']}")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Class distribution:")
    for cls, count in stats['classes'].items():
        logger.info(f"    {cls}: {count}")
    
    # Build dataset from both image and video sources
    dataset_builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR, videos_dir=VIDEOS_DIR, annotation_loader=annotation_loader)
    dataset = dataset_builder.get_dataset()
    logger.info(f"\nBuilt dataset with {len(dataset)} samples")
    
    # Show sample
    if dataset:
        sample = dataset[0]
        logger.info(f"\nSample from dataset:")
        logger.info(f"  Image: {sample['image_name']}")
        logger.info(f"  Class: {sample['class_label']}")
        logger.info(f"  Timestamp: {sample['timestamp']}")
        logger.info(f"  Attributes: {sample['attributes']}")
    
    return dataset_builder


def demonstrate_preprocessing(dataset_builder):
    """Demonstrate preprocessing pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Preprocessing and Feature Extraction")
    logger.info("=" * 60)
    
    # Initialize processors
    image_processor = ImageProcessor(
        target_size=(MODEL_CONFIG['input_size'], MODEL_CONFIG['input_size'])
    )
    
    # Process a sample
    dataset = dataset_builder.get_dataset()
    if not dataset:
        logger.warning("No samples in dataset")
        return
    
    sample = dataset[0]
    image_path = Path(sample['image_path'])
    
    logger.info(f"Processing image: {image_path.name}")
    
    # Load and preprocess image
    processed_image = image_processor.preprocess(image_path)
    if processed_image is not None:
        logger.info(f"  Preprocessed shape: {processed_image.shape}")
    else:
        logger.error("  Failed to preprocess image")
        return
    
    # Extract features from annotations
    features = FeatureExtractor.extract_all_features(sample['attributes'])
    logger.info(f"\nExtracted features:")
    logger.info(f"  PERCLOS: {features['perclos']:.4f}")
    logger.info(f"  Eye State: {features['eye_state']}")
    logger.info(f"  Eye State Label: {features['eye_state_label']}")
    logger.info(f"  Head Pose: {features['head_pose']}")
    logger.info(f"  Zone: {features['zone']}")


def demonstrate_inference():
    """Demonstrate inference and classification"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Inference and Classification")
    logger.info("=" * 60)
    
    # Initialize components
    classifier = RuleBasedClassifier()
    inference_engine = RealtimeInference(
        classifier,
        sequence_length=INFERENCE_CONFIG['frame_buffer_size'],
        alert_threshold=INFERENCE_CONFIG['drowsiness_threshold'],
        alert_cooldown_seconds=INFERENCE_CONFIG['alert_cooldown_seconds']
    )
    alert_system = AlertSystem(
        log_file=LOGS_DIR / "alerts.log",
        enable_audio=False,  # Disable for demo
        enable_visual=True,
        enable_sms=False
    )
    
    # Simulate inference on dataset
    dataset_builder = DatasetBuilder(LABELS_DIRS, IMAGES_DIR, videos_dir=VIDEOS_DIR)
    dataset = dataset_builder.get_dataset()
    
    # Take first 15 samples for demo
    demo_samples = dataset[:15]
    
    logger.info(f"\nSimulating inference on {len(demo_samples)} samples")
    
    for i, sample in enumerate(demo_samples):
        # Extract features
        features = FeatureExtractor.extract_all_features(sample['attributes'])
        
        # Make prediction
        result = inference_engine.predict(features)
        
        # Process result
        alert = alert_system.process_inference(result)
        
        logger.info(f"\nSample {i+1}:")
        logger.info(f"  Ground Truth: {sample['class_label']}")
        logger.info(f"  Predicted: {result['class_name']}")
        logger.info(f"  Confidence: {result['confidence']:.2%}")
        logger.info(f"  Temporal Score: {result['temporal_score']:.2%}")
        logger.info(f"  Alert Triggered: {result['should_alert']}")
        
        if alert:
            inference_engine.update_alert_state(True)
            logger.warning(f"  ⚠️  ALERT: {alert['reason']}")
    
    # Print statistics
    logger.info(f"\n" + "=" * 60)
    logger.info("Inference Statistics")
    logger.info("=" * 60)
    buffer_stats = inference_engine.get_buffer_statistics()
    logger.info(f"Buffer Statistics:")
    for key, value in buffer_stats.items():
        logger.info(f"  {key}: {value}")
    
    alert_stats = alert_system.get_statistics()
    logger.info(f"\nAlert Statistics:")
    logger.info(f"  Total alerts: {alert_stats['total_alerts']}")
    logger.info(f"  Alerts by class: {alert_stats['alerts_by_class']}")


def main():
    """Main demonstration"""
    logger.info("\n")
    logger.info("#" * 60)
    logger.info("# DRIVER DROWSINESS DETECTION SYSTEM")
    logger.info("# Architecture Demonstration")
    logger.info("#" * 60)
    logger.info("\n")
    
    try:
        # Step 1: Data Loading
        dataset_builder = demonstrate_data_loading()
        
        # Step 2: Preprocessing
        demonstrate_preprocessing(dataset_builder)
        
        # Step 3: Inference
        demonstrate_inference()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Demonstration completed successfully!")
        logger.info("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
