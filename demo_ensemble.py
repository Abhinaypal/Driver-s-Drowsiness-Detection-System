"""
Demonstration script for multi-algorithm ensemble drowsiness detection.

This script shows how to:
1. Use individual algorithms (CNN, LSTM, GRU, XGBoost, etc.)
2. Combine them in an ensemble
3. Compare predictions across models
4. Achieve higher reliability through voting
"""
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import LABELS_DIR, IMAGES_DIR, MODEL_CONFIG, INFERENCE_CONFIG
from src.data import AnnotationLoader, DatasetBuilder
from src.preprocessing import ImageProcessor, FeatureExtractor, TemporalFeatureExtractor
from src.inference import RuleBasedClassifier, RealtimeInference, AlertSystem, EnsembleDrowsinessDetector
from src.inference import CNNImageClassifier
from src.models import (
    SimpleDrowsinessCNN,
    LSTMDrowsinessDetector,
    GRUDrowsinessDetector,
    CNNWithAttention,
    ResNetWithAttention,
    XGBoostDrowsinessClassifier,
    LightGBMDrowsinessClassifier,
    extract_feature_vector
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAlgorithmDrowsinessSystem:
    """Demonstrates using multiple algorithms for drowsiness detection."""
    
    def __init__(self):
        logger.info("=" * 80)
        logger.info("MULTI-ALGORITHM DROWSINESS DETECTION SYSTEM")
        logger.info("=" * 80)
        
        # Load data
        logger.info("\n[1] Loading dataset...")
        self.load_dataset()
        
        # Initialize algorithms
        logger.info("\n[2] Initializing individual algorithms...")
        self.initialize_algorithms()
        
        # Create ensemble
        logger.info("\n[3] Creating ensemble detector...")
        self.create_ensemble()
    
    def load_dataset(self):
        """Load and prepare dataset."""
        loader = AnnotationLoader(LABELS_DIR)
        builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR, loader)
        self.dataset = builder.get_dataset()
        
        stats = loader.get_statistics()
        logger.info(f"Dataset loaded: {stats['total_samples']} samples")
        logger.info(f"Classes: {stats['classes']}")
    
    def initialize_algorithms(self):
        """Initialize all individual algorithms."""
        self.algorithms = {}
        
        # 1. Rule-Based Classifier (baseline)
        logger.info("  - Rule-Based Classifier (instant, no training)")
        self.algorithms['rule_based'] = RuleBasedClassifier()
        
        # 2. CNN (Image-based)
        logger.info("  - SimpleDrowsinessCNN (basic CNN)")
        try:
            self.algorithms['cnn'] = CNNImageClassifier(device="cpu")
        except Exception as e:
            logger.warning(f"    Could not initialize CNN: {e}")
        
        # 3. CNN with Attention
        logger.info("  - CNNWithAttention (learns important image regions)")
        try:
            self.algorithms['cnn_attention'] = CNNWithAttention()
        except Exception as e:
            logger.warning(f"    Could not initialize CNN+Attention: {e}")
        
        # 4. ResNet50 Transfer Learning
        logger.info("  - ResNetWithAttention (pre-trained backbone)")
        try:
            self.algorithms['resnet'] = ResNetWithAttention(pretrained=True)
        except Exception as e:
            logger.warning(f"    Could not initialize ResNet: {e}")
        
        # 5. LSTM (Temporal)
        logger.info("  - LSTMDrowsinessDetector (temporal sequences)")
        try:
            self.algorithms['lstm'] = LSTMDrowsinessDetector()
        except Exception as e:
            logger.warning(f"    Could not initialize LSTM: {e}")
        
        # 6. GRU (Temporal, lightweight)
        logger.info("  - GRUDrowsinessDetector (lightweight temporal)")
        try:
            self.algorithms['gru'] = GRUDrowsinessDetector()
        except Exception as e:
            logger.warning(f"    Could not initialize GRU: {e}")
        
        # 7. XGBoost (Feature-based)
        logger.info("  - XGBoostDrowsinessClassifier (gradient boosting)")
        try:
            self.algorithms['xgboost'] = XGBoostDrowsinessClassifier()
        except Exception as e:
            logger.warning(f"    Could not initialize XGBoost: {e}")
        
        # 8. LightGBM (Feature-based, fast)
        logger.info("  - LightGBMDrowsinessClassifier (lightweight boosting)")
        try:
            self.algorithms['lightgbm'] = LightGBMDrowsinessClassifier()
        except Exception as e:
            logger.warning(f"    Could not initialize LightGBM: {e}")
        
        logger.info(f"✓ Initialized {len(self.algorithms)} algorithms")
    
    def create_ensemble(self):
        """Create ensemble combining all algorithms."""
        self.ensemble = EnsembleDrowsinessDetector(voting_strategy='weighted')
        
        # Add algorithms to ensemble with weights
        # Rule-based gets higher weight (proven threshold-based approach)
        self.ensemble.add_model('rule_based', self.algorithms['rule_based'], weight=2.0)
        
        # Deep learning models get moderate weight
        for algo_name in ['cnn', 'cnn_attention', 'resnet', 'lstm', 'gru']:
            if algo_name in self.algorithms:
                self.ensemble.add_model(algo_name, self.algorithms[algo_name], weight=1.5)
        
        # Gradient boosting gets moderate weight
        for algo_name in ['xgboost', 'lightgbm']:
            if algo_name in self.algorithms:
                self.ensemble.add_model(algo_name, self.algorithms[algo_name], weight=1.5)
        
        logger.info(f"✓ Ensemble created with {len(self.algorithms)} models")
        logger.info(f"  Voting strategy: {self.ensemble.voting_strategy}")
    
    def demonstrate_predictions(self):
        """Demonstrate predictions on sample data."""
        logger.info("\n" + "=" * 80)
        logger.info("DEMONSTRATION: Multi-Algorithm Predictions")
        logger.info("=" * 80)
        
        if not self.dataset:
            logger.warning("No dataset samples available")
            return
        
        # Take first few samples
        num_samples = min(3, len(self.dataset))
        
        for idx in range(num_samples):
            sample = self.dataset[idx]
            logger.info(f"\n[Sample {idx + 1}/{num_samples}]")
            logger.info(f"  Image: {Path(sample['image_path']).name}")
            logger.info(f"  True class: {sample['class_name']}")
            
            # Extract features
            attributes = sample['attributes']
            features = FeatureExtractor.extract_all_features(attributes)
            
            logger.info(f"\n  Feature Summary:")
            logger.info(f"    - PERCLOS: {features['perclos']:.3f}")
            logger.info(f"    - Eye State: {features['eye_state']}")
            logger.info(f"    - Head Pose (P,Y,R): {features['head_pose']['p']:.1f}, {features['head_pose']['y']:.1f}, {features['head_pose']['r']:.1f}")
            
            # Get predictions from individual algorithms
            logger.info(f"\n  Individual Algorithm Predictions:")
            
            for algo_name, algo in self.algorithms.items():
                try:
                    class_id, confidence = algo.predict(features)
                    class_name = ["Awake", "Drowsy", "Asleep"][class_id]
                    logger.info(f"    {algo_name:20s}: {class_name:15s} ({confidence:.2%})")
                except Exception as e:
                    logger.debug(f"    {algo_name}: Error - {e}")
            
            # Ensemble prediction
            logger.info(f"\n  🎯 Ensemble Prediction (Voting Strategy: {self.ensemble.voting_strategy}):")
            try:
                class_id, confidence, details = self.ensemble.predict(features, return_details=True)
                class_name = ["Awake", "Drowsy", "Asleep"][class_id]
                logger.info(f"    ➜ {class_name:15s} ({confidence:.2%})")
                logger.info(f"    Consensus from {details['num_models']} models")
            except Exception as e:
                logger.warning(f"    Ensemble error: {e}")
    
    def print_algorithm_comparison(self):
        """Print comparison of algorithms."""
        logger.info("\n" + "=" * 80)
        logger.info("ALGORITHM COMPARISON & CHARACTERISTICS")
        logger.info("=" * 80)
        
        algorithms_info = {
            'Rule-Based': {
                'type': 'Symbolic AI',
                'speed': '🚀 Very Fast',
                'training': '⏭️ None needed',
                'features': 'PERCLOS, eye state',
                'interpretability': '✅ Fully interpretable',
                'edge_device': '✅ Yes'
            },
            'SimpleDrowsinessCNN': {
                'type': 'Deep Learning (CNN)',
                'speed': '⚡ Fast',
                'training': '📚 Needs data',
                'features': 'Learns from images',
                'interpretability': '❓ Black box',
                'edge_device': '✅ Yes'
            },
            'CNN + Attention': {
                'type': 'Deep Learning (CNN)',
                'speed': '⚡ Fast',
                'training': '📚 Needs data',
                'features': 'Learns important regions',
                'interpretability': '✅ Attention maps show focus',
                'edge_device': '✅ Yes'
            },
            'ResNet Transfer Learning': {
                'type': 'Deep Learning (CNN)',
                'speed': '⚡ Fast',
                'training': '📚 Transfer learning',
                'features': 'Pre-trained backbone',
                'interpretability': '❓ Black box',
                'edge_device': '⚠️ Large'
            },
            'LSTM': {
                'type': 'Deep Learning (RNN)',
                'speed': '⚠️ Slow',
                'training': '📚 Needs sequences',
                'features': 'Temporal patterns',
                'interpretability': '❓ Black box',
                'edge_device': '⚠️ Medium'
            },
            'GRU': {
                'type': 'Deep Learning (RNN)',
                'speed': '⚡ Faster than LSTM',
                'training': '📚 Needs sequences',
                'features': 'Temporal patterns (lighter)',
                'interpretability': '❓ Black box',
                'edge_device': '✅ Smaller'
            },
            'XGBoost': {
                'type': 'Gradient Boosting',
                'speed': '🚀 Very Fast',
                'training': '📚 Needs features',
                'features': 'Hand-crafted features',
                'interpretability': '✅ Feature importance',
                'edge_device': '✅ Yes'
            },
            'LightGBM': {
                'type': 'Gradient Boosting',
                'speed': '🚀 Fastest boosting',
                'training': '📚 Needs features',
                'features': 'Hand-crafted features',
                'interpretability': '✅ Feature importance',
                'edge_device': '✅ Small'
            },
        }
        
        for algo_name, info in algorithms_info.items():
            logger.info(f"\n{algo_name}")
            logger.info(f"  Type: {info['type']}")
            logger.info(f"  Speed: {info['speed']}")
            logger.info(f"  Training: {info['training']}")
            logger.info(f"  Features: {info['features']}")
            logger.info(f"  Interpretability: {info['interpretability']}")
            logger.info(f"  Edge Device: {info['edge_device']}")


def main():
    """Run demonstration."""
    system = MultiAlgorithmDrowsinessSystem()
    
    # Print comparison
    system.print_algorithm_comparison()
    
    # Demonstrate predictions
    system.demonstrate_predictions()
    
    logger.info("\n" + "=" * 80)
    logger.info("KEY BENEFITS OF MULTI-ALGORITHM ENSEMBLE:")
    logger.info("=" * 80)
    logger.info("✓ Reduces individual model bias through voting")
    logger.info("✓ Handles different types of failures (image blur, temporal gaps, etc.)")
    logger.info("✓ Provides redundancy - if one model fails, others classify")
    logger.info("✓ Confidence scoring reflects consensus among models")
    logger.info("✓ Can customize voting strategy for specific needs")
    logger.info("✓ Better generalization to unseen scenarios")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
