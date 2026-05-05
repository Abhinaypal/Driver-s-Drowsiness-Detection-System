"""
XGBoost and LightGBM classifiers for feature-based drowsiness detection.

These gradient boosting models work on extracted features like PERCLOS,
head pose, eye state, and temporal statistics.
"""
from typing import Dict, Tuple, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class XGBoostDrowsinessClassifier:
    """
    XGBoost classifier for feature-based drowsiness detection.
    
    Works on extracted features:
    - PERCLOS (eye closure percentage)
    - Head pose (pitch, yaw, roll)
    - Eye state
    - Temporal statistics (trends, frequencies)
    
    Advantages:
    - Fast inference
    - Handles feature interactions
    - Gradient boosting captures non-linear patterns
    - Can work with missing features
    """
    
    def __init__(self, num_classes: int = 3, tree_depth: int = 5, learning_rate: float = 0.1):
        """
        Initialize XGBoost classifier.
        
        Args:
            num_classes: Number of output classes (3: Awake, Drowsy, Asleep)
            tree_depth: Maximum tree depth
            learning_rate: Boosting learning rate
        """
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost required. Install with: pip install xgboost")
        
        self.num_classes = num_classes
        self.tree_depth = tree_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_names = None
    
    def build_model(self):
        """Build untrained model."""
        self.model = self.xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=self.num_classes,
            max_depth=self.tree_depth,
            learning_rate=self.learning_rate,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            device='cuda'  # Use GPU if available
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """
        Train the XGBoost model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Class labels (n_samples,)
            validation_split: Fraction of data for validation
        """
        if self.model is None:
            self.build_model()
        
        # Split data
        n_train = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        logger.info(f"XGBoost trained. Best score: {self.model.best_score:.4f}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class and confidence.
        
        Args:
            X: Feature matrix (n_samples, n_features) or (n_features,)
        
        Returns:
            Tuple of (class_ids, confidences)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Handle single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences
    
    def save(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")
        self.model.save_model(filepath)
        logger.info(f"XGBoost model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        if self.model is None:
            self.build_model()
        self.model.load_model(filepath)
        logger.info(f"XGBoost model loaded from {filepath}")


class LightGBMDrowsinessClassifier:
    """
    LightGBM classifier for feature-based drowsiness detection.
    
    Similar to XGBoost but:
    - Even faster training and inference
    - More memory efficient
    - Better for large datasets
    - Good for edge devices
    """
    
    def __init__(self, num_classes: int = 3, tree_depth: int = 5, learning_rate: float = 0.1):
        """
        Initialize LightGBM classifier.
        
        Args:
            num_classes: Number of output classes
            tree_depth: Maximum tree depth
            learning_rate: Boosting learning rate
        """
        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError:
            raise ImportError("LightGBM required. Install with: pip install lightgbm")
        
        self.num_classes = num_classes
        self.tree_depth = tree_depth
        self.learning_rate = learning_rate
        self.model = None
    
    def build_model(self):
        """Build untrained model."""
        params = {
            'objective': 'multiclass',
            'num_class': self.num_classes,
            'max_depth': self.tree_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        self.params = params
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, num_rounds: int = 100):
        """
        Train the LightGBM model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Class labels (n_samples,)
            validation_split: Fraction of data for validation
            num_rounds: Number of boosting rounds
        """
        if not hasattr(self, 'params'):
            self.build_model()
        
        # Split data
        n_train = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Create datasets
        train_data = self.lgb.Dataset(X_train, label=y_train)
        val_data = self.lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train
        self.model = self.lgb.train(
            self.params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=[val_data],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        logger.info(f"LightGBM trained with {num_rounds} rounds")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class and confidence.
        
        Args:
            X: Feature matrix (n_samples, n_features) or (n_features,)
        
        Returns:
            Tuple of (class_ids, confidences)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Handle single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = self.model.predict(X)  # Returns probabilities
        class_ids = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return class_ids, confidences
    
    def save(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")
        self.model.save_model(filepath)
        logger.info(f"LightGBM model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        self.model = self.lgb.Booster(model_file=filepath)
        logger.info(f"LightGBM model loaded from {filepath}")


def extract_feature_vector(features: Dict) -> np.ndarray:
    """
    Convert feature dict to vector for gradient boosting models.
    
    Args:
        features: Dictionary with keys: perclos, eye_state, head_pose, zone, etc.
    
    Returns:
        Feature vector as numpy array
    """
    feature_vector = []
    
    # PERCLOS
    feature_vector.append(features.get('perclos', 0.0))
    
    # Eye state (one-hot encoding)
    eye_state = features.get('eye_state', 'Open')
    feature_vector.extend([
        1.0 if 'Open' in eye_state else 0.0,
        1.0 if 'Drowsy' in eye_state or 'Microsleep' in eye_state else 0.0,
        1.0 if 'Closed' in eye_state else 0.0,
    ])
    
    # Head pose
    head_pose = features.get('head_pose', {})
    feature_vector.extend([
        head_pose.get('p', 0.0),  # pitch
        head_pose.get('y', 0.0),  # yaw
        head_pose.get('r', 0.0),  # roll
    ])
    
    # Temporal statistics (if available)
    if 'perclos_mean' in features:
        feature_vector.extend([
            features.get('perclos_mean', 0.0),
            features.get('perclos_std', 0.0),
            features.get('blink_frequency', 0.0),
        ])
    
    return np.array(feature_vector, dtype=np.float32)
