"""
Ensemble system that combines multiple classifiers for improved reliability.

Combines predictions from:
- CNN (SimpleDrowsinessCNN)
- ResNet50 with Attention
- LSTM / GRU (temporal models)
- XGBoost / LightGBM (feature-based)
- Rule-based classifier

Uses voting and averaging strategies for consensus predictions.
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class EnsembleDrowsinessDetector:
    """
    Ensemble detector combining multiple drowsiness classifiers.
    
    Strategies:
    1. **Majority Voting**: Each model votes for a class, majority wins
    2. **Confidence Averaging**: Average softmax probabilities across models
    3. **Weighted Voting**: Weight models by their performance on validation set
    4. **Stacking**: Train a meta-classifier on model outputs
    
    Benefits:
    - Reduces individual model bias and overfitting
    - Improves reliability through consensus
    - Redundancy - if one model fails, others can still classify
    - More robust to edge cases
    
    Attributes:
        models: Dictionary of (name, classifier) pairs
        model_weights: Dictionary of model importance weights
        voting_strategy: 'majority', 'confidence', 'weighted', 'stacking'
    """
    
    def __init__(self, voting_strategy: str = 'weighted'):
        """
        Initialize ensemble detector.
        
        Args:
            voting_strategy: How to combine predictions
                - 'majority': Simple majority voting
                - 'confidence': Average probabilities
                - 'weighted': Weight by model performance
                - 'stacking': Use meta-classifier
        """
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.voting_strategy = voting_strategy
        self.meta_model = None
        self.class_names = ("Awake", "Drowsy/Microsleep", "Asleep")
        self.num_classes = 3
    
    def add_model(self, name: str, classifier: Any, weight: float = 1.0):
        """
        Add a classifier to the ensemble.
        
        Args:
            name: Unique identifier for the model (e.g., 'cnn', 'xgboost', 'lstm')
            classifier: Classifier instance with predict() or predict_proba() method
            weight: Importance weight for this model (used in weighted voting)
        """
        self.models[name] = classifier
        self.model_weights[name] = weight
        logger.info(f"Added model '{name}' to ensemble with weight {weight}")
    
    def remove_model(self, name: str):
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            del self.model_weights[name]
            logger.info(f"Removed model '{name}' from ensemble")
    
    def predict(self, features: Dict, return_details: bool = False) -> Tuple[int, float]:
        """
        Make ensemble prediction.
        
        Args:
            features: Feature dictionary
            return_details: Whether to return detailed per-model predictions
        
        Returns:
            Tuple of (class_id, confidence)
            If return_details=True, returns (class_id, confidence, details_dict)
        """
        if not self.models:
            raise RuntimeError("No models in ensemble. Add models with add_model().")
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            try:
                # Try different prediction interfaces
                if hasattr(model, 'predict_proba'):
                    prob_dict = self._get_probabilities_proba(model_name, model, features)
                elif hasattr(model, 'predict'):
                    prob_dict = self._get_probabilities_predict(model_name, model, features)
                else:
                    logger.warning(f"Model '{model_name}' has no predict methods")
                    continue
                
                predictions[model_name] = prob_dict
            except Exception as e:
                logger.warning(f"Error predicting with model '{model_name}': {e}")
                continue
        
        if not predictions:
            raise RuntimeError("No models could make predictions")
        
        # Combine predictions
        if self.voting_strategy == 'majority':
            class_id, confidence = self._majority_voting(predictions)
        elif self.voting_strategy == 'confidence':
            class_id, confidence = self._confidence_averaging(predictions)
        elif self.voting_strategy == 'weighted':
            class_id, confidence = self._weighted_voting(predictions)
        elif self.voting_strategy == 'stacking':
            class_id, confidence = self._stacking_prediction(predictions)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        
        if return_details:
            details = {
                'individual_predictions': predictions,
                'strategy': self.voting_strategy,
                'num_models': len(predictions)
            }
            return class_id, confidence, details
        
        return class_id, confidence
    
    def _majority_voting(self, predictions: Dict[str, Dict]) -> Tuple[int, float]:
        """Simple majority voting."""
        votes = defaultdict(int)
        
        for model_name, prob_dict in predictions.items():
            class_id = np.argmax(list(prob_dict.values()))
            votes[class_id] += 1
        
        # Get majority class
        class_id = max(votes.keys(), key=lambda k: votes[k])
        
        # Confidence: fraction of votes
        confidence = votes[class_id] / len(predictions)
        
        logger.debug(f"Majority voting: {dict(votes)}, winner: {class_id} ({confidence:.2f})")
        
        return int(class_id), float(confidence)
    
    def _confidence_averaging(self, predictions: Dict[str, Dict]) -> Tuple[int, float]:
        """Average softmax probabilities."""
        avg_probs = np.zeros(self.num_classes)
        
        for model_name, prob_dict in predictions.items():
            probs = np.array([prob_dict.get(i, 0.0) for i in range(self.num_classes)])
            avg_probs += probs
        
        avg_probs /= len(predictions)
        
        class_id = int(np.argmax(avg_probs))
        confidence = float(np.max(avg_probs))
        
        logger.debug(f"Confidence averaging: {avg_probs}, winner: {class_id} ({confidence:.2f})")
        
        return class_id, confidence
    
    def _weighted_voting(self, predictions: Dict[str, Dict]) -> Tuple[int, float]:
        """Weighted average of probabilities."""
        weighted_probs = np.zeros(self.num_classes)
        total_weight = 0.0
        
        for model_name, prob_dict in predictions.items():
            weight = self.model_weights.get(model_name, 1.0)
            probs = np.array([prob_dict.get(i, 0.0) for i in range(self.num_classes)])
            weighted_probs += weight * probs
            total_weight += weight
        
        weighted_probs /= total_weight
        
        class_id = int(np.argmax(weighted_probs))
        confidence = float(np.max(weighted_probs))
        
        logger.debug(f"Weighted voting: {weighted_probs}, winner: {class_id} ({confidence:.2f})")
        
        return class_id, confidence
    
    def _stacking_prediction(self, predictions: Dict[str, Dict]) -> Tuple[int, float]:
        """
        Use stacking meta-model to combine predictions.
        Requires meta_model to be trained first.
        """
        if self.meta_model is None:
            # If no meta-model trained, use confidence averaging as fallback
            logger.warning("No meta-model. Falling back to confidence averaging.")
            return self._confidence_averaging(predictions)
        
        # Build feature vector from model outputs
        meta_features = self._build_meta_features(predictions)
        
        # Predict with meta-model
        try:
            class_id, confidence = self.meta_model.predict(meta_features)
            logger.debug(f"Stacking prediction: {class_id} ({confidence:.2f})")
            return class_id, confidence
        except Exception as e:
            logger.warning(f"Meta-model prediction failed: {e}. Using averaging.")
            return self._confidence_averaging(predictions)
    
    def _build_meta_features(self, predictions: Dict[str, Dict]) -> np.ndarray:
        """Build feature vector from model predictions."""
        meta_features = []
        for model_name in sorted(self.models.keys()):
            if model_name in predictions:
                prob_dict = predictions[model_name]
                # Add probabilities for all classes
                for i in range(self.num_classes):
                    meta_features.append(prob_dict.get(i, 0.0))
        
        return np.array(meta_features, dtype=np.float32)
    
    def _get_probabilities_proba(self, model_name: str, model: Any, features: Dict) -> Dict[int, float]:
        """Get class probabilities from predict_proba method."""
        # Different models have different interfaces
        try:
            # For torch models
            import torch
            if isinstance(features, dict) and 'image_path' in features:
                # CNN-like models
                probas = model.predict_proba(features)
                if isinstance(probas, torch.Tensor):
                    probas = probas.cpu().numpy()[0]
            else:
                # Feature-based models
                probas = model.predict_proba(features)
                if isinstance(probas, torch.Tensor):
                    probas = probas.cpu().numpy()
            
            # Ensure shape is (num_classes,)
            if len(probas.shape) > 1:
                probas = probas[0]
            
            return {i: float(p) for i, p in enumerate(probas)}
        except Exception as e:
            logger.debug(f"predict_proba failed for {model_name}: {e}")
            return self._get_probabilities_predict(model_name, model, features)
    
    def _get_probabilities_predict(self, model_name: str, model: Any, features: Dict) -> Dict[int, float]:
        """Get class probabilities from predict method."""
        class_id, confidence = model.predict(features)
        
        # Build probability distribution: high confidence for predicted class
        probas = {}
        for i in range(self.num_classes):
            if i == class_id:
                probas[i] = confidence
            else:
                # Distribute remaining probability
                probas[i] = (1.0 - confidence) / (self.num_classes - 1)
        
        return probas
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.model_weights.copy()
    
    def set_model_weights(self, weights: Dict[str, float]):
        """Update model weights."""
        for name, weight in weights.items():
            if name in self.models:
                self.model_weights[name] = weight
                logger.info(f"Updated weight for model '{name}': {weight}")
    
    def get_summary(self) -> Dict:
        """Get ensemble summary."""
        return {
            'num_models': len(self.models),
            'models': list(self.models.keys()),
            'voting_strategy': self.voting_strategy,
            'weights': self.model_weights.copy(),
            'class_names': self.class_names
        }
