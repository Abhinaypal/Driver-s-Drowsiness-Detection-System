"""
Inference engine and drowsiness classification for real-time prediction.

This module provides:
- DrowsinessClassifier: Base interface for classifiers
- RuleBasedClassifier: Production-ready classifier using PERCLOS thresholds
- RealtimeInference: Real-time prediction with temporal reasoning

The system maintains frame history and detects drowsiness patterns using:
- PERCLOS values (eye closure percentage)
- Eye state transitions
- Temporal trends
- Multi-frame consensus
"""
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class DrowsinessClassifier:
    """Base classifier interface"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
    
    def load_model(self, model_path: str):
        """Load trained model"""
        raise NotImplementedError
    
    def predict(self, features: Dict) -> Tuple[int, float]:
        """
        Predict drowsiness class
        
        Returns:
            Tuple of (class_id, confidence)
            class_id: 0=awake, 1=drowsy, 2=asleep
            confidence: 0-1 confidence score
        """
        raise NotImplementedError


class RuleBasedClassifier(DrowsinessClassifier):
    """
    Rule-based drowsiness classifier using PERCLOS and eye state.
    
    This classifier is production-ready and doesn't require training.
    It uses hand-crafted rules based on PERCLOS values and eye states
    to classify drowsiness levels.
    
    Classification Rules:
    1. If eyes closed -> Asleep (confidence based on PERCLOS)
    2. If drowsy/microsleep and high PERCLOS -> Asleep
    3. If drowsy/microsleep -> Drowsy (confidence = PERCLOS)
    4. If eyes open and high PERCLOS -> Asleep
    5. If eyes open and medium PERCLOS -> Drowsy
    6. Otherwise -> Awake
    
    Attributes:
        perclos_drowsy_threshold: PERCLOS value indicating drowsiness
        perclos_asleep_threshold: PERCLOS value indicating sleep
    """
    
    def __init__(self,
                 perclos_drowsy_threshold: float = 0.2,
                 perclos_asleep_threshold: float = 0.8):
        """
        Initialize the rule-based classifier with thresholds.
        
        Args:
            perclos_drowsy_threshold: PERCLOS value to flag as drowsy (default: 0.2 = 20%)
            perclos_asleep_threshold: PERCLOS value to flag as asleep (default: 0.8 = 80%)
        """
        super().__init__()
        self.perclos_drowsy_threshold = perclos_drowsy_threshold
        self.perclos_asleep_threshold = perclos_asleep_threshold
        self.is_loaded = True  # No loading needed for rule-based classifier
    
    def predict(self, features: Dict) -> Tuple[int, float]:
        """
        Predict drowsiness using rule-based logic.
        
        Decision Logic:
        1. Check if eyes are closed -> Asleep
        2. Check if drowsy/microsleep state -> Drowsy or Asleep
        3. Check PERCLOS value against thresholds
        4. Default to awake if no conditions met
        
        Args:
            features: Feature dictionary with keys:
                     - perclos: float (0-1)
                     - eye_state: string (Open/Drowsy/Closed)
            
        Returns:
            Tuple of (class_id, confidence):
            - class_id: 0=Awake, 1=Drowsy, 2=Asleep
            - confidence: float (0-1) indicating prediction confidence
        """
        # Extract features
        perclos = float(features.get('perclos', 0.0) or 0.0)
        eye_state = str(features.get('eye_state', 'Open')).lower()
        
        # Rule 1: If eyes closed, classify as asleep (highest confidence)
        if 'closed' in eye_state:
            confidence = min(perclos + 0.4, 1.0)  # Very high confidence
            return 2, confidence
        
        # Rule 2: If eyes drowsy/microsleep
        if 'drowsy' in eye_state or 'microsleep' in eye_state:
            confidence = max(perclos, 0.55)  # At least 55% confidence
            # Check if should be asleep instead (lower threshold for asleep)
            if perclos > self.perclos_asleep_threshold * 0.8:  # More sensitive
                return 2, confidence  # Asleep
            return 1, confidence  # Drowsy
        
        # Rule 3: If eyes open but high PERCLOS (eyes closing rapidly)
        # Lower thresholds to be more sensitive to eye closure
        if perclos > self.perclos_asleep_threshold * 0.9:  # More sensitive
            return 2, perclos  # Asleep
        elif perclos > self.perclos_drowsy_threshold * 1.2:  # More sensitive
            return 1, perclos  # Drowsy
        
        # Rule 4: Default to awake
        return 0, 1.0 - perclos  # Confidence decreases as PERCLOS increases


class RealtimeInference:
    """
    Real-time inference engine with temporal reasoning and memory.
    
    This class maintains a sliding window of frame predictions and uses
    temporal patterns to detect drowsiness more reliably than single-frame
    classification.
    
    Key Features:
    - Frame history buffer (maintains last N predictions)
    - Temporal scoring (considers prediction trends)
    - State transition detection (sudden changes)
    - Alert cooldown (prevents alert spam)
    - Multi-factor decision making
    
    Attributes:
        classifier: DrowsinessClassifier instance
        sequence_length: Number of frames to maintain in buffer
        alert_threshold: Confidence threshold to trigger alerts
        predictions_buffer: Deque of recent class predictions
        confidence_buffer: Deque of recent confidence scores
        features_buffer: Deque of recent features
    """
    
    def __init__(self,
                 classifier: DrowsinessClassifier,
                 sequence_length: int = 10,
                 alert_threshold: float = 0.6,
                 alert_cooldown_seconds: float = 5.0,
                 asleep_duration_threshold: float = 3.0):
        """
        Initialize the real-time inference engine.
        
        Args:
            classifier: Classifier instance (RuleBasedClassifier, etc.)
            sequence_length: Number of frames to keep in history (default: 10)
            alert_threshold: Temporal score threshold for alerts (default: 0.6)
            alert_cooldown_seconds: Minimum seconds between alerts (default: 5)
            asleep_duration_threshold: Required seconds of continuous Asleep detection before alert
        """
        self.classifier = classifier
        self.sequence_length = sequence_length
        self.alert_threshold = alert_threshold
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.asleep_duration_threshold = asleep_duration_threshold
        
        # Buffers for temporal analysis
        self.predictions_buffer = deque(maxlen=sequence_length)  # Class IDs
        self.confidence_buffer = deque(maxlen=sequence_length)   # Confidence scores
        self.features_buffer = deque(maxlen=sequence_length)     # Feature dicts
        
        # Alert state tracking
        self.last_alert_time = None  # When was last alert triggered
        self.alert_count = 0         # Total alerts generated
        self.is_alerting = False     # Current alert state
        self.asleep_start_time = None  # When continuous Asleep detection began
    
    def predict(self, features: Dict) -> Dict:
        """
        Make a prediction and return detailed result with temporal analysis.
        
        Process:
        1. Get single-frame prediction from classifier
        2. Add to frame history buffers
        3. Calculate temporal score from history
        4. Determine if alert should be triggered
        5. Return comprehensive result dictionary
        
        Args:
            features: Feature dictionary with keys: perclos, eye_state, head_pose, zone
            
        Returns:
            Dictionary with keys:
            - class_id: 0=Awake, 1=Drowsy, 2=Asleep
            - class_name: String class name
            - confidence: Single-frame confidence (0-1)
            - temporal_score: Multi-frame confidence (0-1)
            - should_alert: Boolean whether to trigger alert
            - timestamp: ISO format timestamp
            - reason: Human-readable explanation
            - alert_count: Number of alerts (if should_alert)
            - time_since_last_alert: Seconds since last alert
        """
        # Step 1: Get prediction from classifier
        class_id, confidence = self.classifier.predict(features)
        
        # Step 2: Store in buffers for temporal analysis
        self.predictions_buffer.append(class_id)
        self.confidence_buffer.append(confidence)
        self.features_buffer.append(features)
        self._update_asleep_timer(class_id)
        
        # Step 3: Calculate temporal metrics
        temporal_score = self._calculate_temporal_score()
        
        # Step 4: Determine if alert should be triggered
        should_alert = self._should_alert(class_id, confidence, temporal_score)
        alert_count = self.alert_count
        time_since_last_alert = self._get_alert_cooldown_status()
        if should_alert:
            self.update_alert_state(True)
        else:
            self.update_alert_state(False)
        
        # Step 5: Build comprehensive response
        result = {
            'class_id': class_id,
            'class_name': self._id_to_class(class_id),
            'confidence': float(confidence),
            'temporal_score': float(temporal_score),
            'should_alert': should_alert,
            'timestamp': datetime.now().isoformat(),
            'reason': self._get_reason(class_id, confidence, temporal_score, should_alert),
        }
        
        # Add alert info if triggered
        if should_alert:
            result['alert_count'] = alert_count + 1
            result['time_since_last_alert'] = time_since_last_alert
        
        return result
    
    def _calculate_temporal_score(self) -> float:
        """
        Calculate drowsiness score based on frame history and trends.
        
        This method is key to robust drowsiness detection. Instead of relying
        on a single frame, it analyzes patterns in the frame history:
        
        Factors considered:
        1. Average severity of recent predictions
           - Higher severity (Asleep=3, Drowsy=2, Awake=1) = higher score
        2. Trend of predictions
           - Increasing drowsiness trend increases score
        3. Historical pattern
           - Consistent drowsiness increases score
        
        Returns:
            float: Temporal drowsiness score (0.0-1.0)
        """
        # Return 0 if no predictions yet
        if len(self.predictions_buffer) == 0:
            return 0.0
        
        # Get current buffers
        predictions = list(self.predictions_buffer)
        confidences = list(self.confidence_buffer)
        
        # Calculate weighted average based on class severity
        # Asleep is more severe than Drowsy which is more severe than Awake
        severity_weights = [1.0, 2.0, 3.0]  # Weights: Awake, Drowsy, Asleep
        weighted_score = sum(
            severity_weights[pred] * conf 
            for pred, conf in zip(predictions, confidences)
        ) / (len(predictions) * max(severity_weights))
        
        # Factor in trend: are recent predictions more drowsy?
        if len(predictions) >= 2:
            recent = predictions[-5:] if len(predictions) >= 5 else predictions
            trend = sum(1 for p in recent if p > 0) / len(recent)  # Ratio of non-awake frames
            # Weight: 70% average score, 30% trend
            weighted_score = 0.7 * weighted_score + 0.3 * trend
        
        # Clamp to 0-1 range
        return min(weighted_score, 1.0)
    
    def _update_asleep_timer(self, class_id: int) -> None:
        """Record when continuous Asleep frames begin."""
        if class_id == 2:
            if self.asleep_start_time is None:
                self.asleep_start_time = datetime.now()
        else:
            self.asleep_start_time = None

    def _asleep_duration(self) -> float:
        """Return how many seconds Asleep has been continuous."""
        if self.asleep_start_time is None:
            return 0.0
        return (datetime.now() - self.asleep_start_time).total_seconds()

    def _should_alert(self, class_id: int, confidence: float, temporal_score: float) -> bool:
        """Determine if alert should be triggered"""
        # Check cooldown
        if not self._check_alert_cooldown():
            return False
        
        # Alert conditions
        # 1. Continuous eyes-closed detection for the required duration
        if class_id == 2 and self._asleep_duration() >= self.asleep_duration_threshold:
            return True

        # 2. Persistent drowsy state (high temporal score)
        if class_id == 1 and temporal_score > self.alert_threshold:
            return True
        
        # 3. Sudden state transition to drowsy/asleep
        if len(self.predictions_buffer) >= 3:
            recent = list(self.predictions_buffer)[-3:]
            if recent[-2] == 0 and recent[-1] > 0:  # Transition from awake to drowsy
                return True
        
        return False
    
    def _check_alert_cooldown(self) -> bool:
        """Check if enough time has passed since last alert"""
        if self.last_alert_time is None:
            return True
        
        time_since_last = datetime.now() - self.last_alert_time
        return time_since_last.total_seconds() >= self.alert_cooldown_seconds
    
    def _get_alert_cooldown_status(self) -> float:
        """Get seconds since last alert"""
        if self.last_alert_time is None:
            return 0.0
        
        return (datetime.now() - self.last_alert_time).total_seconds()
    
    @staticmethod
    def _id_to_class(class_id: int) -> str:
        """Convert class ID to name"""
        mapping = {
            0: 'Awake',
            1: 'Drowsy',
            2: 'Asleep',
        }
        return mapping.get(class_id, 'Unknown')
    
    @staticmethod
    def _get_reason(class_id: int, confidence: float, temporal_score: float, should_alert: bool) -> str:
        """Generate human-readable reason for prediction"""
        class_name = RealtimeInference._id_to_class(class_id)
        
        if should_alert:
            if class_id == 2:
                return f"ALERT: Eyes closed (confidence: {confidence:.2f})"
            elif class_id == 1:
                return f"ALERT: Persistent drowsiness detected (temporal score: {temporal_score:.2f})"
            return "ALERT: Drowsiness detected"
        
        return f"Driver is {class_name.lower()} (confidence: {confidence:.2f})"
    
    def update_alert_state(self, alert_triggered: bool):
        """Update alert state after processing"""
        if alert_triggered:
            if (
                self.is_alerting
                and self.last_alert_time is not None
                and (datetime.now() - self.last_alert_time).total_seconds() < 0.5
            ):
                return
            self.alert_count += 1
            self.last_alert_time = datetime.now()
            self.is_alerting = True
        else:
            self.is_alerting = False
    
    def get_buffer_statistics(self) -> Dict:
        """Get statistics from current buffers"""
        predictions = list(self.predictions_buffer)
        if len(predictions) == 0:
            return {}
        
        confidences = list(self.confidence_buffer)
        
        return {
            'buffer_size': len(predictions),
            'avg_class': float(np.mean(predictions)),
            'avg_confidence': float(np.mean(confidences)),
            'max_confidence': float(max(confidences)),
            'min_confidence': float(min(confidences)),
            'class_counts': {
                'awake': predictions.count(0),
                'drowsy': predictions.count(1),
                'asleep': predictions.count(2),
            }
        }
    
    def reset(self):
        """Reset all buffers and state"""
        self.predictions_buffer.clear()
        self.confidence_buffer.clear()
        self.features_buffer.clear()
        self.last_alert_time = None
        self.alert_count = 0
        self.is_alerting = False
        self.asleep_start_time = None
