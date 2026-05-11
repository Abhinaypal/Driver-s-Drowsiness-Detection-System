"""
Image preprocessing and feature extraction for drowsiness detection.

This module provides:
- Image loading, resizing, and normalization (ImageProcessor)
- Feature extraction from annotations and images (FeatureExtractor)
- Temporal analysis of feature sequences (TemporalFeatureExtractor)

Key features extracted:
- PERCLOS: Percentage of eyelid closure (0-1, higher = more drowsy)
- Eye State: Open/Drowsy/Closed classification
- Head Pose: Pitch, yaw, roll angles in 3D space
- Temporal Trends: PERCLOS trends, blink frequency, state ratios
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv'}


class ImageProcessor:
    """
    Image preprocessing pipeline for deep learning models.
    
    Handles:
    - Loading images from disk
    - Resizing to target dimensions
    - Normalizing with ImageNet statistics
    - Converting to PyTorch format (channels-first)
    
    Attributes:
        target_size: Target image dimensions (width, height)
        mean: ImageNet normalization mean (R, G, B)
        std: ImageNet normalization std (R, G, B)
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """
        Initialize ImageProcessor with preprocessing parameters.
        
        Args:
            target_size: Target image size (width, height). Default: 224x224 (standard for ResNet)
            mean: Mean values for normalization. Default: ImageNet statistics
            std: Standard deviation for normalization. Default: ImageNet statistics
        """
        self.target_size = target_size
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load image from disk and convert from BGR to RGB.
        
        OpenCV loads images in BGR format by default, but most deep learning
        models expect RGB format. This method handles the conversion.
        
        Args:
            image_path: Path to image file
            
        Returns:
            np.ndarray: Image in RGB format, or None if loading fails
        """
        try:
            image_path = Path(image_path)
            # Load image from disk or extract the first frame from a video file.
            if image_path.suffix.lower() in VIDEO_EXTENSIONS:
                cap = cv2.VideoCapture(str(image_path))
                if not cap.isOpened():
                    logger.error(f"Failed to open video: {image_path}")
                    return None
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    logger.error(f"Failed to read first frame from video: {image_path}")
                    return None
                image = frame
            else:
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    return None
            # Convert BGR to RGB for model input
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(image, self.target_size)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using ImageNet statistics.
        
        This standardizes the input using mean and standard deviation computed
        from millions of ImageNet images. This normalization improves training
        stability and model performance.
        
        Formula: (x - mean) / std for each channel
        
        Args:
            image: Image array (should be in 0-255 range)
            
        Returns:
            np.ndarray: Normalized image in range (-2 to +2 approximately)
        """
        # Convert to float32 and normalize to 0-1 range
        image = image.astype(np.float32) / 255.0
        
        # Apply mean subtraction and std scaling
        image = (image - self.mean) / self.std
        
        return image
    
    def preprocess(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Complete preprocessing pipeline: load -> resize -> normalize -> format.
        
        This is the main method to use for preprocessing images. It handles
        the entire pipeline from disk to model-ready format.
        
        Process:
        1. Load image from disk (BGR -> RGB conversion)
        2. Resize to target dimensions
        3. Normalize with ImageNet statistics
        4. Convert to CHW format (PyTorch standard)
        
        Args:
            image_path: Path to image file
            
        Returns:
            np.ndarray: Preprocessed image in shape (C, H, W), or None if failed
        """
        image_path = Path(image_path)
        # Step 1: Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Step 2: Resize to target dimensions
        image = self.resize(image)
        
        # Step 3: Normalize using ImageNet statistics
        image = self.normalize(image)
        
        # Step 4: Convert to PyTorch format (channels-first: CHW)
        image = np.transpose(image, (2, 0, 1))
        
        return image

    def preprocess_array(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess an already-loaded RGB image array."""
        if image is None:
            return None
        image = self.resize(image)
        image = self.normalize(image)
        return np.transpose(image, (2, 0, 1))


class FeatureExtractor:
    """
    Extract features from images and annotations for drowsiness detection.
    
    This class provides static methods to extract various features:
    - PERCLOS: Eye closure percentage (key drowsiness indicator)
    - Eye State: Classification (Open/Drowsy/Closed)
    - Head Pose: 3D head orientation (pitch, yaw, roll)
    - Zone: Driver location (on road, off road, etc.)
    
    All methods work with annotation dictionaries from the dataset.
    """
    
    @staticmethod
    def extract_perclos(attributes: Dict) -> float:
        """
        Extract PERCLOS (Percentage of Eyelid Closure) from attributes.
        
        PERCLOS is the percentage of time the eyes are closed or nearly closed
        and is one of the most reliable indicators of driver drowsiness.
        
        Values:
        - 0.0 - 0.1: Eyes fully open, alert
        - 0.1 - 0.2: Normal blinking
        - 0.2 - 0.8: Drowsy state
        - 0.8 - 1.0: Asleep or closed eyes
        
        Args:
            attributes: Annotation attributes dictionary
            
        Returns:
            float: PERCLOS value between 0.0 (open) and 1.0 (closed)
        """
        return attributes.get('perclos', 0.0)
    
    @staticmethod
    def extract_eye_state(attributes: Dict) -> str:
        """Extract eye state"""
        return attributes.get('eye_state', 'Unknown')
    
    @staticmethod
    def extract_head_pose(attributes: Dict) -> Dict[str, float]:
        """Extract head pose angles"""
        head_pose = attributes.get('head_pose', {})
        return {
            'pitch': head_pose.get('p', 0.0),   # x-rotation
            'yaw': head_pose.get('y', 0.0),     # y-rotation
            'roll': head_pose.get('r', 0.0),    # z-rotation
        }
    
    @staticmethod
    def extract_zone(attributes: Dict) -> str:
        """Extract driver zone (on road, off road, etc.)"""
        return attributes.get('zone', 'Unknown')
    
    @staticmethod
    def eye_state_to_label(eye_state: str) -> int:
        """
        Convert eye state string to numeric label for ML models.
        
        Converts categorical eye states to numeric labels:
        - 0: Open (Eyes fully open, driver is alert)
        - 1: Drowsy/Microsleep (Eyes partially closed or brief closure)
        - 2: Closed (Eyes fully closed, driver is asleep)
        - -1: Unknown state (unrecognized)
        
        Args:
            eye_state: Eye state string from annotations (e.g., 'Open', 'Closed')
            
        Returns:
            int: Label ID (0, 1, 2, or -1 if unknown)
        """
        state_lower = eye_state.lower()
        
        # Map string states to numeric labels
        if 'open' in state_lower:
            return 0  # Open eyes = alert
        elif 'drowsy' in state_lower or 'microsleep' in state_lower:
            return 1  # Drowsy/Microsleep = warning
        elif 'closed' in state_lower:
            return 2  # Closed eyes = asleep
        else:
            return -1  # Unknown state
    
    @classmethod
    def extract_all_features(cls, attributes: Dict) -> Dict:
        """Extract all available features from attributes"""
        return {
            'perclos': cls.extract_perclos(attributes),
            'eye_state': cls.extract_eye_state(attributes),
            'eye_state_label': cls.eye_state_to_label(cls.extract_eye_state(attributes)),
            'head_pose': cls.extract_head_pose(attributes),
            'zone': cls.extract_zone(attributes),
        }


class TemporalFeatureExtractor:
    """Extract temporal features from sequences of frames"""
    
    @staticmethod
    def compute_perclos_statistics(perclos_values: list) -> Dict:
        """
        Compute statistics from PERCLOS values over time
        
        PERCLOS trends indicate drowsiness progression:
        - Gradually increasing PERCLOS = progressive drowsiness
        - Sudden spikes = microsleep events
        """
        if not perclos_values:
            return {}
        
        perclos_arr = np.array(perclos_values)
        
        return {
            'mean': float(np.mean(perclos_arr)),
            'std': float(np.std(perclos_arr)),
            'max': float(np.max(perclos_arr)),
            'min': float(np.min(perclos_arr)),
            'trend': float(perclos_arr[-1] - perclos_arr[0]),  # Recent trend
        }
    
    @staticmethod
    def compute_blink_frequency(eye_states: list, time_window: float = 1.0) -> float:
        """
        Compute blink frequency (transitions from open to closed)
        
        Lower blink rate indicates drowsiness
        """
        if len(eye_states) < 2:
            return 0.0
        
        blinks = 0
        for i in range(len(eye_states) - 1):
            # Transition from open to closed
            if eye_states[i] == 0 and eye_states[i + 1] >= 1:
                blinks += 1
        
        # Normalize by time window
        return blinks / time_window
    
    @classmethod
    def extract_temporal_features(cls, 
                                  perclos_values: list,
                                  eye_states: list,
                                  time_deltas: Optional[list] = None) -> Dict:
        """
        Extract temporal features from a sequence
        
        Args:
            perclos_values: List of PERCLOS values over time
            eye_states: List of eye state labels (0=open, 1=drowsy, 2=closed)
            time_deltas: Time deltas between frames (optional)
            
        Returns:
            Dictionary of temporal features
        """
        features = {
            'perclos_stats': cls.compute_perclos_statistics(perclos_values),
        }
        
        if time_deltas:
            time_window = sum(time_deltas)
            features['blink_frequency'] = cls.compute_blink_frequency(eye_states, time_window)
        else:
            features['blink_frequency'] = cls.compute_blink_frequency(eye_states, len(eye_states))
        
        # Count drowsy/closed states
        total_states = len(eye_states)
        drowsy_closed = sum(1 for s in eye_states if s >= 1)
        features['drowsy_closed_ratio'] = drowsy_closed / total_states if total_states > 0 else 0.0
        
        return features
