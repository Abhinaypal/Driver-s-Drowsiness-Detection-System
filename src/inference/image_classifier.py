"""
Image-based inference wrapper for the trained CNN drowsiness model.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from src.inference.classifier import DrowsinessClassifier
from src.models import SimpleDrowsinessCNN
from src.preprocessing import ImageProcessor


class CNNImageClassifier(DrowsinessClassifier):
    """
    Classify drowsiness directly from an image using SimpleDrowsinessCNN.

    Accepted predict inputs:
    - {"image_path": "..."}
    - {"image_tensor": torch.Tensor or np.ndarray}
    - Path/string image path
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        image_processor: Optional[ImageProcessor] = None,
        num_classes: int = 3,
    ):
        super().__init__()
        self.device = self._resolve_device(device)
        self.image_processor = image_processor or ImageProcessor()
        self.model = SimpleDrowsinessCNN(num_classes=num_classes).to(self.device)
        self.class_names = ("Awake", "Drowsy/Microsleep", "Asleep")
        self.is_loaded = True

        if checkpoint_path is not None:
            self.load_model(str(checkpoint_path))

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)

    def load_model(self, model_path: str):
        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.class_names = tuple(checkpoint.get("class_names", self.class_names))
        self.model.eval()
        self.is_loaded = True

    def predict(self, features: Union[Dict, str, Path]) -> Tuple[int, float]:
        image_batch = self._prepare_input(features)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(image_batch)
            probabilities = torch.softmax(logits, dim=1)
            confidence, class_id = torch.max(probabilities, dim=1)

        return int(class_id.item()), float(confidence.item())

    def predict_details(self, features: Union[Dict, str, Path]) -> Dict:
        image_batch = self._prepare_input(features)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(image_batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_id = torch.max(probabilities, dim=0)

        return {
            "class_id": int(class_id.item()),
            "class_name": self.class_names[int(class_id.item())],
            "confidence": float(confidence.item()),
            "probabilities": {
                self.class_names[index]: float(value.item())
                for index, value in enumerate(probabilities)
            },
        }

    def _prepare_input(self, features: Union[Dict, str, Path]) -> torch.Tensor:
        if isinstance(features, (str, Path)):
            return self._image_path_to_batch(Path(features))

        if not isinstance(features, dict):
            raise TypeError("CNNImageClassifier expects an image path or a feature dictionary")

        if "image_path" in features:
            return self._image_path_to_batch(Path(features["image_path"]))

        if "image_tensor" in features:
            return self._tensor_to_batch(features["image_tensor"])

        raise ValueError("Feature dictionary must contain 'image_path' or 'image_tensor'")

    def _image_path_to_batch(self, image_path: Path) -> torch.Tensor:
        image = self.image_processor.preprocess(image_path)
        if image is None:
            raise RuntimeError(f"Failed to preprocess image: {image_path}")
        return self._tensor_to_batch(image)

    def _tensor_to_batch(self, image_tensor: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor)

        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("image_tensor must be a numpy array or torch tensor")

        image_tensor = image_tensor.float()
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.ndim != 4:
            raise ValueError("image_tensor must have shape CHW or BCHW")

        return image_tensor.to(self.device)
