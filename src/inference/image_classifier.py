"""
Image-based inference wrapper for the trained CNN drowsiness model.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.inference.classifier import DrowsinessClassifier
from src.models import ShallowDrowsinessCNN, SimpleDrowsinessCNN, TinyDrowsinessCNN
from src.preprocessing import ImageProcessor


MODEL_ARCHITECTURES = {
    "SimpleDrowsinessCNN": SimpleDrowsinessCNN,
    "TinyDrowsinessCNN": TinyDrowsinessCNN,
    "ShallowDrowsinessCNN": ShallowDrowsinessCNN,
}


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
        architecture = checkpoint.get("architecture", "SimpleDrowsinessCNN") if isinstance(checkpoint, dict) else "SimpleDrowsinessCNN"
        model_class = MODEL_ARCHITECTURES.get(architecture, SimpleDrowsinessCNN)
        if not isinstance(self.model, model_class):
            self.model = model_class(num_classes=len(self.class_names)).to(self.device)
        if isinstance(checkpoint, dict) and "input_size" in checkpoint:
            input_size = int(checkpoint["input_size"])
            self.image_processor = ImageProcessor(target_size=(input_size, input_size))
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.class_names = tuple(checkpoint.get("class_names", self.class_names))
        self.model.eval()
        self.is_loaded = True

    def predict(self, features: Union[Dict, str, Path]) -> Tuple[int, float]:
        details = self.predict_details(features)
        return details["class_id"], details["confidence"]

    def predict_details(self, features: Union[Dict, str, Path]) -> Dict:
        image_path = self._extract_image_path(features)
        image_batch = self._prepare_input(features)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(image_batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_id = torch.max(probabilities, dim=0)

        details = {
            "class_id": int(class_id.item()),
            "class_name": self.class_names[int(class_id.item())],
            "confidence": float(confidence.item()),
            "probabilities": {
                self.class_names[index]: float(value.item())
                for index, value in enumerate(probabilities)
            },
        }
        return self._apply_visual_sanity_check(details, image_path)

    def _extract_image_path(self, features: Union[Dict, str, Path]) -> Optional[Path]:
        if isinstance(features, (str, Path)):
            return Path(features)
        if isinstance(features, dict) and "image_path" in features:
            return Path(features["image_path"])
        return None

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

    def _apply_visual_sanity_check(self, details: Dict, image_path: Optional[Path]) -> Dict:
        if image_path is None or details["class_id"] != 2:
            return details
        if not self._detect_open_eyes(image_path):
            return details

        corrected = dict(details)
        corrected["model_class_id"] = details["class_id"]
        corrected["model_class_name"] = details["class_name"]
        corrected["model_confidence"] = details["confidence"]
        corrected["model_probabilities"] = details["probabilities"]
        corrected["class_id"] = 0
        corrected["class_name"] = self.class_names[0]
        corrected["confidence"] = max(details["probabilities"].get(self.class_names[0], 0.0), 0.80)
        corrected["probabilities"] = {
            self.class_names[0]: corrected["confidence"],
            self.class_names[1]: min(details["probabilities"].get(self.class_names[1], 0.0), 0.10),
            self.class_names[2]: min(details["probabilities"].get(self.class_names[2], 0.0), 0.10),
        }
        corrected["visual_override"] = "Open eyes detected; corrected raw CNN Asleep prediction."
        return corrected

    def _detect_open_eyes(self, image_path: Path) -> bool:
        image = self.image_processor.load_image(image_path)
        if image is None:
            return False

        face_cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        eye_cascades = [
            cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml")),
            cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml")),
        ]
        eye_cascades = [cascade for cascade in eye_cascades if not cascade.empty()]
        if face_cascade.empty() or not eye_cascades:
            return False

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(60, 60))
        if len(faces) == 0:
            return False

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        upper_face = gray[y:y + int(h * 0.6), x:x + w]
        eye_count = 0
        for eye_cascade in eye_cascades:
            eyes = eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(max(10, w // 14), max(8, h // 24)),
            )
            eye_count = max(eye_count, len(eyes))
        return eye_count >= 1


class EnsembleCNNImageClassifier(DrowsinessClassifier):
    """Average probabilities from multiple trained CNN checkpoints."""

    def __init__(
        self,
        checkpoint_paths: List[Union[str, Path]],
        device: str = "auto",
    ):
        super().__init__()
        self.models = [
            CNNImageClassifier(checkpoint_path=path, device=device)
            for path in checkpoint_paths
            if Path(path).exists()
        ]
        if not self.models:
            raise FileNotFoundError("No ensemble checkpoints were found")
        self.class_names = self.models[0].class_names
        self.is_loaded = True

    def predict(self, features: Union[Dict, str, Path]) -> Tuple[int, float]:
        details = self.predict_details(features)
        return details["class_id"], details["confidence"]

    def predict_details(self, features: Union[Dict, str, Path]) -> Dict:
        model_details = [model.predict_details(features) for model in self.models]
        probability_sum = {class_name: 0.0 for class_name in self.class_names}

        for details in model_details:
            for class_name in self.class_names:
                probability_sum[class_name] += details["probabilities"].get(class_name, 0.0)

        probabilities = {
            class_name: probability_sum[class_name] / len(model_details)
            for class_name in self.class_names
        }
        class_name = max(probabilities, key=probabilities.get)
        class_id = self.class_names.index(class_name)
        confidence = probabilities[class_name]

        result = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "probabilities": probabilities,
            "ensemble_models": len(model_details),
            "model_votes": [details["class_name"] for details in model_details],
        }

        attribute_override = next((details.get("attribute_override") for details in model_details if details.get("attribute_override")), None)
        if attribute_override:
            result["attribute_override"] = attribute_override

        override = next((details.get("visual_override") for details in model_details if details.get("visual_override")), None)
        if override and class_id == 2:
            result["class_id"] = 0
            result["class_name"] = self.class_names[0]
            result["confidence"] = max(probabilities.get(self.class_names[0], 0.0), 0.80)
            result["probabilities"] = {
                self.class_names[0]: result["confidence"],
                self.class_names[1]: min(probabilities.get(self.class_names[1], 0.0), 0.10),
                self.class_names[2]: min(probabilities.get(self.class_names[2], 0.0), 0.10),
            }
            result["visual_override"] = override

        return result
