#!/usr/bin/env python3
"""
Comprehensive training script for all deep learning models.
Handles both small and large datasets with automatic augmentation.

Usage:
    # Train all models
    python train_all_models.py --all

    # Train specific models
    python train_all_models.py --model cnn --model lstm

    # For large datasets (>1000 samples)
    python train_all_models.py --all --large-dataset --batch-size 32

    # For small datasets (<500 samples) - uses augmentation
    python train_all_models.py --all --small-dataset --augment
"""
import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

from src.config import CHECKPOINTS_DIR, IMAGES_DIR, LABELS_DIRS, VIDEOS_DIR, MODEL_CONFIG
from src.data import DatasetBuilder, AnnotationLoader
from src.models import ShallowDrowsinessCNN, SimpleDrowsinessCNN, TinyDrowsinessCNN
from src.preprocessing import ImageProcessor, FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
CLASS_NAMES = ("Awake", "Drowsy/Microsleep", "Asleep")
TRAIN_IMAGE_SIZE = 96
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DatasetHelper:
    """Helper for loading and preparing datasets for both small and large scenarios."""

    def __init__(self, images_dir: Path, labels_dir: Union[Path, List[Path]], device: str = "cpu"):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.device = device
        self.image_processor = ImageProcessor(target_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))

    def load_dataset(self) -> Tuple[List, List, List]:
        """Load dataset and return images, labels, and features."""
        images = []
        labels = []
        features_list = []

        loader = AnnotationLoader(LABELS_DIRS, videos_dir=None)
        builder = DatasetBuilder(LABELS_DIRS, self.images_dir, videos_dir=None, annotation_loader=loader)
        annotation_samples = builder.get_dataset()
        logger.info(f"Loading {len(annotation_samples)} annotated image samples...")
        self._append_image_samples(annotation_samples, images, labels, features_list)

        video_samples = self._load_video_frame_samples(frames_per_video=2)
        logger.info(f"Loading {len(video_samples)} sampled video-frame samples...")
        self._append_array_samples(video_samples, images, labels, features_list)

        logger.info(f"Successfully loaded {len(images)} samples")
        return images, labels, features_list

    def _append_image_samples(self, samples: List[Dict], images: List, labels: List, features_list: List) -> None:
        for idx, sample in enumerate(samples):
            if (idx + 1) % 20 == 0:
                logger.info(f"  Processed image sample {idx + 1}/{len(samples)}")
            try:
                img = self.image_processor.preprocess(sample["image_path"])
                if img is None:
                    continue
                images.append(img)
                labels.append(self._class_to_id(sample["class_label"]))
                features_list.append(FeatureExtractor.extract_all_features(sample.get("attributes", {})))
            except Exception as exc:
                logger.warning(f"Failed to load image sample {idx}: {exc}")

    def _append_array_samples(self, samples: List[Dict], images: List, labels: List, features_list: List) -> None:
        for idx, sample in enumerate(samples):
            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed video frame {idx + 1}/{len(samples)}")
            img = self.image_processor.preprocess_array(sample["image"])
            if img is None:
                continue
            images.append(img)
            labels.append(self._class_to_id(sample["class_label"]))
            features_list.append({})

    def _load_video_frame_samples(self, frames_per_video: int = 2) -> List[Dict]:
        samples = []
        if not VIDEOS_DIR.exists():
            return samples

        video_paths = [path for path in VIDEOS_DIR.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS]
        for video_path in video_paths:
            class_label = self._video_class_label(video_path)
            if class_label == "unknown":
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                positions = [0]
            else:
                positions = np.linspace(0.15, 0.75, frames_per_video)
                positions = [min(frame_count - 1, max(0, int(frame_count * pos))) for pos in positions]

            for position in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                samples.append({"image": image, "class_label": class_label})
            cap.release()

        return samples

    @staticmethod
    def _video_class_label(video_path: Path) -> str:
        name = video_path.name.lower()
        if "yawn" in name:
            return "drowsy"
        if "normal" in name or "talking" in name:
            return "awake"
        if "sleep" in name and "no_sleep" not in name:
            return "asleep"
        return "unknown"

    @staticmethod
    def _class_to_id(class_label: str) -> int:
        return {"awake": 0, "drowsy": 1, "asleep": 2}.get(class_label, 0)

    def create_image_tensors(self, images: List, labels: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert images to PyTorch tensors."""
        images_array = np.array(images)  # Shape: (N, C, H, W)
        labels_array = np.array(labels)

        images_tensor = torch.from_numpy(images_array).float().to(self.device)
        labels_tensor = torch.from_numpy(labels_array).long().to(self.device)

        return images_tensor, labels_tensor

    def create_feature_tensors(self, features_list: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert features to PyTorch tensors."""
        features_array = np.zeros((len(features_list), 6), dtype=np.float32)
        
        for i, f in enumerate(features_list):
            # Convert eye_state_label to numeric if string
            eye_state = f.get('eye_state_label', 'Open')
            if isinstance(eye_state, str):
                eye_state_map = {'Open': 0, 'Drowsy/Microsleep': 1, 'Drowsy': 1, 'Microsleep': 1, 'Closed': 2}
                eye_state_num = eye_state_map.get(eye_state, 0)
            else:
                eye_state_num = eye_state
            
            # Convert zone to numeric if string
            zone = f.get('zone', 'Unknown')
            if isinstance(zone, str):
                zone_map = {'Zone_On_Road': 0, 'Zone_Off_Road': 1, 'Unknown': 2}
                zone_num = zone_map.get(zone, 2)
            else:
                zone_num = zone
            
            features_array[i] = [
                float(f.get('perclos', 0.0)),
                float(eye_state_num),
                float(zone_num),
                float(f.get('head_pose', {}).get('pitch', 0.0)),
                float(f.get('head_pose', {}).get('yaw', 0.0)),
                float(f.get('head_pose', {}).get('roll', 0.0))
            ]

        return torch.from_numpy(features_array).float().to(self.device)

    def split_data(self, images: torch.Tensor, labels: torch.Tensor,
                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """Split data into deterministic stratified train/val/test sets."""
        labels_np = labels.detach().cpu().numpy()
        train_indices = []
        val_indices = []
        test_indices = []

        rng = np.random.default_rng(42)
        for class_id in sorted(set(labels_np.tolist())):
            class_indices = np.where(labels_np == class_id)[0]
            rng.shuffle(class_indices)
            train_end = int(len(class_indices) * train_ratio)
            val_end = train_end + int(len(class_indices) * val_ratio)
            train_indices.extend(class_indices[:train_end].tolist())
            val_indices.extend(class_indices[train_end:val_end].tolist())
            test_indices.extend(class_indices[val_end:].tolist())

        train_idx = torch.tensor(train_indices, dtype=torch.long, device=images.device)
        val_idx = torch.tensor(val_indices, dtype=torch.long, device=images.device)
        test_idx = torch.tensor(test_indices, dtype=torch.long, device=images.device)

        return {
            'train': (images[train_idx], labels[train_idx]),
            'val': (images[val_idx], labels[val_idx]),
            'test': (images[test_idx], labels[test_idx]),
        }


class ModelTrainer:
    """Base trainer class for all models."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def train_cnn(
        self,
        train_data: Tuple,
        val_data: Tuple,
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 0.001,
        model_class=SimpleDrowsinessCNN,
        checkpoint_name: str = "drowsiness_cnn.pt",
        model_name: str = "SimpleDrowsinessCNN",
    ) -> str:
        """Train an image CNN."""
        logger.info("=" * 60)
        logger.info(f"Training {model_name}")
        logger.info("=" * 60)

        train_images, train_labels = train_data
        val_images, val_labels = val_data

        model = model_class(num_classes=3).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = optim.Adam(model.parameters(), lr=lr)
        class_counts = torch.bincount(train_labels.detach().cpu(), minlength=3).float()
        class_weights = class_counts.sum() / (len(class_counts) * torch.clamp(class_counts, min=1.0))
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        logger.info(f"Class weights: {[round(weight.item(), 3) for weight in class_weights]}")
        train_loader = DataLoader(
            TensorDataset(train_images, train_labels),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_images, val_labels),
            batch_size=batch_size
        )

        best_val_acc = 0.0
        best_metrics = {}
        history = []
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()

            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            train_acc = train_correct / len(train_labels)
            val_acc = val_correct / len(val_labels)

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss / len(train_loader):.4f}, "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss / len(val_loader):.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_loader),
                "train_accuracy": train_acc,
                "val_loss": val_loss / len(val_loader),
                "val_accuracy": val_acc,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_metrics = history[-1]
                checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "architecture": model_class.__name__,
                        "input_size": TRAIN_IMAGE_SIZE,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": best_metrics,
                        "history": history,
                        "class_names": CLASS_NAMES,
                    },
                    checkpoint_path,
                )
                logger.info(f"✓ Saved checkpoint: {checkpoint_path}")

        return str(CHECKPOINTS_DIR / checkpoint_name)

    def evaluate_cnn(self, checkpoint_path: str, test_data: Tuple, batch_size: int = 16) -> Dict:
        """Evaluate the trained CNN on the held-out test split."""
        test_images, test_labels = test_data
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        architecture = checkpoint.get("architecture", "SimpleDrowsinessCNN") if isinstance(checkpoint, dict) else "SimpleDrowsinessCNN"
        model_class = {
            "SimpleDrowsinessCNN": SimpleDrowsinessCNN,
            "TinyDrowsinessCNN": TinyDrowsinessCNN,
            "ShallowDrowsinessCNN": ShallowDrowsinessCNN,
        }.get(architecture, SimpleDrowsinessCNN)
        model = model_class(num_classes=3).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=batch_size)
        confusion = torch.zeros((3, 3), dtype=torch.long)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images)
                predictions = logits.argmax(1)
                correct += (predictions == labels).sum().item()
                total += labels.numel()
                for target, prediction in zip(labels.detach().cpu(), predictions.detach().cpu()):
                    confusion[int(target), int(prediction)] += 1

        accuracy = correct / total if total else 0.0
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Confusion matrix rows=true cols=pred:\n{confusion.numpy()}")
        return {"accuracy": accuracy, "confusion_matrix": confusion.tolist()}

    def train_lstm(self, train_data: Tuple, val_data: Tuple, epochs: int = 20,
                   batch_size: int = 16, lr: float = 0.001, seq_len: int = 5) -> str:
        """Train LSTMDrowsinessDetector."""
        logger.info("=" * 60)
        logger.info("Training LSTMDrowsinessDetector")
        logger.info("=" * 60)

        train_images, train_labels = train_data
        val_images, val_labels = val_data

        # Create sequences from images (simulate temporal data)
        # For real application, use actual video frames
        train_sequences = self._create_sequences(train_images, seq_len)
        val_sequences = self._create_sequences(val_images, seq_len)

        if len(train_sequences) == 0:
            logger.warning("Not enough samples for LSTM sequences. Skipping LSTM training.")
            return ""

        model = LSTMDrowsinessDetector(num_classes=3, hidden_size=128).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for seq, label in zip(train_sequences, train_labels[:len(train_sequences)]):
                optimizer.zero_grad()
                seq_tensor = seq.unsqueeze(0)  # Add batch dimension
                outputs = model(seq_tensor)
                loss = criterion(outputs, torch.tensor([label], device=self.device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss / len(train_sequences):.4f}")

        checkpoint_path = CHECKPOINTS_DIR / "drowsiness_lstm.pt"
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"✓ Saved checkpoint: {checkpoint_path}")

        return str(checkpoint_path)

    def train_resnet_attention(self, train_data: Tuple, val_data: Tuple, epochs: int = 20,
                               batch_size: int = 16, lr: float = 0.0001) -> str:
        """Train ResNetWithAttention (transfer learning)."""
        logger.info("=" * 60)
        logger.info("Training ResNetWithAttention (Transfer Learning)")
        logger.info("=" * 60)

        train_images, train_labels = train_data
        val_images, val_labels = val_data

        model = ResNetWithAttention(num_classes=3, pretrained=True).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Only train last layers for transfer learning
        for param in model.backbone.parameters():
            param.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(train_images, train_labels),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_images, val_labels),
            batch_size=batch_size
        )

        best_val_acc = 0.0
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()

            model.eval()
            val_correct = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            train_acc = train_correct / len(train_labels)
            val_acc = val_correct / len(val_labels)

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = CHECKPOINTS_DIR / "drowsiness_resnet_attention.pt"
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"✓ Saved checkpoint: {checkpoint_path}")

        return str(CHECKPOINTS_DIR / "drowsiness_resnet_attention.pt")

    @staticmethod
    def _create_sequences(images: torch.Tensor, seq_len: int) -> List[torch.Tensor]:
        """Create sequences for LSTM/GRU training."""
        sequences = []
        for i in range(len(images) - seq_len + 1):
            seq = images[i:i + seq_len]  # Shape: (seq_len, C, H, W)
            sequences.append(seq)
        return sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all drowsiness detection models")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument(
        "--model",
        action="append",
        choices=["cnn", "tiny_cnn", "shallow_cnn"],
        help="Specific model to train (can be repeated)"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--small-dataset",
        action="store_true",
        help="Use settings optimized for small datasets (<500 samples)"
    )
    parser.add_argument(
        "--large-dataset",
        action="store_true",
        help="Use settings optimized for large datasets (>1000 samples)"
    )
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect dataset size if not specified
    if not args.small_dataset and not args.large_dataset:
        sample_count = len(list(IMAGES_DIR.glob("*.jpg")))
        if sample_count < 500:
            logger.info(f"Detected small dataset ({sample_count} samples). Adjusting settings...")
            args.small_dataset = True
            args.augment = True
        else:
            logger.info(f"Detected large dataset ({sample_count} samples). Adjusting settings...")
            args.large_dataset = True

    return args


def main() -> int:
    args = parse_args()
    set_seed(42)

    logger.info("=" * 60)
    logger.info("DROWSINESS DETECTION MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")

    # Create checkpoint directory
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    helper = DatasetHelper(IMAGES_DIR, LABELS_DIRS, args.device)
    images, labels, features = helper.load_dataset()

    if len(images) == 0:
        logger.error("No samples loaded. Aborting.")
        return 1

    logger.info(f"\nDataset Summary:")
    logger.info(f"  Total samples: {len(images)}")
    logger.info(f"  Awake: {labels.count(0)}")
    logger.info(f"  Drowsy: {labels.count(1)}")
    logger.info(f"  Asleep: {labels.count(2)}")

    # Create tensors
    images_tensor, labels_tensor = helper.create_image_tensors(images, labels)
    feature_tensors = helper.create_feature_tensors(features)

    # Split data
    splits = helper.split_data(images_tensor, labels_tensor)
    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(splits['train'][0])} samples")
    logger.info(f"  Val: {len(splits['val'][0])} samples")
    logger.info(f"  Test: {len(splits['test'][0])} samples")

    # Determine which models to train
    models_to_train = []
    if args.all:
        models_to_train = ["cnn", "tiny_cnn", "shallow_cnn"]
        logger.info("Training all implemented neural models: cnn, tiny_cnn, shallow_cnn")
    elif args.model:
        models_to_train = args.model
    else:
        models_to_train = ["cnn"]  # Default to CNN

    # Train models
    trainer = ModelTrainer(device=args.device)
    trained_models = []

    if "cnn" in models_to_train:
        path = trainer.train_cnn(
            splits['train'], splits['val'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            model_class=SimpleDrowsinessCNN,
            checkpoint_name="drowsiness_cnn.pt",
            model_name="SimpleDrowsinessCNN",
        )
        trainer.evaluate_cnn(path, splits['test'], batch_size=args.batch_size)
        trained_models.append(("CNN", path))

    if "tiny_cnn" in models_to_train:
        path = trainer.train_cnn(
            splits['train'], splits['val'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            model_class=TinyDrowsinessCNN,
            checkpoint_name="drowsiness_tiny_cnn.pt",
            model_name="TinyDrowsinessCNN",
        )
        trainer.evaluate_cnn(path, splits['test'], batch_size=args.batch_size)
        trained_models.append(("Tiny CNN", path))

    if "shallow_cnn" in models_to_train:
        path = trainer.train_cnn(
            splits['train'], splits['val'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            model_class=ShallowDrowsinessCNN,
            checkpoint_name="drowsiness_shallow_cnn.pt",
            model_name="ShallowDrowsinessCNN",
        )
        trainer.evaluate_cnn(path, splits['test'], batch_size=args.batch_size)
        trained_models.append(("Shallow CNN", path))

    if "resnet" in models_to_train:
        path = trainer.train_resnet_attention(
            splits['train'], splits['val'],
            epochs=args.epochs // 2,  # Transfer learning converges faster
            batch_size=args.batch_size,
            lr=args.learning_rate / 10  # Lower learning rate for pretrained
        )
        trained_models.append(("ResNet+Attention", path))

    if "lstm" in models_to_train:
        path = trainer.train_lstm(
            splits['train'], splits['val'],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate
        )
        if path:
            trained_models.append(("LSTM", path))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Trained models:")
    for name, path in trained_models:
        logger.info(f"  ✓ {name}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
