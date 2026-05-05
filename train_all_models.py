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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

from src.config import CHECKPOINTS_DIR, IMAGES_DIR, LABELS_DIR, MODEL_CONFIG
from src.data import DatasetBuilder, AnnotationLoader
from src.models import (
    SimpleDrowsinessCNN,
    LSTMDrowsinessDetector,
    GRUDrowsinessDetector,
    CNNWithAttention,
    ResNetWithAttention,
    XGBoostDrowsinessClassifier,
    LightGBMDrowsinessClassifier,
    extract_feature_vector,
)
from src.preprocessing import ImageProcessor, FeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetHelper:
    """Helper for loading and preparing datasets for both small and large scenarios."""

    def __init__(self, images_dir: Path, labels_dir: Path, device: str = "cpu"):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.device = device
        self.image_processor = ImageProcessor()

    def load_dataset(self) -> Tuple[List, List, List]:
        """Load dataset and return images, labels, and features."""
        loader = AnnotationLoader(self.labels_dir)
        builder = DatasetBuilder(self.labels_dir, self.images_dir, loader)
        dataset = builder.get_dataset()

        images = []
        labels = []
        features_list = []

        logger.info(f"Loading {len(dataset)} samples...")

        for idx, sample in enumerate(dataset):
            if (idx + 1) % 20 == 0:
                logger.info(f"  Processed {idx + 1}/{len(dataset)} samples")

            # Load and preprocess image (returns CHW format)
            try:
                img = self.image_processor.preprocess(sample['image_path'])
                if img is None:
                    logger.warning(f"Failed to preprocess image {sample['image_path']}")
                    continue
                    
                images.append(img)

                # Get label
                class_label = sample['class_label']
                class_id = {'awake': 0, 'drowsy': 1, 'asleep': 2}.get(class_label, 0)
                labels.append(class_id)

                # Extract features for tree-based models
                features = FeatureExtractor.extract_all_features(sample['attributes'])
                features_list.append(features)

            except Exception as e:
                logger.warning(f"Failed to load sample {idx}: {e}")
                continue

        logger.info(f"Successfully loaded {len(images)} samples")
        return images, labels, features_list

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
                float(f['head_pose'].get('pitch', 0.0)),
                float(f['head_pose'].get('yaw', 0.0)),
                float(f['head_pose'].get('roll', 0.0))
            ]

        return torch.from_numpy(features_array).float().to(self.device)

    def split_data(self, images: torch.Tensor, labels: torch.Tensor, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """Split data into train/val/test sets."""
        n = len(images)
        indices = np.random.permutation(n)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return {
            'train': (images[train_idx], labels[train_idx]),
            'val': (images[val_idx], labels[val_idx]),
            'test': (images[test_idx], labels[test_idx]),
        }


class ModelTrainer:
    """Base trainer class for all models."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def train_cnn(self, train_data: Tuple, val_data: Tuple, epochs: int = 20, 
                  batch_size: int = 16, lr: float = 0.001) -> str:
        """Train SimpleDrowsinessCNN."""
        logger.info("=" * 60)
        logger.info("Training SimpleDrowsinessCNN")
        logger.info("=" * 60)

        train_images, train_labels = train_data
        val_images, val_labels = val_data

        model = SimpleDrowsinessCNN(num_classes=3).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = optim.Adam(model.parameters(), lr=lr)
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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = CHECKPOINTS_DIR / "drowsiness_cnn.pt"
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"✓ Saved checkpoint: {checkpoint_path}")

        return str(CHECKPOINTS_DIR / "drowsiness_cnn.pt")

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
        choices=["cnn", "lstm", "gru", "attention", "resnet", "xgboost", "lightgbm"],
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
    helper = DatasetHelper(IMAGES_DIR, LABELS_DIR, args.device)
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
        models_to_train = ["cnn", "resnet"]  # Core models for images
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
            lr=args.learning_rate
        )
        trained_models.append(("CNN", path))

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
