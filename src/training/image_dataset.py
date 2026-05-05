"""
PyTorch dataset adapter for Simuletic DMS image samples.
"""
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.data import DatasetBuilder
from src.preprocessing import ImageProcessor


class DrowsinessImageDataset(Dataset):
    """Convert DatasetBuilder samples into image tensors and class labels."""

    def __init__(
        self,
        samples: Sequence[Dict],
        image_processor: Optional[ImageProcessor] = None,
    ):
        self.samples = list(samples)
        self.image_processor = image_processor or ImageProcessor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = self.image_processor.preprocess(Path(sample["image_path"]))
        if image is None:
            raise RuntimeError(f"Failed to preprocess image: {sample['image_path']}")

        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.tensor(sample["class_id"], dtype=torch.long)
        return image_tensor, label_tensor


def build_dataloaders(
    dataset_builder: DatasetBuilder,
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders from the project dataset."""
    return build_dataloaders_from_samples(
        dataset_builder.get_dataset(),
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        num_workers=num_workers,
        seed=seed,
    )


def build_dataloaders_from_samples(
    samples: Sequence[Dict],
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders from manifest or dataset rows."""
    train_samples, val_samples, test_samples = split_samples(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_dataset = DrowsinessImageDataset(train_samples)
    val_dataset = DrowsinessImageDataset(val_samples)
    test_dataset = DrowsinessImageDataset(test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def split_samples(
    samples: Sequence[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[list, list, list]:
    """Split samples reproducibly into train, validation, and test sets."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    shuffled_samples = list(samples)
    random.Random(seed).shuffle(shuffled_samples)

    total = len(shuffled_samples)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_samples = shuffled_samples[:train_size]
    val_samples = shuffled_samples[train_size:train_size + val_size]
    test_samples = shuffled_samples[train_size + val_size:]

    return train_samples, val_samples, test_samples
