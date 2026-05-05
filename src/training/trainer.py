"""
Training and evaluation loop for image-based drowsiness classification.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    device: str = "auto"
    checkpoint_path: Path = Path("checkpoints/drowsiness_cnn.pt")
    class_names: tuple = ("Awake", "Drowsy/Microsleep", "Asleep")


class DrowsinessTrainer:
    """Owns model training, validation, testing, and checkpoint saving."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.device = self._resolve_device(self.config.device)
        self.model.to(self.device)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.history: List[Dict[str, float]] = []

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> List[Dict[str, float]]:
        best_val_loss = float("inf")

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
            }

            if val_loader is not None and len(val_loader.dataset) > 0:
                val_metrics = self.evaluate(val_loader)
                row.update(
                    {
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                    }
                )

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(self.config.checkpoint_path, epoch, val_metrics)
            else:
                self.save_checkpoint(self.config.checkpoint_path, epoch, train_metrics)

            self.history.append(row)
            print(self._format_epoch(row))

        return self.history

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        return self._run_epoch(data_loader, training=False)

    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(images.to(self.device))
            return torch.softmax(logits, dim=1)

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "class_names": self.config.class_names,
            },
            path,
        )

    def _run_epoch(self, data_loader: DataLoader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if training:
                    loss.backward()
                    self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

        return {
            "loss": total_loss / total if total else 0.0,
            "accuracy": correct / total if total else 0.0,
        }

    @staticmethod
    def _format_epoch(metrics: Dict[str, float]) -> str:
        parts = [
            f"Epoch {int(metrics['epoch'])}",
            f"train_loss={metrics['train_loss']:.4f}",
            f"train_acc={metrics['train_accuracy']:.2%}",
        ]
        if "val_loss" in metrics:
            parts.extend(
                [
                    f"val_loss={metrics['val_loss']:.4f}",
                    f"val_acc={metrics['val_accuracy']:.2%}",
                ]
            )
        return " - ".join(parts)


def compute_class_weights(samples: Iterable[Dict], num_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for sample in samples:
        class_id = sample.get("class_id", -1)
        if 0 <= class_id < num_classes:
            counts[class_id] += 1

    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights
