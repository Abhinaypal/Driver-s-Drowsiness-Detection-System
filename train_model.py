"""
Train the image-based CNN drowsiness classifier.

Example:
    python train_model.py --epochs 10 --batch-size 16
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import CHECKPOINTS_DIR, DATA_SPLIT, IMAGES_DIR, LABELS_DIR, MODEL_CONFIG
from src.data import DatasetBuilder, load_manifest
from src.models import SimpleDrowsinessCNN, count_parameters
from src.training import (
    DrowsinessTrainer,
    TrainingConfig,
    build_dataloaders,
    build_dataloaders_from_samples,
)
from src.training.trainer import compute_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN drowsiness classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=MODEL_CONFIG["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=MODEL_CONFIG["learning_rate"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path (.csv or .json). If set, training uses manifest rows instead of raw labels.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "drowsiness_cnn.pt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.manifest is not None:
        dataset = load_manifest(args.manifest)
        if not dataset:
            print(f"No samples found in manifest: {args.manifest}")
            return 1

        print(f"Training from manifest: {args.manifest}")
        train_loader, val_loader, test_loader = build_dataloaders_from_samples(
            dataset,
            batch_size=args.batch_size,
            train_ratio=DATA_SPLIT["train"],
            val_ratio=DATA_SPLIT["val"],
            test_ratio=DATA_SPLIT["test"],
            num_workers=args.num_workers,
        )
    else:
        dataset_builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR)
        dataset = dataset_builder.get_dataset()
        if not dataset:
            print("No dataset samples found. Check LABELS_DIR and IMAGES_DIR in src/config.py.")
            return 1

        print("Training from annotation dataset builder")
        train_loader, val_loader, test_loader = build_dataloaders(
            dataset_builder,
            batch_size=args.batch_size,
            train_ratio=DATA_SPLIT["train"],
            val_ratio=DATA_SPLIT["val"],
            test_ratio=DATA_SPLIT["test"],
            num_workers=args.num_workers,
        )

    model = SimpleDrowsinessCNN(num_classes=MODEL_CONFIG["num_classes"])
    params = count_parameters(model)
    print(f"Model parameters: {params['trainable']:,} trainable")

    train_samples = train_loader.dataset.samples
    class_weights = compute_class_weights(train_samples, MODEL_CONFIG["num_classes"])
    print(f"Class weights: {[round(weight.item(), 3) for weight in class_weights]}")

    trainer = DrowsinessTrainer(
        model,
        TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            checkpoint_path=args.checkpoint,
        ),
        class_weights=class_weights,
    )

    trainer.fit(train_loader, val_loader)

    test_metrics = trainer.evaluate(test_loader)
    print(
        "Test metrics: "
        f"loss={test_metrics['loss']:.4f}, "
        f"accuracy={test_metrics['accuracy']:.2%}"
    )
    print(f"Checkpoint saved to: {args.checkpoint}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
