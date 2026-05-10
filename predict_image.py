"""
Run image or video-based drowsiness inference with a trained CNN checkpoint.

Example:
    python predict_image.py "video dataset/Dash/Dash/Female/1-FemaleNoGlasses.avi"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import CHECKPOINTS_DIR, INFERENCE_CONFIG, LOGS_DIR
from src.inference import AlertSystem, CNNImageClassifier, RealtimeInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict drowsiness from one image")
    parser.add_argument("image", type=Path, help="Path to an input image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "drowsiness_cnn.pt",
        help="Path to trained CNN checkpoint",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--audio", action="store_true", help="Enable audio alert if triggered")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        return 1
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train one first with: python train_model.py --epochs 10 --batch-size 16")
        return 1

    classifier = CNNImageClassifier(args.checkpoint, device=args.device)
    inference = RealtimeInference(
        classifier,
        sequence_length=INFERENCE_CONFIG["frame_buffer_size"],
        alert_threshold=INFERENCE_CONFIG["drowsiness_threshold"],
        alert_cooldown_seconds=INFERENCE_CONFIG["alert_cooldown_seconds"],
    )
    alerts = AlertSystem(
        log_file=LOGS_DIR / "alerts.log",
        enable_audio=args.audio,
        enable_visual=False,
        enable_sms=False,
    )

    result = inference.predict({"image_path": str(args.image)})
    alert = alerts.process_inference(result)
    if alert:
        inference.update_alert_state(True)

    details = classifier.predict_details({"image_path": str(args.image)})

    print(f"Image: {args.image}")
    print(f"Prediction: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Temporal score: {result['temporal_score']:.2%}")
    print(f"Should alert: {result['should_alert']}")
    print("Probabilities:")
    for class_name, probability in details["probabilities"].items():
        print(f"  {class_name}: {probability:.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
