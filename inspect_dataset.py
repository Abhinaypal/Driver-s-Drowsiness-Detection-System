"""
Deep inspection of the video dataset to verify structure and compatibility.
"""
import os
import sys
from pathlib import Path
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.config import VIDEOS_DIR, IMAGES_DIR, LABELS_DIR

def inspect_data():
    print("="*60)
    print("VIDEO DATASET INSPECTOR")
    print("="*60)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    image_extensions = ('.jpg', '.jpeg', '.png')
    
    # Look for videos in the video dataset root or images directory
    videos = [f for f in VIDEOS_DIR.rglob("**/*") if f.suffix.lower() in video_extensions]
    images = [f for f in IMAGES_DIR.glob("**/*") if f.suffix.lower() in image_extensions]
    labels = list(LABELS_DIR.glob("**/*.json"))

    print(f"Found {len(videos)} video files.")
    print(f"Found {len(images)} image files.")
    print(f"Found {len(labels)} annotation files.")

    total_frames_estimated = 0

    if videos:
        print("\nAnalyzing Video Integrity (First 5 videos):")
        for v_path in videos[:5]:
            cap = cv2.VideoCapture(str(v_path))
            if not cap.isOpened():
                print(f"  [!] FAILED to open: {v_path.name}")
                continue
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Check for matching label
            label_match = any(v_path.stem == l.stem for l in labels)
            status = "✓ Labeled" if label_match else "✗ Missing Label"
            
            print(f"  - {v_path.name}: {width}x{height}, {frame_count} frames @ {fps:.1f}fps [{status}]")
            total_frames_estimated += frame_count
            cap.release()

    # Check for storage usage
    total_size_gb = sum(f.stat().st_size for f in videos + images) / (1024**3)
    print(f"\nTotal Dataset Size on Disk: {total_size_gb:.2f} GB")
    if videos and not images:
        print(f"Estimated total frames across all videos: ~{total_frames_estimated if total_frames_estimated > 0 else 'Unknown'}")
    
    if total_size_gb > 50:
        print("[ADVICE] Your dataset is quite large. Ensure you use the 'manifest' workflow to avoid RAM exhaustion.")

    print("\nNext Recommended Steps:")
    if videos and not images:
        print("1. Your data is in raw video format. You should extract frames to the 'images/' directory.")
        print("2. Run 'python extract_frames.py' to begin extraction.")
    else:
        print("1. Run 'python generate_manifest.py' to index the 100k samples.")
        print("2. Run 'python train_model.py --subsample 0.05' for your first 8GB RAM test.")
    print("="*60)

if __name__ == "__main__":
    if not IMAGES_DIR.exists():
        print(f"Warning: IMAGES_DIR not found at {IMAGES_DIR}. Creating it...")
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    inspect_data()