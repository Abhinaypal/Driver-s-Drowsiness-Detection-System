"""
Frame extractor for video-based datasets.
Aligns raw videos with the image-based training pipeline.
"""
import cv2
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.config import VIDEOS_DIR, IMAGES_DIR

def main(sample_rate: int = 1):
    """
    Args:
        sample_rate: Extract one frame every N seconds of video.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    # Search for videos in the configured dataset root
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(VIDEOS_DIR.rglob(f"**/*{ext}")))

    if not video_files:
        print("No video files found in the project directory.")
        return

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(video_files)} videos. Extracting to {IMAGES_DIR}...")

    total_saved = 0
    for v_path in tqdm(video_files, desc="Processing Videos"):
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: 
            fps = 30
        
        # Calculate how many frames to skip
        frame_interval = int(fps * sample_rate)
        frame_idx = 0
        video_name = v_path.stem
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only save the frame if it matches our sampling interval
            if frame_idx % frame_interval == 0:
                # Naming convention matches what DatasetBuilder expects
                img_name = f"{video_name}_f{frame_idx}.jpg"
                img_path = IMAGES_DIR / img_name
                
                if not img_path.exists():
                    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    total_saved += 1
            
            frame_idx += 1
        
        cap.release()

    print(f"\nSuccessfully extracted {total_saved} frames.")
    print("Next steps:")
    print("1. Ensure your JSON labels match the video filenames.")
    print("2. Run 'python generate_manifest.py'")

if __name__ == "__main__":
    # Adjust sample_rate: 1 = 1 frame/sec, 0.5 = 2 frames/sec
    main(sample_rate=1)