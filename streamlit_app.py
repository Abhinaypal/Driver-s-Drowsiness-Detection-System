"""
Streamlit frontend for the Driver Drowsiness Detection System.

Run:
    streamlit run streamlit_app.py
"""
from pathlib import Path
import tempfile
import time
from typing import Dict, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.config import CHECKPOINTS_DIR, VIDEOS_DIR, IMAGES_DIR, LABELS_DIRS, LOGS_DIR
from src.data import DatasetBuilder
from src.inference import AlertSystem, CNNImageClassifier, EnsembleCNNImageClassifier, RealtimeInference, RuleBasedClassifier


CLASS_BADGES = {
    0: ("Awake", "#1f9d55"),
    1: ("Drowsy/Microsleep", "#c27803"),
    2: ("Asleep", "#d92d20"),
}


st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="D",
    layout="wide",
)


VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv'}


def local_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.5rem;
            max-width: 1180px;
        }
        .status-pill {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            color: white;
            font-weight: 700;
            font-size: 0.92rem;
        }
        .metric-panel {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.9rem;
            background: #ffffff;
        }
        .small-muted {
            color: #667085;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_dataset() -> DatasetBuilder:
    return DatasetBuilder(LABELS_DIRS, IMAGES_DIR, videos_dir=VIDEOS_DIR)


@st.cache_resource(show_spinner=False)
def load_cnn_classifier(checkpoint_path: str) -> CNNImageClassifier:
    return CNNImageClassifier(checkpoint_path=checkpoint_path, device="cpu")


@st.cache_resource(show_spinner=False)
def load_inference_engine() -> RealtimeInference:
    classifier = RuleBasedClassifier()
    return RealtimeInference(classifier, sequence_length=5, alert_threshold=0.6)


@st.cache_resource(show_spinner=False)
def load_alert_system() -> AlertSystem:
    return AlertSystem(log_file=LOGS_DIR / "dashcam_journey.log", enable_audio=True)


def read_video_first_frame(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        return None

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_face_and_eye_count(rgb_image: np.ndarray) -> tuple[bool, int]:
    """Detect a face and count eyes in the upper face region."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))
    eye_tree_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"))

    if face_cascade.empty() or eye_cascade.empty() or eye_tree_cascade.empty():
        return False, 0

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return False, 0

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    upper_face = gray[y:y + int(h * 0.6), x:x + w]

    eyes = eye_cascade.detectMultiScale(
        upper_face,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(max(12, w // 15), max(10, h // 20)),
    )
    eyes_alt = eye_tree_cascade.detectMultiScale(
        upper_face,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(max(12, w // 15), max(10, h // 20)),
    )
    eye_count = max(len(eyes), len(eyes_alt))
    return True, eye_count


def render_prediction(details: Dict, title: str = "CNN Prediction") -> None:
    class_id = details["class_id"]
    class_name, color = CLASS_BADGES.get(class_id, ("Unknown", "#475467"))

    st.subheader(title)
    st.markdown(
        f"<span class='status-pill' style='background:{color}'>{class_name}</span>",
        unsafe_allow_html=True,
    )
    if details.get("visual_override"):
        st.info(details["visual_override"])
    if details.get("attribute_override"):
        st.info(details["attribute_override"])
    if details.get("ensemble_models"):
        st.caption(f"Ensemble models: {details['ensemble_models']} | Votes: {', '.join(details.get('model_votes', []))}")
    st.metric("Confidence", f"{details['confidence']:.2%}")

    probabilities = details.get("probabilities", {})
    if probabilities:
        st.write("Class probabilities")
        for name, probability in probabilities.items():
            st.progress(float(probability), text=f"{name}: {probability:.2%}")


def dataset_sample_card(sample: Dict, cnn: Optional[CNNImageClassifier]) -> None:
    path = Path(sample['image_path'])
    is_video = path.suffix.lower() in VIDEO_EXTENSIONS

    st.write(f"**Source file:** {path.name}")
    st.write(f"**Derived label:** {sample['class_label'].title()}")

    if is_video:
        st.video(str(path))
        frame = read_video_first_frame(path)
        if frame is not None:
            st.image(frame, caption="First frame preview", use_container_width=True)
            if cnn is not None:
                details = cnn.predict_details({
                    "image_path": str(path),
                    "attributes": sample.get("attributes", {}),
                })
                render_prediction(details, "CNN prediction from first frame")
        else:
            st.warning("Unable to extract the first frame from this video.")
    else:
        st.image(str(path), caption="Image sample", use_container_width=True)
        if cnn is not None:
            details = cnn.predict_details({
                "image_path": str(path),
                "attributes": sample.get("attributes", {}),
            })
            render_prediction(details, "CNN prediction")


def render_dataset_page(dataset_builder: DatasetBuilder, cnn: Optional[CNNImageClassifier]) -> None:
    dataset = dataset_builder.get_dataset()
    stats = dataset_builder.get_statistics()

    st.header("Dataset Explorer")
    st.write("Supports both video and image datasets")
    st.write(f"Video root: {VIDEOS_DIR}")
    st.write(f"Image root: {IMAGES_DIR}")

    top_cols = st.columns(4)
    top_cols[0].metric("Samples", stats["total_samples"])
    top_cols[1].metric("Awake", stats["class_distribution"].get("awake", 0))
    top_cols[2].metric("Drowsy", stats["class_distribution"].get("drowsy", 0))
    top_cols[3].metric("Asleep", stats["class_distribution"].get("asleep", 0))

    if not dataset:
        st.warning("No dataset sample found in the configured paths.")
        return

    options = [
        f"{index + 1}. {Path(sample['image_path']).name} — {sample['class_label']}"
        for index, sample in enumerate(dataset)
    ]
    selection = st.selectbox("Choose a dataset sample", options)
    selected_index = options.index(selection)
    dataset_sample_card(dataset[selected_index], cnn)


def _save_temp_image_bytes(image_bytes: bytes, suffix: str = ".png") -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(image_bytes)
        tmp.flush()
        return Path(tmp.name)
    finally:
        tmp.close()



def render_upload_page(cnn: Optional[CNNImageClassifier]) -> None:
    st.header("Upload Video or Image")
    st.write("Upload a sample video or image and run the CNN on the first frame.")

    uploaded_file = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
    if uploaded_file is None:
        st.info("Upload a driver image or video to run inference.")
        return

    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        video_path = Path(tmp.name)

    if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
        st.video(str(video_path))
        frame = read_video_first_frame(video_path)
        if frame is not None:
            st.image(frame, caption="First frame preview", use_container_width=True)
            if cnn is not None:
                details = cnn.predict_details({"image_path": str(video_path)})
                render_prediction(details, "CNN prediction from first frame")
        else:
            st.warning("Unable to read video frames.")
    else:
        image = Image.open(video_path).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        if cnn is not None:
            details = cnn.predict_details({"image_path": str(video_path)})
            render_prediction(details, "CNN prediction")

    try:
        video_path.unlink()
    except OSError:
        pass


def render_camera_page(cnn: Optional[CNNImageClassifier]) -> None:
    st.header("Driver Drowsiness Detection - Live Webcam")
    st.write(
        "Continuous webcam monitoring with automatic drowsiness alerts. "
        "System will play audio alerts if eyes are closed for 3+ seconds."
    )

    if cnn is None:
        st.warning("No CNN checkpoint loaded. Live webcam alert requires a trained model.")
        return

    # Initialize session state
    if "webcam_streaming" not in st.session_state:
        st.session_state.webcam_streaming = False
    if "music_playing" not in st.session_state:
        st.session_state.music_playing = False
    if "alert_audio_start" not in st.session_state:
        st.session_state.alert_audio_start = None
    if "alert_phase" not in st.session_state:
        st.session_state.alert_phase = None  # None, 'beep', 'music'
    if "closed_eye_start" not in st.session_state:
        st.session_state.closed_eye_start = None

    alert_system = load_alert_system()
    inference_engine = RealtimeInference(
        cnn,
        sequence_length=5,
        alert_threshold=0.5,
        asleep_duration_threshold=2.0,
    )

    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Start Monitoring"):
            st.session_state.webcam_streaming = True
            st.session_state.closed_eye_start = None
    with col2:
        if st.button("⏹ Stop Monitoring"):
            st.session_state.webcam_streaming = False
            st.session_state.music_playing = False
            st.session_state.alert_phase = None
            st.session_state.closed_eye_start = None
            if alert_system.audio_manager:
                alert_system.audio_manager.stop_playback()
    with col3:
        if st.button("🔇 Stop Audio"):
            st.session_state.music_playing = False
            st.session_state.alert_phase = None
            alert_system.audio_manager.stop_playback() if alert_system.audio_manager else None

    # Status placeholders
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    prediction_placeholder = st.empty()
    alert_placeholder = st.empty()

    if not st.session_state.webcam_streaming:
        status_placeholder.info("📹 Press Start Monitoring to begin webcam capture")
        return

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        st.error("❌ Could not open webcam. Check camera connection and permissions.")
        st.session_state.webcam_streaming = False
        return

    status_placeholder.info("🟢 Webcam monitoring active...")
    frame_count = 0
    alert_beep_duration = 10  # Total duration before music escalation (seconds)
    alert_music_escalation_threshold = 0.5  # Escalate to music at 50% of beep duration (5 seconds)

    try:
        while st.session_state.webcam_streaming:
            ret, frame = capture.read()
            if not ret or frame is None:
                status_placeholder.error("❌ Unable to read from webcam.")
                break

            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            _, buffer = cv2.imencode(".jpg", frame)
            image_path = _save_temp_image_bytes(buffer.tobytes(), suffix=".jpg")

            try:
                details = cnn.predict_details({"image_path": str(image_path)})
                result = inference_engine.predict({"image_path": str(image_path)})

                # Eye detection fallback: if face is visible and no eyes are detected for a while,
                # consider the driver asleep even if the CNN predicts awake.
                face_found, eye_count = detect_face_and_eye_count(rgb)
                current_time = time.time()
                if face_found and eye_count == 0:
                    if st.session_state.closed_eye_start is None:
                        st.session_state.closed_eye_start = current_time
                    closed_duration = current_time - st.session_state.closed_eye_start
                else:
                    st.session_state.closed_eye_start = None
                    closed_duration = 0.0

                if closed_duration >= 1.0:
                    result["class_id"] = 2
                    result["class_name"] = "Asleep"
                    result["confidence"] = max(result.get("confidence", 0.0), 0.95)
                    result["probabilities"] = {
                        "Awake": 0.01,
                        "Drowsy/Microsleep": 0.04,
                        "Asleep": 0.95,
                    }
                    details["class_id"] = 2
                    details["class_name"] = "Asleep"
                    details["confidence"] = 0.95
                    details["probabilities"] = result["probabilities"]
                    details["visual_override"] = (
                        "Eyes not detected for over 1 second; overriding CNN to Asleep."
                    )

                frame_placeholder.image(rgb, caption=f"Frame {frame_count}", use_column_width=True)
                with prediction_placeholder.container():
                    render_prediction(details, "CNN Prediction")

                is_asleep = result.get("class_id") == 2

                if face_found:
                    alert_placeholder.info(f"Face detected | Eyes found: {eye_count} | Closed duration: {closed_duration:.1f}s")
                else:
                    alert_placeholder.info("No face detected. Waiting for driver to appear.")

                if is_asleep:
                    if st.session_state.alert_phase is None:
                        st.session_state.alert_phase = "beep"
                        st.session_state.alert_audio_start = current_time
                        alert_placeholder.warning("🔔 ALERT: Eyes Closed - Playing warning beep...")
                        if alert_system.audio_manager:
                            alert_system.audio_manager.play_alert("beep", intensity=1, wait=False)

                    elapsed = current_time - st.session_state.alert_audio_start
                    escalation_time = alert_beep_duration * alert_music_escalation_threshold

                    if st.session_state.alert_phase == "beep" and elapsed >= escalation_time:
                        st.session_state.alert_phase = "music"
                        alert_placeholder.error(f"🎵 CRITICAL: Still asleep after {escalation_time:.1f}s - Playing high intensity music!")
                        if alert_system.audio_manager:
                            alert_system.audio_manager.play_alert("music", intensity=3, wait=False)

                    if st.session_state.alert_phase == "music":
                        alert_placeholder.error(f"🎵 Driver asleep for {elapsed:.1f}s - High intensity alert playing!")

                else:
                    if st.session_state.alert_phase is not None:
                        alert_placeholder.success("✅ Driver awake - alerts stopped")
                        if alert_system.audio_manager:
                            alert_system.audio_manager.stop_playback()
                        st.session_state.alert_phase = None
                    else:
                        alert_placeholder.success("✅ Driver is awake")

            finally:
                try:
                    image_path.unlink()
                except OSError:
                    pass

            time.sleep(0.1)
    finally:
        capture.release()
        if alert_system.audio_manager:
            alert_system.audio_manager.stop_playback()
        status_placeholder.info("⏹ Webcam monitoring stopped")


def load_optional_cnn() -> Optional[CNNImageClassifier]:
    checkpoints = [
        CHECKPOINTS_DIR / "drowsiness_cnn.pt",
        CHECKPOINTS_DIR / "drowsiness_tiny_cnn.pt",
        CHECKPOINTS_DIR / "drowsiness_shallow_cnn.pt",
    ]
    existing_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.exists()]
    if not existing_checkpoints:
        st.sidebar.warning("CNN checkpoints missing. Train models first with train_all_models.py.")
        return None

    try:
        if len(existing_checkpoints) == 1:
            return load_cnn_classifier(str(existing_checkpoints[0]))
        return EnsembleCNNImageClassifier(existing_checkpoints, device="cpu")
    except Exception as exc:
        st.sidebar.error(f"Could not load CNN checkpoint(s): {exc}")
        return None


def main() -> None:
    local_css()

    st.title("Driver Drowsiness Detection System")
    st.caption("Streamlit frontend for the video dataset and model preview.")

    with st.sidebar:
        st.header("Demo Controls")
        page = st.radio(
            "Choose demo",
            ["Upload Video/Image", "Webcam Capture"],
        )
        st.divider()
        st.write("Model checkpoints")
        st.code("\n".join(str(path) for path in [
            CHECKPOINTS_DIR / "drowsiness_cnn.pt",
            CHECKPOINTS_DIR / "drowsiness_tiny_cnn.pt",
            CHECKPOINTS_DIR / "drowsiness_shallow_cnn.pt",
        ]))

    cnn = load_optional_cnn()

    if page == "Upload Video/Image":
        render_upload_page(cnn)
    else:
        render_camera_page(cnn)


if __name__ == "__main__":
    main()
