"""
Streamlit frontend for the Driver Drowsiness Detection System.

Run:
    streamlit run streamlit_app.py
"""
from pathlib import Path
import tempfile
from typing import Dict, Optional

import cv2
import streamlit as st
from PIL import Image

from src.config import CHECKPOINTS_DIR, VIDEOS_DIR, IMAGES_DIR, LABELS_DIRS, LOGS_DIR
from src.data import DatasetBuilder
from src.inference import AlertSystem, CNNImageClassifier, RealtimeInference, RuleBasedClassifier


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


def render_prediction(details: Dict, title: str = "CNN Prediction") -> None:
    class_id = details["class_id"]
    class_name, color = CLASS_BADGES.get(class_id, ("Unknown", "#475467"))

    st.subheader(title)
    st.markdown(
        f"<span class='status-pill' style='background:{color}'>{class_name}</span>",
        unsafe_allow_html=True,
    )
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
                details = cnn.predict_details({"image_path": str(path)})
                render_prediction(details, "CNN prediction from first frame")
        else:
            st.warning("Unable to extract the first frame from this video.")
    else:
        st.image(str(path), caption="Image sample", use_container_width=True)
        if cnn is not None:
            details = cnn.predict_details({"image_path": str(path)})
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


def load_optional_cnn() -> Optional[CNNImageClassifier]:
    checkpoint = CHECKPOINTS_DIR / "drowsiness_cnn.pt"
    if not checkpoint.exists():
        st.sidebar.warning("CNN checkpoint missing: checkpoints/drowsiness_cnn.pt")
        return None

    try:
        return load_cnn_classifier(str(checkpoint))
    except Exception as exc:
        st.sidebar.error(f"Could not load CNN checkpoint: {exc}")
        return None


def main() -> None:
    local_css()

    st.title("Driver Drowsiness Detection System")
    st.caption("Streamlit frontend for the video dataset and model preview.")

    with st.sidebar:
        st.header("Demo Controls")
        page = st.radio(
            "Choose demo",
            ["Video Dataset Explorer", "Upload Video/Image"],
        )
        st.divider()
        st.write("Model checkpoint")
        st.code(str(CHECKPOINTS_DIR / "drowsiness_cnn.pt"))

    dataset_builder = load_dataset()
    cnn = load_optional_cnn()

    if page == "Video Dataset Explorer":
        render_dataset_page(dataset_builder, cnn)
    else:
        render_upload_page(cnn)


if __name__ == "__main__":
    main()
