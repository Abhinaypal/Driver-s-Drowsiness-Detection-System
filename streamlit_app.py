"""
Streamlit frontend for the Driver Drowsiness Detection System.

Run:
    streamlit run streamlit_app.py
"""
from pathlib import Path
import time
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.config import CHECKPOINTS_DIR, IMAGES_DIR, LABELS_DIR
from src.data import AnnotationLoader, DatasetBuilder
from src.inference import (
    CNNImageClassifier,
    OpenCVEyeStateDetector,
    RealtimeInference,
    RuleBasedClassifier,
)
from src.preprocessing import FeatureExtractor, ImageProcessor


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
    loader = AnnotationLoader(LABELS_DIR)
    return DatasetBuilder(LABELS_DIR, IMAGES_DIR, loader)


@st.cache_resource(show_spinner=False)
def load_cnn_classifier(checkpoint_path: str) -> CNNImageClassifier:
    return CNNImageClassifier(checkpoint_path=checkpoint_path, device="cpu")


@st.cache_resource(show_spinner=False)
def load_rule_inference() -> RealtimeInference:
    classifier = RuleBasedClassifier()
    return RealtimeInference(classifier, sequence_length=5, alert_threshold=0.6)


@st.cache_resource(show_spinner=False)
def load_eye_detector() -> OpenCVEyeStateDetector:
    return OpenCVEyeStateDetector()


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to the CHW tensor format expected by CNNImageClassifier."""
    processor = ImageProcessor()
    rgb = np.array(image.convert("RGB"))
    resized = cv2.resize(rgb, processor.target_size)
    normalized = processor.normalize(resized)
    return np.transpose(normalized, (2, 0, 1))


def extract_webcam_features(eye_result: Dict) -> Dict:
    """Extract drowsiness features from OpenCV eye detection results for webcam images."""
    eye_count = eye_result["eye_count"]
    avg_openness = eye_result.get("avg_openness", 0.5)

    # Estimate PERCLOS based on eye detection and openness
    # PERCLOS = percentage of time eyes are closed
    # For single frame: if eyes detected and open = low PERCLOS, if no eyes or closed = high PERCLOS
    if eye_count >= 2:
        # Both eyes detected
        if avg_openness > 0.6:
            perclos = 0.1  # Eyes open
            eye_state = "Open"
        elif avg_openness > 0.3:
            perclos = 0.4  # Eyes drowsy
            eye_state = "Drowsy/Microsleep"
        else:
            perclos = 0.8  # Eyes closed
            eye_state = "Closed"
    elif eye_count == 1:
        # One eye detected - likely drowsy
        perclos = 0.6
        eye_state = "Drowsy/Microsleep"
    else:
        # No eyes detected - likely asleep
        perclos = 0.9
        eye_state = "Closed"

    return {
        "perclos": perclos,
        "eye_state": eye_state,
        "eye_state_label": eye_state,
        "zone": "Webcam",
        "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}  # Default neutral pose
    }


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


def render_eye_detection(result: Dict, title: str = "OpenCV Eye-State Detection") -> None:
    class_id = result["class_id"]
    class_name, color = CLASS_BADGES.get(class_id, (result["class_name"], "#475467"))

    st.subheader(title)
    st.markdown(
        f"<span class='status-pill' style='background:{color}'>{class_name}</span>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Confidence", f"{result['confidence']:.2%}")
    cols[1].metric("Detected Eyes", result["eye_count"])
    cols[2].metric("Avg Openness", f"{result.get('avg_openness', 0.0):.3f}")
    cols[3].metric("Face", "Found" if result.get("face_box") else "Not found")
    st.caption(result["reason"])

    # Show openness scores for each eye if available
    openness_scores = result.get("openness_scores", [])
    if openness_scores:
        st.write("Individual eye openness scores:")
        eye_cols = st.columns(len(openness_scores))
        for i, openness in enumerate(openness_scores):
            eye_cols[i].metric(f"Eye {i+1}", f"{openness:.3f}")


def render_features(features: Dict) -> None:
    head_pose = features["head_pose"]
    st.subheader("Extracted Annotation Features")
    cols = st.columns(4)
    cols[0].metric("PERCLOS", f"{features['perclos']:.3f}")
    cols[1].metric("Eye State", features["eye_state"])
    cols[2].metric("Eye Label", features["eye_state_label"])
    cols[3].metric("Zone", features["zone"])
    st.caption(
        "Head pose: "
        f"pitch={head_pose['pitch']:.2f}, "
        f"yaw={head_pose['yaw']:.2f}, "
        f"roll={head_pose['roll']:.2f}"
    )


def render_dataset_page(dataset_builder: DatasetBuilder, cnn: Optional[CNNImageClassifier]) -> None:
    dataset = dataset_builder.get_dataset()
    stats = dataset_builder.get_statistics()

    st.header("Dataset Demonstration")
    top_cols = st.columns(4)
    top_cols[0].metric("Images", stats["total_samples"])
    top_cols[1].metric("Awake", stats["class_distribution"]["awake"])
    top_cols[2].metric("Drowsy", stats["class_distribution"]["drowsy"])
    top_cols[3].metric("Asleep", stats["class_distribution"]["asleep"])

    labels = [
        f"{index:03d} - {Path(sample['image_path']).name} ({sample['class_label']})"
        for index, sample in enumerate(dataset)
    ]
    selected_label = st.selectbox("Choose a dataset image", labels)
    selected_index = labels.index(selected_label)
    sample = dataset[selected_index]

    left, right = st.columns([1.05, 1])
    with left:
        st.image(sample["image_path"], caption=Path(sample["image_path"]).name, use_container_width=True)
        st.write(f"Ground truth label: **{sample['class_label']}**")
        st.write(f"Timestamp: `{sample['timestamp']}`")

    with right:
        features = FeatureExtractor.extract_all_features(sample["attributes"])
        render_features(features)

        rule_result = load_rule_inference().predict(features)
        st.subheader("Rule-Based Inference")
        st.write(f"Prediction: **{rule_result['class_name']}**")
        st.write(f"Confidence: **{rule_result['confidence']:.2%}**")
        st.write(f"Temporal score: **{rule_result['temporal_score']:.2%}**")
        st.write(f"Alert: **{rule_result['should_alert']}**")

        if cnn is not None:
            details = cnn.predict_details({"image_path": sample["image_path"]})
            render_prediction(details)


def render_upload_page(
    cnn: Optional[CNNImageClassifier],
    eye_detector: OpenCVEyeStateDetector,
) -> None:
    st.header("Image Upload Check")
    st.write("Upload any driver image and compare the webcam-focused eye detector with the saved CNN.")

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Upload an image to run inference.")
        return

    image = Image.open(uploaded_file)
    left, right = st.columns([1, 1])
    eye_result = eye_detector.predict(image)
    annotated = eye_detector.draw_detections(image, eye_result)

    with left:
        st.image(annotated, caption=uploaded_file.name, use_container_width=True)
    with right:
        render_eye_detection(eye_result, "Webcam Eye-State Detection")

        # Extract features and use rule-based classifier
        webcam_features = extract_webcam_features(eye_result)
        rule_classifier = load_rule_inference().classifier
        rule_result = rule_classifier.predict(webcam_features)

        st.subheader("Rule-Based Classification")
        rule_class_name = {0: "Awake", 1: "Drowsy/Microsleep", 2: "Asleep"}.get(rule_result[0], "Unknown")
        st.write(f"Prediction: **{rule_class_name}**")
        st.write(f"Confidence: **{rule_result[1]:.2%}**")

        # Show extracted features
        st.subheader("Extracted Features")
        cols = st.columns(2)
        cols[0].metric("Estimated PERCLOS", f"{webcam_features['perclos']:.2f}")
        cols[1].metric("Eye State", webcam_features['eye_state'])

        if cnn is None:
            st.warning("CNN checkpoint is not available.")
        else:
            render_prediction(predict_pil_image(image, cnn), "Saved CNN Prediction")


def render_webcam_snapshot_page(
    cnn: Optional[CNNImageClassifier],
    eye_detector: OpenCVEyeStateDetector,
) -> None:
    st.header("Webcam Snapshot Check")
    st.write("Capture a frame. The OpenCV detector checks visible eyes; the CNN result is shown as a secondary model.")

    camera_image = st.camera_input("Capture a webcam frame")
    if camera_image is None:
        st.info("Capture a frame to run inference.")
        return

    image = Image.open(camera_image)
    left, right = st.columns([1, 1])
    eye_result = eye_detector.predict(image)
    annotated = eye_detector.draw_detections(image, eye_result)

    with left:
        st.image(annotated, caption="Captured webcam frame", use_container_width=True)
    with right:
        render_eye_detection(eye_result, "Webcam Eye-State Detection")
        if cnn is None:
            st.warning("CNN checkpoint is not available, so only OpenCV eye detection is shown.")
        else:
            render_prediction(predict_pil_image(image, cnn), "Saved CNN Prediction")


def render_live_webcam_page(
    cnn: Optional[CNNImageClassifier],
    eye_detector: OpenCVEyeStateDetector,
) -> None:
    st.header("OpenCV Live Webcam")
    st.write(
        "This mode reads from the webcam connected to the machine running Streamlit. "
        "Use it for a short live demonstration."
    )

    camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    max_frames = st.slider("Frames to process", min_value=30, max_value=300, value=120, step=30)
    analyze_every = st.slider("Analyze every N frames", min_value=1, max_value=15, value=5)
    use_cnn = st.checkbox("Also run saved CNN", value=False, disabled=cnn is None)

    if not st.button("Start live camera"):
        st.info("Click Start live camera to open a short OpenCV webcam session.")
        return

    video_slot = st.empty()
    result_slot = st.empty()
    cap = cv2.VideoCapture(int(camera_index))

    if not cap.isOpened():
        st.error("Could not open webcam. Try a different camera index or use Webcam Snapshot.")
        return

    try:
        latest_eye_result = None
        latest_cnn_details = None
        for frame_index in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                st.warning("Could not read a webcam frame.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_index % analyze_every == 0:
                latest_eye_result = eye_detector.predict(rgb)
                if use_cnn and cnn is not None:
                    latest_cnn_details = predict_pil_image(Image.fromarray(rgb), cnn)

            if latest_eye_result:
                rgb = eye_detector.draw_detections(rgb, latest_eye_result)

                # Get rule-based classification
                webcam_features = extract_webcam_features(latest_eye_result)
                rule_classifier = load_rule_inference().classifier
                rule_result = rule_classifier.predict(webcam_features)
                rule_class_name = {0: "Awake", 1: "Drowsy", 2: "Asleep"}.get(rule_result[0], "Unknown")

                message = (
                    f"OpenCV: {latest_eye_result['class_name']} ({latest_eye_result['confidence']:.0%}), "
                    f"eyes={latest_eye_result['eye_count']} | "
                    f"Rule-Based: {rule_class_name} ({rule_result[1]:.0%})"
                )
                if latest_cnn_details:
                    message += (
                        f" | CNN: {latest_cnn_details['class_name']} ({latest_cnn_details['confidence']:.0%})"
                    )
                result_slot.info(message)

            video_slot.image(rgb, channels="RGB", use_container_width=True)
            time.sleep(0.03)
    finally:
        cap.release()


def load_optional_cnn() -> Optional[CNNImageClassifier]:
    checkpoint = CHECKPOINTS_DIR / "drowsiness_cnn.pt"
    if not checkpoint.exists():
        st.sidebar.error("CNN checkpoint missing: checkpoints/drowsiness_cnn.pt")
        return None

    try:
        return load_cnn_classifier(str(checkpoint))
    except Exception as exc:
        st.sidebar.error(f"Could not load CNN checkpoint: {exc}")
        return None


def main() -> None:
    local_css()

    st.title("Driver Drowsiness Detection System")
    st.caption("Mentor demo frontend for dataset samples, uploaded images, and webcam checks.")

    with st.sidebar:
        st.header("Demo Controls")
        page = st.radio(
            "Choose demo",
            [
                "Dataset Demo",
                "Image Upload",
                "Webcam Snapshot",
                "OpenCV Live Webcam",
            ],
        )
        st.divider()
        st.write("Model checkpoint")
        st.code(str(CHECKPOINTS_DIR / "drowsiness_cnn.pt"))

    dataset_builder = load_dataset()
    cnn = load_optional_cnn()
    eye_detector = load_eye_detector()

    if page == "Dataset Demo":
        render_dataset_page(dataset_builder, cnn)
    elif page == "Image Upload":
        render_upload_page(cnn, eye_detector)
    elif page == "Webcam Snapshot":
        render_webcam_snapshot_page(cnn, eye_detector)
    else:
        render_live_webcam_page(cnn, eye_detector)


if __name__ == "__main__":
    main()
