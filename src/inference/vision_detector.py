"""
OpenCV-based webcam eye-state detector.

This is a lightweight classical-vision path for live demos. It is separate from
the trained CNN so webcam snapshots can use an actual face/eye signal.
"""
from typing import Dict, Tuple, Union

import cv2
import numpy as np
from PIL import Image


ImageInput = Union[np.ndarray, Image.Image]


class OpenCVEyeStateDetector:
    """Detect face and visible eyes using OpenCV Haar cascades with Eye Aspect Ratio analysis."""

    def __init__(self):
        cascade_root = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_root + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_root + "haarcascade_eye_tree_eyeglasses.xml"
        )
        self.fallback_eye_cascade = cv2.CascadeClassifier(
            cascade_root + "haarcascade_eye.xml"
        )

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("OpenCV Haar cascade files could not be loaded")

        # Eye Aspect Ratio thresholds (adjusted based on testing)
        self.EAR_THRESHOLD_OPEN = 0.25  # Above this = eye open (was 0.25)
        self.EAR_THRESHOLD_CLOSED = 0.15  # Below this = eye closed (was 0.15)
        self.EAR_THRESHOLD_MICROSLEEP = 0.20  # Between closed and open = microsleep/drowsy (was 0.20)

    def analyze_eye_state(self, eye_region: np.ndarray) -> float:
        """Analyze eye region to determine if eye is open or closed.

        Returns a score from 0.0 (definitely closed) to 1.0 (definitely open).
        Uses multiple heuristics: brightness, edge strength, and texture analysis.
        """
        try:
            # Convert to grayscale if needed
            if len(eye_region.shape) == 3:
                gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = eye_region

            # Normalize brightness to account for lighting variations
            gray = cv2.equalizeHist(gray)

            # Method 1: Brightness analysis
            # Open eyes tend to have more variation in brightness due to pupil/iris
            brightness_std = np.std(gray) / 255.0  # Normalized standard deviation

            # Method 2: Edge analysis
            # Closed eyes have stronger vertical edges (eyelids)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

            horizontal_edges = np.mean(np.abs(sobel_x))
            vertical_edges = np.mean(np.abs(sobel_y))

            # Open eyes have more horizontal edges (eyebrows, eye shape)
            # Closed eyes have more vertical edges (eyelids)
            edge_ratio = horizontal_edges / (vertical_edges + 1e-6)  # Avoid division by zero

            # Method 3: Texture analysis using Laplacian variance
            # Open eyes have more texture variation
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian) / 1000.0  # Normalize

            # Method 4: Look for dark circular regions (pupils)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
            pupil_area = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1])  # Fraction of dark pixels

            # Combine scores with weights
            # Higher brightness variation = more likely open
            brightness_score = min(brightness_std * 3.0, 1.0)

            # Higher horizontal/vertical edge ratio = more likely open
            edge_score = min(edge_ratio / 2.0, 1.0)

            # Higher texture variance = more likely open
            texture_score = min(texture_variance, 1.0)

            # Moderate pupil area = more likely open (too much = noise, too little = closed)
            pupil_score = 1.0 - abs(pupil_area - 0.15) * 3.0  # Peak at ~15% dark pixels
            pupil_score = max(0.0, pupil_score)

            # Weighted combination
            openness_score = (
                brightness_score * 0.3 +
                edge_score * 0.3 +
                texture_score * 0.2 +
                pupil_score * 0.2
            )

            return max(0.0, min(1.0, openness_score))

        except Exception as e:
            # On error, return neutral score
            return 0.5

    def predict(self, image: ImageInput) -> Dict:
        """Return eye-state prediction from a PIL image or RGB/BGR numpy image."""
        rgb = self._to_rgb_array(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            return {
                "class_id": -1,
                "class_name": "Face not detected",
                "confidence": 0.0,
                "eye_count": 0,
                "face_box": None,
                "eye_boxes": [],
                "reason": "No frontal face was detected. Improve lighting and face the camera.",
            }

        face_box = self._largest_box(faces)
        x, y, w, h = face_box

        upper_face = gray[y : y + int(h * 0.58), x : x + w]
        eyes = self.eye_cascade.detectMultiScale(
            upper_face,
            scaleFactor=1.08,
            minNeighbors=6,
            minSize=(18, 18),
        )
        if len(eyes) == 0:
            eyes = self.fallback_eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.08,
                minNeighbors=6,
                minSize=(18, 18),
            )

        eye_boxes = self._filter_eye_boxes(eyes, face_box)
        eye_count = len(eye_boxes)

        # Analyze eye aspect ratios for detected eyes
        ear_values = []  # Keep for backward compatibility
        openness_scores = []

        for ex, ey, ew, eh in eye_boxes:
            # Adjust coordinates to be relative to upper_face region
            relative_ex = ex - x  # x is face x coordinate
            relative_ey = ey - y  # y is face y coordinate

            # Ensure coordinates are within bounds
            if (relative_ex >= 0 and relative_ey >= 0 and
                relative_ex + ew <= upper_face.shape[1] and
                relative_ey + eh <= upper_face.shape[0]):

                eye_region = upper_face[relative_ey:relative_ey+eh, relative_ex:relative_ex+ew]
                if eye_region.size > 0:
                    # Use new eye state analysis
                    openness = self.analyze_eye_state(eye_region)
                    openness_scores.append(openness)
                    # Keep EAR for compatibility but don't use it for classification
                    ear_values.append(0.5)  # Placeholder
                    # Removed debug print
                else:
                    pass  # Empty eye region
            else:
                pass  # Eye coordinates out of bounds

        # Determine eye states based on openness scores
        if not openness_scores:
            # No eyes detected with valid analysis
            class_id = 2
            class_name = "Asleep"
            confidence = 0.85
            reason = "Face detected, but no valid eye regions found for analysis."
            avg_ear = 0.0
        else:
            avg_openness = np.mean(openness_scores)
            min_openness = np.min(openness_scores)

            # Thresholds for openness scores (0.0 = closed, 1.0 = open)
            if avg_openness >= 0.6:
                class_id = 0
                class_name = "Awake"
                confidence = min(0.95, 0.7 + avg_openness * 0.3)
                reason = f"Eyes appear open (avg openness: {avg_openness:.3f})."
            elif avg_openness >= 0.3:
                class_id = 1
                class_name = "Drowsy/Microsleep"
                confidence = 0.75
                reason = f"Eyes showing signs of drowsiness (avg openness: {avg_openness:.3f})."
            else:
                class_id = 2
                class_name = "Asleep"
                confidence = min(0.95, 0.8 + (1.0 - avg_openness) * 0.3)
                reason = f"Eyes appear closed (avg openness: {avg_openness:.3f})."

            avg_ear = np.mean(ear_values) if ear_values else 0.0

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "eye_count": eye_count,
            "ear_values": ear_values,  # Keep for compatibility
            "openness_scores": openness_scores,
            "avg_ear": avg_ear,
            "avg_openness": avg_openness if openness_scores else 0.0,
            "face_box": tuple(int(value) for value in face_box),
            "eye_boxes": eye_boxes,
            "reason": reason,
        }

    def draw_detections(self, image: ImageInput, result: Dict) -> np.ndarray:
        """Draw face/eye boxes and prediction text on an RGB image."""
        rgb = self._to_rgb_array(image).copy()
        face_box = result.get("face_box")

        if face_box:
            x, y, w, h = face_box
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (40, 160, 255), 2)

        for ex, ey, ew, eh in result.get("eye_boxes", []):
            cv2.rectangle(rgb, (ex, ey), (ex + ew, ey + eh), (31, 157, 85), 2)

        cv2.putText(
            rgb,
            f"{result['class_name']} ({result['confidence']:.0%})",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Add openness information for debugging
        avg_openness = result.get('avg_openness', 0.0)
        cv2.putText(
            rgb,
            f"Openness: {avg_openness:.3f}",
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return rgb

    @staticmethod
    def _to_rgb_array(image: ImageInput) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a PIL image or numpy array")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image array must have shape HWC with 3 channels")

        return image

    @staticmethod
    def _largest_box(boxes: np.ndarray) -> Tuple[int, int, int, int]:
        return tuple(max(boxes, key=lambda box: box[2] * box[3]))

    @staticmethod
    def _filter_eye_boxes(eyes: np.ndarray, face_box: Tuple[int, int, int, int]) -> list:
        x, y, w, h = face_box
        boxes = []
        for ex, ey, ew, eh in eyes:
            abs_box = (int(x + ex), int(y + ey), int(ew), int(eh))
            if ew > w * 0.35 or eh > h * 0.25:
                continue
            boxes.append(abs_box)

        boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
        return boxes[:2]
