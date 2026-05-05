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
    """Detect face and visible eyes using OpenCV Haar cascades."""

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

        if eye_count >= 2:
            class_id = 0
            class_name = "Awake"
            confidence = 0.88
            reason = "Both eyes are visibly open."
        elif eye_count == 1:
            class_id = 1
            class_name = "Drowsy/Microsleep"
            confidence = 0.68
            reason = "Only one eye was confidently detected."
        else:
            class_id = 2
            class_name = "Asleep"
            confidence = 0.78
            reason = "Face detected, but open eyes were not detected."

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "eye_count": eye_count,
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
