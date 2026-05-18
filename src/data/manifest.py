"""
Dataset manifest generation and loading utilities.
"""
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config import PROJECT_ROOT
from src.data.dataset import DatasetBuilder
from src.preprocessing import FeatureExtractor

MANIFEST_FIELDS = [
    "image_path",
    "image_rel_path",
    "image_name",
    "annotation_key",
    "label_file",
    "video_id",
    "source_video",
    "class_id",
    "class_label",
    "timestamp",
    "perclos",
    "eye_state",
    "eye_state_label",
    "zone",
    "head_pitch",
    "head_yaw",
    "head_roll",
]


class DatasetManifestBuilder:
    """Build flat manifest rows from the current dataset structure."""

    def __init__(self, dataset_builder: DatasetBuilder, project_root: Optional[Path] = None):
        self.dataset_builder = dataset_builder
        self.project_root = Path(project_root or PROJECT_ROOT)

    def build_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for sample in self.dataset_builder.get_dataset():
            annotation_key = sample["annotation_key"]
            metadata = self._metadata_for_sample(annotation_key)
            attributes = sample.get("attributes", {})
            source_video = metadata.get("source", "")
            if not source_video and attributes.get("video"):
                source_video = attributes.get("source_path", sample["image_path"])
            video_id = Path(source_video).stem if source_video else annotation_key
            head_pose = FeatureExtractor.extract_head_pose(attributes)
            label_file = f"{annotation_key}.json" if metadata else ""

            rows.append(
                {
                    "image_path": str(Path(sample["image_path"]).resolve()),
                    "image_rel_path": self._to_relative_path(sample["image_path"]),
                    "image_name": sample["image_name"],
                    "annotation_key": annotation_key,
                    "label_file": label_file,
                    "video_id": video_id,
                    "source_video": source_video,
                    "class_id": sample["class_id"],
                    "class_label": sample["class_label"],
                    "timestamp": sample.get("timestamp"),
                    "perclos": FeatureExtractor.extract_perclos(attributes),
                    "eye_state": FeatureExtractor.extract_eye_state(attributes),
                    "eye_state_label": FeatureExtractor.eye_state_to_label(
                        FeatureExtractor.extract_eye_state(attributes)
                    ),
                    "zone": FeatureExtractor.extract_zone(attributes),
                    "head_pitch": head_pose["pitch"],
                    "head_yaw": head_pose["yaw"],
                    "head_roll": head_pose["roll"],
                }
            )

        return rows

    def _metadata_for_sample(self, annotation_key: str) -> Dict[str, Any]:
        try:
            return self.dataset_builder.annotation_loader.get_metadata(annotation_key)
        except ValueError:
            return {}

    def save(self, output_path: Path, rows: Optional[Iterable[Dict[str, Any]]] = None) -> Path:
        manifest_rows = list(rows) if rows is not None else self.build_rows()
        return save_manifest(manifest_rows, output_path)

    def _to_relative_path(self, image_path: str) -> str:
        image_path = Path(image_path)
        try:
            return str(image_path.resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return str(image_path)


def save_manifest(rows: Iterable[Dict[str, Any]], output_path: Path) -> Path:
    """Save manifest rows to CSV or JSON based on the output suffix."""
    rows = list(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        return output_path

    if output_path.suffix.lower() != ".csv":
        raise ValueError("Manifest output must end with .csv or .json")

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in MANIFEST_FIELDS})

    return output_path


def load_manifest(input_path: Path) -> List[Dict[str, Any]]:
    """Load manifest rows from CSV or JSON."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Manifest not found: {input_path}")

    if input_path.suffix.lower() == ".json":
        with open(input_path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        return [_coerce_row_types(row) for row in rows]

    if input_path.suffix.lower() != ".csv":
        raise ValueError("Manifest input must end with .csv or .json")

    with open(input_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_coerce_row_types(row) for row in reader]


def summarize_manifest(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Return simple summary statistics for a manifest."""
    summary = {
        "total_rows": 0,
        "class_distribution": {},
        "video_count": 0,
    }
    video_ids = set()

    for row in rows:
        summary["total_rows"] += 1
        class_label = row.get("class_label", "unknown")
        summary["class_distribution"][class_label] = summary["class_distribution"].get(class_label, 0) + 1

        video_id = row.get("video_id")
        if video_id:
            video_ids.add(video_id)

    summary["video_count"] = len(video_ids)
    return summary


def _coerce_row_types(row: Dict[str, Any]) -> Dict[str, Any]:
    coerced = dict(row)
    int_fields = {"class_id", "eye_state_label"}
    float_fields = {"timestamp", "perclos", "head_pitch", "head_yaw", "head_roll"}

    for field in int_fields:
        value = coerced.get(field)
        if value in ("", None):
            coerced[field] = None
        else:
            coerced[field] = int(value)

    for field in float_fields:
        value = coerced.get(field)
        if value in ("", None):
            coerced[field] = None
        else:
            coerced[field] = float(value)

    return coerced
