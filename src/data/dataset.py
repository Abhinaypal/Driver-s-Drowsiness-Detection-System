"""
Data loading and annotation handling for image and video dataset formats.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv'}
VIDEO_CLASS_KEYWORDS = {
    'no_sleep': 'awake',
    'yawn': 'drowsy',
    'yawning': 'drowsy',
    'talking': 'awake',
    'normal': 'awake',
    'awake': 'awake',
    'sleep': 'asleep',
}


class AnnotationLoader:
    """Load and parse JSON annotations or build a synthetic video annotation index."""

    def __init__(
        self,
        labels_dir: Optional[Path] = None,
        videos_dir: Optional[Path] = None,
    ):
        """
        Args:
            labels_dir: Path or list of paths to label directories containing JSON files
            videos_dir: Path to video dataset root for synthetic annotations
        """
        if labels_dir is None:
            self.labels_dirs = []
        elif isinstance(labels_dir, list):
            self.labels_dirs = [Path(path) for path in labels_dir if path is not None]
        else:
            self.labels_dirs = [Path(labels_dir)]

        self.videos_dir = Path(videos_dir) if videos_dir is not None else None
        self.annotations: Dict[str, Dict] = {}
        self._load_all_annotations()

    def _load_all_annotations(self):
        """Load JSON annotations if available, otherwise scan video files."""
        json_files = []
        for labels_dir in self.labels_dirs:
            if labels_dir.exists():
                json_files.extend(labels_dir.glob("*.json"))

        if json_files:
            logger.info(f"Found {len(json_files)} annotation files")
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        annotation_key = f"{json_file.parent.name}/{json_file.stem}"
                        self.annotations[annotation_key] = data
                        logger.info(f"Loaded annotations from {json_file.name} in {json_file.parent}")
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
        elif self.videos_dir is not None and self.videos_dir.exists():
            self._load_synthetic_video_annotations()
        else:
            logger.warning("No annotation files or video dataset found.")

    def _load_synthetic_video_annotations(self):
        video_files = [
            path for path in self.videos_dir.rglob("*")
            if path.suffix.lower() in VIDEO_EXTENSIONS
        ]

        logger.info(f"Found {len(video_files)} video files for synthetic annotation")
        for video_path in video_files:
            key = video_path.stem
            self.annotations[key] = {
                'samples': [
                    {
                        'video': str(video_path),
                        'video_name': video_path.name,
                        'attributes': {
                            'source_path': str(video_path),
                            'dataset_source': str(video_path.relative_to(self.videos_dir)),
                        },
                    }
                ],
                'info': {
                    'source': str(video_path),
                },
            }

    def get_samples(self, annotation_key: str) -> List[Dict]:
        """Get all samples from an annotation entry."""
        if annotation_key not in self.annotations:
            raise ValueError(f"Annotation key '{annotation_key}' not found")

        return self.annotations[annotation_key].get('samples', [])

    def get_metadata(self, annotation_key: str) -> Dict:
        """Get metadata for an annotation."""
        if annotation_key not in self.annotations:
            raise ValueError(f"Annotation key '{annotation_key}' not found")

        return self.annotations[annotation_key].get('info', {})

    def get_all_keys(self) -> List[str]:
        """Get all available annotation keys."""
        return list(self.annotations.keys())

    def get_class_label(self, annotation_key: str) -> str:
        """Determine class label from annotation key or video filename."""
        key_lower = annotation_key.lower()
        for keyword, label in VIDEO_CLASS_KEYWORDS.items():
            if keyword in key_lower:
                return label
        return 'awake'

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_annotations': len(self.annotations),
            'total_samples': 0,
            'classes': {'awake': 0, 'drowsy': 0, 'asleep': 0},
            'annotation_details': {}
        }

        for key, data in self.annotations.items():
            samples = data.get('samples', [])
            num_samples = len(samples)
            class_label = self.get_class_label(key)

            stats['total_samples'] += num_samples
            stats['classes'][class_label] += num_samples
            stats['annotation_details'][key] = {
                'num_samples': num_samples,
                'class': class_label,
                'metadata': data.get('info', {})
            }

        return stats


class DatasetBuilder:
    """Build dataset from annotations, images, or videos."""

    def __init__(
        self,
        labels_dir: Optional[Path] = None,
        images_dir: Optional[Path] = None,
        videos_dir: Optional[Path] = None,
        annotation_loader: Optional[AnnotationLoader] = None,
    ):
        """Initialize dataset builder."""
        if labels_dir is None:
            self.labels_dirs = []
        elif isinstance(labels_dir, list):
            self.labels_dirs = [Path(path) for path in labels_dir if path is not None]
        else:
            self.labels_dirs = [Path(labels_dir)]

        self.labels_dir = self.labels_dirs[0] if self.labels_dirs else None
        self.images_dir = Path(images_dir) if images_dir is not None else None
        self.videos_dir = Path(videos_dir) if videos_dir is not None else None

        self.annotation_loader = annotation_loader or AnnotationLoader(self.labels_dirs, self.videos_dir)
        self.dataset: List[Dict] = []
        self._build_dataset()

    def _build_dataset(self):
        """Build dataset from annotations, videos, and images."""
        seen_paths = set()
        built = False

        if self.labels_dir is not None and self.labels_dir.exists() and self._has_json_annotations():
            self._build_from_annotations(seen_paths)
            built = True

        if self.videos_dir is not None and self.videos_dir.exists():
            self._build_from_videos(seen_paths)
            built = True

        if self.images_dir is not None and self.images_dir.exists():
            self._build_from_images(seen_paths)
            built = True

        if not built:
            logger.warning("No valid dataset source found. Check VIDEO, IMAGE and LABELS paths.")

        logger.info(f"Built dataset with {len(self.dataset)} samples")

    def _has_json_annotations(self) -> bool:
        return bool(self.labels_dirs and any(labels_dir.exists() and any(labels_dir.glob("*.json")) for labels_dir in self.labels_dirs))

    def _build_from_annotations(self, seen_paths: set):
        for annotation_key in self.annotation_loader.get_all_keys():
            samples = self.annotation_loader.get_samples(annotation_key)
            fallback_class_label = self.annotation_loader.get_class_label(annotation_key)

            for sample in samples:
                class_label = self._class_from_attributes(
                    sample.get('attributes', {}),
                    fallback_class_label,
                )
                class_id = self._class_to_id(class_label)
                image_path = None
                if sample.get('image'):
                    image_path = Path(sample['image'])
                elif sample.get('video'):
                    image_path = Path(sample['video'])
                elif sample.get('video_name'):
                    image_path = Path(sample['video_name'])
                elif sample.get('image_name'):
                    image_path = Path(sample['image_name'])

                if image_path is None:
                    continue

                if not image_path.is_absolute():
                    if self.images_dir is not None and (self.images_dir / image_path).exists():
                        image_path = self.images_dir / image_path
                    elif self.videos_dir is not None and (self.videos_dir / image_path).exists():
                        image_path = self.videos_dir / image_path

                if not image_path.exists():
                    logger.warning(f"Annotated sample file not found: {image_path}")
                    continue

                image_path_str = str(image_path)
                if image_path_str in seen_paths:
                    continue
                seen_paths.add(image_path_str)

                self.dataset.append({
                    'image_path': image_path_str,
                    'image_name': image_path.name,
                    'annotation_key': annotation_key,
                    'class_label': class_label,
                    'class_id': class_id,
                    'timestamp': sample.get('timestamp'),
                    'attributes': sample.get('attributes', {}),
                })

    def _build_from_videos(self, seen_paths: set):
        video_files = self._find_video_files()
        if not video_files:
            logger.warning("No video files found in the configured videos path.")
            return

        for video_path in video_files:
            image_path_str = str(video_path)
            if image_path_str in seen_paths:
                continue
            seen_paths.add(image_path_str)

            annotation_key = str(video_path.relative_to(self.videos_dir)) if self.videos_dir is not None else video_path.stem
            class_label = self.annotation_loader.get_class_label(video_path.stem)
            class_id = self._class_to_id(class_label)

            self.dataset.append({
                'image_path': image_path_str,
                'image_name': video_path.name,
                'annotation_key': annotation_key,
                'class_label': class_label,
                'class_id': class_id,
                'timestamp': None,
                'attributes': {
                    'source_path': image_path_str,
                    'video': True,
                },
            })

    def _build_from_images(self, seen_paths: set):
        image_files = [
            path for path in self.images_dir.rglob("**/*")
            if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ]
        for image_path in image_files:
            image_path_str = str(image_path)
            if image_path_str in seen_paths:
                continue
            seen_paths.add(image_path_str)

            annotation_key = image_path.stem
            class_label = 'awake'
            class_id = self._class_to_id(class_label)

            self.dataset.append({
                'image_path': image_path_str,
                'image_name': image_path.name,
                'annotation_key': annotation_key,
                'class_label': class_label,
                'class_id': class_id,
                'timestamp': None,
                'attributes': {'source_path': image_path_str},
            })

    def _find_video_files(self) -> List[Path]:
        if self.videos_dir is None:
            return []
        return [
            path for path in self.videos_dir.rglob("**/*")
            if path.suffix.lower() in VIDEO_EXTENSIONS
        ]

    @staticmethod
    def _class_to_id(class_label: str) -> int:
        """Convert class label to ID"""
        mapping = {
            'awake': 0,
            'drowsy': 1,
            'asleep': 2,
        }
        return mapping.get(class_label, -1)

    @staticmethod
    def _id_to_class(class_id: int) -> str:
        """Convert class ID to label"""
        mapping = {
            0: 'awake',
            1: 'drowsy',
            2: 'asleep',
        }
        return mapping.get(class_id, 'unknown')

    @staticmethod
    def _class_from_attributes(attributes: Dict, fallback: str) -> str:
        """Derive a per-frame class from annotation attributes when available."""
        eye_state = str(attributes.get('eye_state', '')).lower()
        perclos = float(attributes.get('perclos', 0.0) or 0.0)

        if 'closed' in eye_state:
            return 'asleep'
        if 'drowsy' in eye_state or 'microsleep' in eye_state:
            return 'asleep' if perclos > 0.8 * 0.8 else 'drowsy'
        if 'open' in eye_state:
            if perclos > 0.8 * 0.9:
                return 'asleep'
            if perclos > 0.2 * 1.2:
                return 'drowsy'
            return 'awake'

        return fallback

    def get_dataset(self) -> List[Dict]:
        """Get the built dataset."""
        return self.dataset

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.dataset),
            'class_distribution': {
                'awake': 0,
                'drowsy': 0,
                'asleep': 0,
            },
            'samples_by_annotation': {}
        }

        for sample in self.dataset:
            class_label = sample['class_label']
            if class_label not in stats['class_distribution']:
                stats['class_distribution'][class_label] = 0
            stats['class_distribution'][class_label] += 1

            key = sample['annotation_key']
            if key not in stats['samples_by_annotation']:
                stats['samples_by_annotation'][key] = 0
            stats['samples_by_annotation'][key] += 1

        return stats

    def split_dataset(self,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into train, val, test
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        import random

        dataset = self.dataset.copy()
        random.shuffle(dataset)

        total = len(dataset)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        train_set = dataset[:train_size]
        val_set = dataset[train_size:train_size + val_size]
        test_set = dataset[train_size + val_size:]

        logger.info(f"Dataset split - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        return train_set, val_set, test_set
