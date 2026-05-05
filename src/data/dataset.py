"""
Data loading and annotation handling
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AnnotationLoader:
    """Load and parse JSON annotations from Simuletic DMS dataset"""
    
    def __init__(self, labels_dir: Path):
        """
        Args:
            labels_dir: Path to labels directory containing JSON files
        """
        self.labels_dir = Path(labels_dir)
        self.annotations = {}
        self._load_all_annotations()
    
    def _load_all_annotations(self):
        """Load all JSON annotation files"""
        json_files = list(self.labels_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} annotation files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    self.annotations[json_file.stem] = data
                    logger.info(f"Loaded annotations from {json_file.name}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
    
    def get_samples(self, annotation_key: str) -> List[Dict]:
        """
        Get all samples from an annotation file
        
        Args:
            annotation_key: Key of annotation (e.g., 'driver_full_sleep')
            
        Returns:
            List of samples with image and attributes
        """
        if annotation_key not in self.annotations:
            raise ValueError(f"Annotation key '{annotation_key}' not found")
        
        return self.annotations[annotation_key].get('samples', [])
    
    def get_metadata(self, annotation_key: str) -> Dict:
        """Get metadata for an annotation"""
        if annotation_key not in self.annotations:
            raise ValueError(f"Annotation key '{annotation_key}' not found")
        
        return self.annotations[annotation_key].get('info', {})
    
    def get_all_keys(self) -> List[str]:
        """Get all available annotation keys"""
        return list(self.annotations.keys())
    
    def get_class_label(self, annotation_key: str) -> str:
        """
        Determine class label from annotation key
        
        Returns:
            'awake', 'drowsy', or 'asleep'
        """
        key_lower = annotation_key.lower()
        
        if 'no_sleep' in key_lower or 'awake' in key_lower:
            return 'awake'
        elif 'microsleep' in key_lower or 'drowsy' in key_lower:
            return 'drowsy'
        elif 'full_sleep' in key_lower or 'sleep' in key_lower:
            return 'asleep'
        else:
            return 'unknown'
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
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
    """Build dataset from annotations and images"""
    
    def __init__(self, 
                 labels_dir: Path, 
                 images_dir: Path,
                 annotation_loader: Optional[AnnotationLoader] = None):
        """
        Args:
            labels_dir: Path to labels directory
            images_dir: Path to images directory
            annotation_loader: Pre-loaded annotation loader (optional)
        """
        self.labels_dir = Path(labels_dir)
        self.images_dir = Path(images_dir)
        
        self.annotation_loader = annotation_loader or AnnotationLoader(labels_dir)
        self.dataset = []
        self._build_dataset()
    
    def _build_dataset(self):
        """Build dataset by matching images to annotations"""
        for annotation_key in self.annotation_loader.get_all_keys():
            samples = self.annotation_loader.get_samples(annotation_key)
            class_label = self.annotation_loader.get_class_label(annotation_key)
            class_id = self._class_to_id(class_label)
            
            for sample in samples:
                image_name = sample.get('image')
                image_path = self.images_dir / image_name
                
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                self.dataset.append({
                    'image_path': str(image_path),
                    'image_name': image_name,
                    'annotation_key': annotation_key,
                    'class_label': class_label,
                    'class_id': class_id,
                    'timestamp': sample.get('timestamp'),
                    'attributes': sample.get('attributes', {}),
                })
        
        logger.info(f"Built dataset with {len(self.dataset)} samples")
    
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
    
    def get_dataset(self) -> List[Dict]:
        """Get the built dataset"""
        return self.dataset
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
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
        
        # Shuffle dataset
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
