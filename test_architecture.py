"""
Architecture validation and testing script
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import LABELS_DIR, IMAGES_DIR, MANIFESTS_DIR, MODEL_CONFIG
from src.data import AnnotationLoader, DatasetBuilder, DatasetManifestBuilder, load_manifest
from src.preprocessing import ImageProcessor, FeatureExtractor, TemporalFeatureExtractor
from src.inference import CNNImageClassifier, RuleBasedClassifier, RealtimeInference, AlertSystem


def test_manifest_generation():
    """Test manifest generation and reload"""
    print("\nTesting Manifest Generation...")

    try:
        builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR)
        dataset = builder.get_dataset()
        manifest_builder = DatasetManifestBuilder(builder)
        rows = manifest_builder.build_rows()

        assert len(rows) == len(dataset), "Manifest row count mismatch"
        assert rows, "Manifest is empty"
        assert "video_id" in rows[0], "Manifest missing video_id"
        assert "image_rel_path" in rows[0], "Manifest missing image_rel_path"

        manifest_path = MANIFESTS_DIR / "test_dataset_manifest.csv"
        manifest_builder.save(manifest_path, rows)
        loaded_rows = load_manifest(manifest_path)

        assert len(loaded_rows) == len(rows), "Loaded manifest row count mismatch"
        assert loaded_rows[0]["class_id"] in [0, 1, 2], "Loaded manifest class_id invalid"

        print("[PASS] Manifest Generation: CSV manifest created and reloaded")
        return True
    except Exception as e:
        print(f"[FAIL] Manifest Generation failed: {e}")
        return False


def test_model_components():
    """Test CNN model and training dataset adapter"""
    print("\nTesting Model Components...")

    try:
        import torch

        from src.models import SimpleDrowsinessCNN, count_parameters
        from src.training import DrowsinessImageDataset

        builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR)
        dataset = builder.get_dataset()
        torch_dataset = DrowsinessImageDataset(dataset[:2])

        image, label = torch_dataset[0]
        assert image.shape == (3, MODEL_CONFIG['input_size'], MODEL_CONFIG['input_size']), "Invalid image tensor shape"
        assert label.item() in [0, 1, 2], "Invalid label"

        model = SimpleDrowsinessCNN(num_classes=MODEL_CONFIG['num_classes'])
        params = count_parameters(model)
        assert params['trainable'] > 0, "Model has no trainable parameters"

        logits = model(image.unsqueeze(0))
        assert logits.shape == (1, MODEL_CONFIG['num_classes']), "Invalid model output shape"

        probs = torch.softmax(logits, dim=1)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5), "Invalid probabilities"

        image_classifier = CNNImageClassifier(device="cpu")
        class_id, confidence = image_classifier.predict({"image_path": dataset[0]["image_path"]})
        assert class_id in [0, 1, 2], "Invalid CNN classifier class"
        assert 0.0 <= confidence <= 1.0, "Invalid CNN classifier confidence"

        print("[PASS] Model Components: CNN forward pass succeeded")
        return True
    except Exception as e:
        print(f"[FAIL] Model Components failed: {e}")
        return False


def test_data_loading():
    """Test data loading components"""
    print("Testing Data Loading...")
    
    try:
        loader = AnnotationLoader(LABELS_DIR)
        assert len(loader.get_all_keys()) > 0, "No annotations found"
        
        stats = loader.get_statistics()
        assert stats['total_samples'] > 0, "No samples found"
        
        builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR, loader)
        dataset = builder.get_dataset()
        assert len(dataset) > 0, "Dataset is empty"
        
        print(f"[PASS] Data Loading: {len(dataset)} samples loaded")
        return True
    except Exception as e:
        print(f"[FAIL] Data Loading failed: {e}")
        return False


def test_preprocessing():
    """Test preprocessing components"""
    print("\nTesting Preprocessing...")
    
    try:
        processor = ImageProcessor(target_size=(224, 224))
        
        # Test with actual image
        builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR)
        dataset = builder.get_dataset()
        
        if dataset:
            sample = dataset[0]
            image_path = Path(sample['image_path'])
            
            # Test loading
            image = processor.load_image(image_path)
            assert image is not None, "Failed to load image"
            
            # Test resizing
            resized = processor.resize(image)
            assert resized.shape[:2] == (224, 224), "Resize failed"
            
            # Test normalization
            normalized = processor.normalize(resized)
            assert normalized.dtype == float, "Normalization failed"
            
            # Test full pipeline
            processed = processor.preprocess(image_path)
            assert processed.shape == (3, 224, 224), "Full pipeline failed"
        
        print("[PASS] Preprocessing: All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] Preprocessing failed: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction components"""
    print("\nTesting Feature Extraction...")
    
    try:
        builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR)
        dataset = builder.get_dataset()
        
        if dataset:
            sample = dataset[0]
            attributes = sample['attributes']
            
            # Test static features
            features = FeatureExtractor.extract_all_features(attributes)
            assert 'perclos' in features, "PERCLOS not extracted"
            assert 'eye_state' in features, "Eye state not extracted"
            assert 'head_pose' in features, "Head pose not extracted"
            
            # Test eye state conversion
            label = FeatureExtractor.eye_state_to_label(features['eye_state'])
            assert label in [0, 1, 2, -1], "Invalid eye state label"
            
            # Test temporal features
            perclos_list = [0.2, 0.3, 0.4, 0.5]
            eye_states = [0, 0, 1, 1]
            temporal = TemporalFeatureExtractor.extract_temporal_features(
                perclos_list, eye_states
            )
            assert 'perclos_stats' in temporal, "Temporal features failed"
            assert 'blink_frequency' in temporal, "Blink frequency failed"
        
        print("[PASS] Feature Extraction: All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] Feature Extraction failed: {e}")
        return False


def test_inference():
    """Test inference components"""
    print("\nTesting Inference...")
    
    try:
        classifier = RuleBasedClassifier()
        inference = RealtimeInference(classifier, sequence_length=10)
        
        # Test with mock features
        test_features = {
            'perclos': 0.3,
            'eye_state': 'Open',
            'head_pose': {'p': -2.5, 'y': -0.15, 'r': 0.0},
            'zone': 'Zone_On_Road'
        }
        
        result = inference.predict(test_features)
        assert 'class_id' in result, "Missing class_id in result"
        assert 'class_name' in result, "Missing class_name in result"
        assert 'confidence' in result, "Missing confidence in result"
        assert 'should_alert' in result, "Missing should_alert in result"
        
        # Test with multiple predictions
        for i in range(4):  # Add 4 more (1 from above = 5 total)
            inference.predict(test_features)
        
        buffer_stats = inference.get_buffer_statistics()
        assert buffer_stats['buffer_size'] == 5, f"Buffer size is {buffer_stats['buffer_size']}, expected 5"
        
        print("[PASS] Inference: All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] Inference failed: {e}")
        return False


def test_alert_system():
    """Test alert system"""
    print("\nTesting Alert System...")
    
    try:
        alert_system = AlertSystem(
            enable_audio=False,
            enable_visual=False,
            enable_sms=False
        )
        
        # Test alert processing
        result = {
            'class_id': 1,
            'class_name': 'Drowsy',
            'confidence': 0.75,
            'temporal_score': 0.65,
            'should_alert': True,
            'reason': 'High PERCLOS detected'
        }
        
        alert = alert_system.process_inference(result)
        assert alert is not None, "Alert should be generated"
        assert alert['alert_number'] == 1, "Alert count incorrect"
        
        # Test statistics
        stats = alert_system.get_statistics()
        assert stats['total_alerts'] == 1, "Alert count not updated"
        
        print("[PASS] Alert System: All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] Alert System failed: {e}")
        return False


def test_end_to_end():
    """Test complete pipeline"""
    print("\nTesting End-to-End Pipeline...")
    
    try:
        # Load data
        builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR)
        dataset = builder.get_dataset()
        
        # Initialize components
        processor = ImageProcessor()
        classifier = RuleBasedClassifier()
        inference = RealtimeInference(classifier)
        alerts = AlertSystem(enable_audio=False, enable_visual=False)
        
        # Process first 3 samples
        processed_count = 0
        for sample in dataset[:3]:
            # Extract features
            features = FeatureExtractor.extract_all_features(sample['attributes'])
            
            # Inference
            result = inference.predict(features)
            assert result is not None, "Inference failed"
            
            # Alert
            alert = alerts.process_inference(result)
            # Alert may or may not be generated depending on thresholds
            
            processed_count += 1
        
        assert processed_count == 3, "Not all samples processed"
        
        print(f"[PASS] End-to-End Pipeline: {processed_count} samples processed successfully")
        return True
    except Exception as e:
        print(f"[FAIL] End-to-End Pipeline failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("DRIVER DROWSINESS DETECTION SYSTEM - Architecture Tests")
    print("=" * 60)
    
    tests = [
        test_data_loading,
        test_manifest_generation,
        test_preprocessing,
        test_feature_extraction,
        test_model_components,
        test_inference,
        test_alert_system,
        test_end_to_end,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"Unexpected error in {test.__name__}: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Architecture is valid and ready for development.")
        return 0
    else:
        print(f"\n[FAILED] {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
