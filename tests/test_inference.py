"""Tests for model inference module"""

import pytest
import torch
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from src.models.cnn_model import SimpleCNN
from src.inference.model_utils import (
    load_model,
    preprocess_image,
    predict,
    predict_from_path,
    CLASS_NAMES
)


class TestModelUtils:
    """Test model utility functions"""
    
    @pytest.fixture
    def model(self):
        """Create a test model"""
        return SimpleCNN(num_classes=2)
    
    @pytest.fixture
    def temp_image(self):
        """Create a temporary test image"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(img_path)
            yield str(img_path)
    
    def test_preprocess_image(self, temp_image):
        """Test image preprocessing"""
        image_tensor = preprocess_image(temp_image)
        
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (1, 3, 224, 224)  # Batch size = 1
    
    def test_predict(self, model, temp_image):
        """Test prediction function"""
        image_tensor = preprocess_image(temp_image)
        class_name, confidence, probs = predict(model, image_tensor)
        
        assert isinstance(class_name, str)
        assert class_name in CLASS_NAMES.values()
        assert 0 <= confidence <= 1
        assert isinstance(probs, dict)
        assert len(probs) == 2
    
    def test_predict_from_path(self, model, temp_image):
        """Test prediction from file path"""
        class_name, confidence, probs = predict_from_path(model, temp_image)
        
        assert isinstance(class_name, str)
        assert class_name in ["cat", "dog"]
        assert 0 <= confidence <= 1
        assert isinstance(probs, dict)
    
    def test_class_names(self):
        """Test CLASS_NAMES mapping"""
        assert CLASS_NAMES[0] == "cat"
        assert CLASS_NAMES[1] == "dog"
        assert len(CLASS_NAMES) == 2


class TestLoadModel:
    """Test model loading"""
    
    def test_load_model_creates_model(self):
        """Test that load_model creates a model"""
        model = SimpleCNN(num_classes=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            torch.save(model.state_dict(), model_path)
            
            loaded_model = load_model(str(model_path), SimpleCNN)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, SimpleCNN)
            assert loaded_model.training is False  # Model should be in eval mode


class TestPredictionConsistency:
    """Test prediction consistency"""
    
    def test_predictions_reproducible(self):
        """Test that predictions are consistent for same input"""
        torch.manual_seed(42)
        model = SimpleCNN(num_classes=2)
        model.eval()
        
        # Create fixed input
        fixed_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output1 = model(fixed_input)
            output2 = model(fixed_input)
        
        assert torch.allclose(output1, output2)


class TestBatchPrediction:
    """Test batch prediction"""
    
    @pytest.fixture
    def temp_images(self):
        """Create multiple temporary test images"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_paths = []
            for i in range(3):
                img_path = Path(tmpdir) / f"test_{i}.jpg"
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(img_path)
                img_paths.append(str(img_path))
            yield img_paths
    
    def test_batch_predict(self, temp_images):
        """Test batch prediction"""
        from src.inference.model_utils import batch_predict
        
        model = SimpleCNN(num_classes=2)
        model.eval()
        
        results = batch_predict(model, temp_images)
        
        assert len(results) == 3
        for result in results:
            assert 'class' in result
            assert 'confidence' in result
            assert 'probabilities' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
