"""Tests for data preprocessing module"""

import os
import pytest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.data.preprocessing import (
    CatsDogsDataset,
    get_preprocessing_transforms,
    validate_preprocessed_images,
)


class TestPreprocessing:
    """Test preprocessing functions"""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with test images"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            cats_dir = Path(tmpdir) / "cats"
            dogs_dir = Path(tmpdir) / "dogs"
            cats_dir.mkdir(exist_ok=True)
            dogs_dir.mkdir(exist_ok=True)

            # Create dummy images
            for i in range(5):
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(cats_dir / f"cat_{i}.jpg")
                img.save(dogs_dir / f"dog_{i}.jpg")

            yield tmpdir

    def test_dataset_loading(self, temp_data_dir):
        """Test dataset loading"""
        dataset = CatsDogsDataset(temp_data_dir)

        assert len(dataset) == 10  # 5 cats + 5 dogs

        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert label in [0, 1]

    def test_preprocessing_transforms(self):
        """Test preprocessing transforms"""
        train_transform, val_transform = get_preprocessing_transforms()

        assert train_transform is not None
        assert val_transform is not None

        # Test that transforms are callable
        dummy_img = Image.new("RGB", (224, 224))

        train_output = train_transform(dummy_img)
        assert train_output.shape == (3, 224, 224)

        val_output = val_transform(dummy_img)
        assert val_output.shape == (3, 224, 224)

    def test_validate_preprocessed_images(self, temp_data_dir):
        """Test image validation"""
        # All images should be valid (224x224)
        result = validate_preprocessed_images(temp_data_dir, size=(224, 224))
        assert result is True


class TestCatsDogsDataset:
    """Test CatsDogsDataset class"""

    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cats_dir = Path(tmpdir) / "cats"
            dogs_dir = Path(tmpdir) / "dogs"
            cats_dir.mkdir(exist_ok=True)
            dogs_dir.mkdir(exist_ok=True)

            # Create test images
            for i in range(3):
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(cats_dir / f"cat_{i}.jpg")
                img.save(dogs_dir / f"dog_{i}.jpg")

            yield tmpdir

    def test_dataset_len(self, temp_dataset):
        """Test dataset length"""
        dataset = CatsDogsDataset(temp_dataset)
        assert len(dataset) == 6

    def test_dataset_getitem(self, temp_dataset):
        """Test getting an item from dataset"""
        dataset = CatsDogsDataset(temp_dataset)
        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert label in [0, 1]

    def test_dataset_cat_labels(self, temp_dataset):
        """Test that cat images have label 0"""
        dataset = CatsDogsDataset(temp_dataset)

        # First 3 should be cats (label 0)
        for i in range(3):
            _, label = dataset[i]
            assert label == 0

    def test_dataset_dog_labels(self, temp_dataset):
        """Test that dog images have label 1"""
        dataset = CatsDogsDataset(temp_dataset)

        # Last 3 should be dogs (label 1)
        for i in range(3, 6):
            _, label = dataset[i]
            assert label == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
