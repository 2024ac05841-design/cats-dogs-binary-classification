import os
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class CatsDogsDataset(Dataset):
    """Custom dataset for Cats and Dogs classification"""

    def __init__(self, image_dir: str, labels_file: str = None, transform=None):
        """
        Args:
            image_dir: Directory containing images organized by class (cats/, dogs/)
            labels_file: Optional CSV file with image paths and labels
            transform: Optional image transformations
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        # Load images from directory structure
        if (self.image_dir / "cats").exists() and (self.image_dir / "dogs").exists():
            # Load cats (label=0)
            for img_path in (self.image_dir / "cats").glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(0)

            # Load dogs (label=1)
            for img_path in (self.image_dir / "dogs").glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(1)

        logger.info(f"Loaded {len(self.images)} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image on error
            return torch.zeros(3, 224, 224), label


def get_preprocessing_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data preprocessing transforms for training and validation

    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def split_dataset(
    image_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> None:
    """
    Split dataset into train/val/test sets

    Args:
        image_dir: Directory containing images
        train_ratio: Ratio for training set (default 0.8)
        val_ratio: Ratio for validation set (default 0.1)
    """
    raw_data_dir = Path(image_dir)

    # Create split directories
    for split in ["train", "val", "test"]:
        for cls in ["cats", "dogs"]:
            (raw_data_dir / split / cls).mkdir(parents=True, exist_ok=True)

    logger.info("Dataset directory structure created")


def validate_preprocessed_images(
    image_dir: str, size: Tuple[int, int] = (224, 224)
) -> bool:
    """
    Validate that all images are preprocessed to correct size

    Args:
        image_dir: Directory containing images
        size: Expected image size (height, width)

    Returns:
        True if all images are valid, False otherwise
    """
    image_dir = Path(image_dir)
    invalid_count = 0

    for img_path in image_dir.rglob("*.jpg"):
        try:
            img = Image.open(img_path)
            if img.size != size[::-1]:  # PIL uses (width, height)
                logger.warning(
                    f"Image {img_path} has size {img.size}, expected {size[::-1]}"
                )
                invalid_count += 1
        except Exception as e:
            logger.error(f"Error validating {img_path}: {e}")
            invalid_count += 1

    logger.info(f"Validation complete. Invalid images: {invalid_count}")
    return invalid_count == 0


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Data preprocessing module loaded")
