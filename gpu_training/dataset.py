"""
Dataset and transform utilities for GPU training
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)


class CachedCatsDogsDataset(Dataset):
    """
    Dataset loader for Cats vs Dogs images with optional in-memory caching
    """

    def __init__(
        self,
        split_dir: str,
        transform: Optional[transforms.Compose] = None,
        preload: bool = False,
    ):
        """
        Args:
            split_dir: Directory containing 'cats' and 'dogs' subdirectories
            transform: Transformations to apply to images
            preload: If True, preload all PIL images into RAM for fast GPU throughput
        """
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.preload = preload
        self.samples = []
        self.cached_images = {}

        cats_dir = self.split_dir / "cats"
        dogs_dir = self.split_dir / "dogs"

        if cats_dir.exists():
            for img_path in cats_dir.glob("*.jpg"):
                self.samples.append((str(img_path), 0))

        if dogs_dir.exists():
            for img_path in dogs_dir.glob("*.jpg"):
                self.samples.append((str(img_path), 1))

        logger.info(
            f"Loaded {len(self.samples)} samples from {self.split_dir} "
            f"(Cats: {sum(1 for _, l in self.samples if l == 0)}, Dogs: {sum(1 for _, l in self.samples if l == 1)})"
        )

        if self.preload and len(self.samples) > 0:
            logger.info(f"Preloading {len(self.samples)} images into memory for high-speed GPU throughput...")
            for idx, (path, _) in enumerate(self.samples):
                try:
                    img = Image.open(path).convert("RGB")
                    self.cached_images[idx] = img.copy()
                    img.close()
                except Exception as e:
                    logger.warning(f"Could not preload {path}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            if self.preload and idx in self.cached_images:
                image = self.cached_images[idx]
            else:
                image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 224, 224), label


def get_gpu_transforms() -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """
    Get optimized augmentations for GPU training, validation, and testing
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform, eval_transform


def build_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    preload: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Build train, val, and optional test DataLoaders
    """
    train_tf, val_tf, test_tf = get_gpu_transforms()

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    train_dataset = CachedCatsDogsDataset(train_path, transform=train_tf, preload=preload)
    val_dataset = CachedCatsDogsDataset(val_path, transform=val_tf, preload=preload)

    has_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=has_cuda,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=has_cuda,
    )

    test_loader = None
    if os.path.exists(test_path):
        test_dataset = CachedCatsDogsDataset(test_path, transform=test_tf, preload=preload)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=has_cuda,
        )

    return train_loader, val_loader, test_loader
