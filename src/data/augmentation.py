"""Data augmentation utilities for training"""

import logging
from torchvision import transforms

logger = logging.getLogger(__name__)


def get_augmentation_transforms():
    """
    Get augmentation transforms for better model generalization

    Returns:
        transforms.Compose: Augmentation pipeline
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Augmentation module loaded")
