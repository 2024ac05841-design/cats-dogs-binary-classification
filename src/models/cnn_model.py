"""CNN Model for Cats vs Dogs classification"""

import logging
import torch
import torch.nn as nn
from typing import Tuple

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Simple CNN model for binary image classification"""

    def __init__(self, num_classes: int = 2):
        """
        Initialize SimpleCNN model

        Args:
            num_classes: Number of output classes (default 2 for cats/dogs)
        """
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv Block 1: 224x224 -> 112x112
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv Block 2: 112x112 -> 56x56
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv Block 3: 56x56 -> 28x28
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x


class ResNetBaseline(nn.Module):
    """ResNet-based model for better performance (alternative baseline)"""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize ResNet-based model

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(ResNetBaseline, self).__init__()

        try:
            from torchvision.models import resnet18, ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = resnet18(weights=weights)
        except:
            # Fallback for older PyTorch versions
            from torchvision import models

            self.model = models.resnet18(pretrained=pretrained)

        # Replace final layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)


class LogisticRegressionModel(nn.Module):
    """Logistic Regression model on flattened image features"""

    def __init__(self, num_classes: int = 2, input_size: int = 150528):
        """
        Initialize Logistic Regression model
        Assumes 224x224x3 images flattened = 150528 features

        Args:
            num_classes: Number of output classes
            input_size: Size of flattened input (224*224*3 = 150528)
        """
        super(LogisticRegressionModel, self).__init__()
        self.flatten = nn.Flatten()
        # Simple linear layer (logistic regression)
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.flatten(x)
        x = self.linear(x)
        return x


def create_model(
    model_name: str = "simple_cnn", num_classes: int = 2, device: str = None, pretrained: bool = False
) -> nn.Module:
    """
    Factory function to create model

    Args:
        model_name: Name of model ("simple_cnn", "resnet18", or "logistic_regression")
        num_classes: Number of output classes
        device: Device to put model on
        pretrained: Whether to download ImageNet pretrained weights (set False when loading checkpoint weights)

    Returns:
        Initialized model
    """
    if model_name == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        model = ResNetBaseline(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "logistic_regression":
        model = LogisticRegressionModel(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: simple_cnn, resnet18, logistic_regression"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    logger.info(f"Model {model_name} created on {device}")

    return model


def get_available_models() -> list:
    """
    Get list of available models

    Returns:
        List of model names
    """
    return ["simple_cnn", "logistic_regression", "resnet18"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = SimpleCNN()
    print(model)
    print("Model module loaded")
