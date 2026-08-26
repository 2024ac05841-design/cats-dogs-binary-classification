"""Model inference utilities"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Dict
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Class mapping
CLASS_NAMES = {0: "cat", 1: "dog"}


def load_model(
    model_path: str, model_class: nn.Module, device: str = None
) -> nn.Module:
    """
    Load a trained model from disk

    Args:
        model_path: Path to the saved model weights
        model_class: Model class to instantiate
        device: Device to load model on

    Returns:
        Loaded model in evaluation mode
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_class(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from {model_path} on {device}")
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess an image for inference

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    return image


def predict(
    model: nn.Module, image_tensor: torch.Tensor, device: str = None
) -> Tuple[str, float, Dict]:
    """
    Make a prediction on an image

    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on

    Returns:
        Tuple of (class_name, confidence, probabilities_dict)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    confidence = probabilities[predicted_class].item()
    class_name = CLASS_NAMES.get(predicted_class, "unknown")

    probs_dict = {
        CLASS_NAMES.get(i, f"class_{i}"): float(probabilities[i].item())
        for i in range(len(probabilities))
    }

    logger.info(f"Prediction: {class_name} ({confidence:.4f})")

    return class_name, confidence, probs_dict


def predict_from_path(
    model: nn.Module, image_path: str, device: str = None
) -> Tuple[str, float, Dict]:
    """
    Make a prediction directly from an image path

    Args:
        model: Trained model
        image_path: Path to image file
        device: Device to run inference on

    Returns:
        Tuple of (class_name, confidence, probabilities_dict)
    """
    image_tensor = preprocess_image(image_path)
    return predict(model, image_tensor, device)


def batch_predict(model: nn.Module, image_paths: list, device: str = None) -> list:
    """
    Make predictions on multiple images

    Args:
        model: Trained model
        image_paths: List of image paths
        device: Device to run inference on

    Returns:
        List of prediction tuples
    """
    results = []
    for image_path in image_paths:
        try:
            result = predict_from_path(model, image_path, device)
            results.append(
                {
                    "image": image_path,
                    "class": result[0],
                    "confidence": result[1],
                    "probabilities": result[2],
                }
            )
        except Exception as e:
            logger.error(f"Error predicting on {image_path}: {e}")
            results.append({"image": image_path, "error": str(e)})

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Inference module loaded")
