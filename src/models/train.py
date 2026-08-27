"""Training script for Cats vs Dogs classifier"""

import os
import shutil
import logging
import json
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from .cnn_model import create_model
from ..data import CatsDogsDataset, get_preprocessing_transforms

logger = logging.getLogger(__name__)


def setup_mlflow_experiment():
    """Setup MLFlow experiment - called at runtime instead of import time"""
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "cats-dogs-classification")
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"Set MLFlow experiment to '{experiment_name}'")
    except Exception as e:
        logger.warning(f"Could not set MLFlow experiment '{experiment_name}': {e}")
        try:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            logger.info(f"Created and set MLFlow experiment '{experiment_name}'")
        except Exception as e2:
            logger.warning(f"Could not create MLFlow experiment: {e2}. Continuing without MLFlow.")


class Trainer:
    """Trainer class for model training"""

    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        model_name: str = "simple_cnn",
        experiment_name: str = "baseline",
    ):
        """
        Initialize trainer

        Args:
            model: PyTorch model to train
            device: Device to train on
            model_name: Name of the model
            experiment_name: MLflow experiment name
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(
        self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(
        self, val_loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate the model

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 0.001,
        save_path: str = "models/model.pkl",
    ) -> Dict:
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            save_path: Path to save the model

        Returns:
            Training history
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        # Helper function to log to MLFlow with error handling
        def safe_mlflow_log(func, *args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.debug(f"MLFlow logging failed: {e}")
                pass
        
        # Start MLflow run with error handling
        try:
            mlflow_run = mlflow.start_run(run_name=self.experiment_name)
            mlflow_run.__enter__()
            mlflow_available = True
        except Exception as e:
            logger.warning(f"Could not start MLFlow run: {e}. Training without MLFlow logging.")
            mlflow_available = False
        
        if mlflow_available:
            safe_mlflow_log(mlflow.log_param, "model", self.model_name)
            safe_mlflow_log(mlflow.log_param, "epochs", epochs)
            safe_mlflow_log(mlflow.log_param, "learning_rate", lr)
            safe_mlflow_log(mlflow.log_param, "batch_size", train_loader.batch_size)

        best_val_loss = float("inf")
        best_model_path = None

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion
            )
            val_loss, val_acc = self.validate(val_loader, criterion)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Log to MLflow
            safe_mlflow_log(mlflow.log_metric, "train_loss", train_loss, step=epoch)
            safe_mlflow_log(mlflow.log_metric, "train_acc", train_acc, step=epoch)
            safe_mlflow_log(mlflow.log_metric, "val_loss", val_loss, step=epoch)
            safe_mlflow_log(mlflow.log_metric, "val_acc", val_acc, step=epoch)

            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = save_path
                self.save_model(save_path)
                logger.info(f"Saved best model to {save_path}")

            scheduler.step(val_loss)

        # Log final metrics and model
        safe_mlflow_log(mlflow.log_metric, "best_val_loss", best_val_loss)
        safe_mlflow_log(mlflow.pytorch.log_model, self.model, "model")
        
        if mlflow_available:
            try:
                mlflow.end_run()
            except:
                pass
        
        return self.history

    def save_model(self, path: str) -> None:
        """Save model to disk"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model from disk"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")


def train_model(
    data_dir: str = "data/processed",
    model_name: str = "simple_cnn",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    save_path: str = "models/model.pkl",
) -> None:
    """
    Complete training pipeline

    Args:
        data_dir: Directory containing training data
        model_name: Name of model to use
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        save_path: Path to save trained model
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    train_transform, val_transform = get_preprocessing_transforms()

    train_data = CatsDogsDataset(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_data = CatsDogsDataset(os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    logger.info(f"Training set size: {len(train_data)}")
    logger.info(f"Validation set size: {len(val_data)}")

    # Create model
    model = create_model(model_name=model_name, device=device)

    # Train
    trainer = Trainer(model, device=device, model_name=model_name)
    history = trainer.train(
        train_loader, val_loader, epochs=epochs, lr=lr, save_path=save_path
    )

    logger.info("Training completed!")

    # Save history
    history_path = Path(save_path).parent / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")


def train_multiple_models(
    data_dir: str = "data/processed",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    model_names: list = None,
    best_model_dir: str = "models/best_model",
) -> Dict[str, Dict]:
    """
    Train multiple models, compare them, and save the best one

    Args:
        data_dir: Directory containing training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        model_names: List of model names to train
        best_model_dir: Directory to save best model

    Returns:
        Dictionary with results for all models and best model info
    """
    if model_names is None:
        model_names = ["simple_cnn", "logistic_regression", "resnet18"]

    logging.basicConfig(level=logging.INFO)
    
    # Setup MLFlow at runtime instead of at import time
    setup_mlflow_experiment()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    train_transform, val_transform = get_preprocessing_transforms()

    train_data = CatsDogsDataset(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_data = CatsDogsDataset(os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    logger.info(f"Training set size: {len(train_data)}")
    logger.info(f"Validation set size: {len(val_data)}")

    # Store results
    results = {}
    best_model_info = {"name": None, "val_loss": float("inf"), "val_acc": 0.0}

    # Train each model
    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} model...")
        logger.info(f"{'='*60}")

        try:
            # Create best_model_dir if it doesn't exist
            Path(best_model_dir).mkdir(parents=True, exist_ok=True)
            
            # Use best_model_dir for temporary model files
            temp_path = os.path.join(best_model_dir, f"temp_{model_name}.pkl")

            model = create_model(model_name=model_name, device=device)
            trainer = Trainer(model, device=device, model_name=model_name)
            history = trainer.train(
                train_loader, val_loader, epochs=epochs, lr=lr, save_path=temp_path
            )

            # Store results
            results[model_name] = {
                "val_loss": float(history["val_loss"][-1]),
                "val_acc": float(history["val_acc"][-1]),
                "train_loss": float(history["train_loss"][-1]),
                "train_acc": float(history["train_acc"][-1]),
            }

            # Check if this is the best model
            current_val_acc = results[model_name]["val_acc"]
            if current_val_acc > best_model_info["val_acc"]:
                best_model_info = {
                    "name": model_name,
                    "val_loss": results[model_name]["val_loss"],
                    "val_acc": current_val_acc,
                    "train_loss": results[model_name]["train_loss"],
                    "train_acc": results[model_name]["train_acc"],
                    "temp_path": temp_path,
                }

            logger.info(f"Completed training {model_name}")
            logger.info(f"Val Acc: {current_val_acc:.2f}%, Val Loss: {results[model_name]['val_loss']:.4f}")

        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}

    # Save best model
    if best_model_info["name"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Best model: {best_model_info['name']}")
        logger.info(f"Validation Accuracy: {best_model_info['val_acc']:.2f}%")
        logger.info(f"Validation Loss: {best_model_info['val_loss']:.4f}")
        logger.info(f"{'='*60}\n")

        # Copy best model
        best_model_name = f"best_model_{best_model_info['name']}.pkl"
        best_model_path = Path(best_model_dir) / best_model_name
        shutil.copy2(best_model_info["temp_path"], best_model_path)
        logger.info(f"Best model saved to {best_model_path}")

        # Save comparison results
        results_path = Path(best_model_dir) / "model_comparison.json"
        comparison_data = {
            "best_model": best_model_info["name"],
            "best_model_val_acc": best_model_info["val_acc"],
            "best_model_val_loss": best_model_info["val_loss"],
            "all_models": results,
        }
        with open(results_path, "w") as f:
            json.dump(comparison_data, f, indent=2)
        logger.info(f"Comparison results saved to {results_path}")

        best_model_info["result_path"] = str(results_path)
    else:
        logger.error("No models trained successfully!")

    # Cleanup temporary model files
    for model_name in model_names:
        temp_path = os.path.join(best_model_dir, f"temp_{model_name}.pkl")
        if Path(temp_path).exists():
            Path(temp_path).unlink()
            logger.info(f"Cleaned up {temp_path}")

    return results, best_model_info


if __name__ == "__main__":
    train_model()
