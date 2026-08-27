"""
Multi-Model GPU Training Pipeline for Cats vs Dogs Classification

Trains models locally with GPU acceleration, logs metrics, evaluates performance,
and saves the highest accuracy model to src/models/
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# Import local dataset utilities
try:
    from dataset import build_dataloaders
except ImportError:
    from gpu_training.dataset import build_dataloaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gpu_training")


def setup_mlflow(tracking_uri: Optional[str] = None, experiment_name: str = "cats-dogs-gpu") -> bool:
    """Initialize MLFlow logging if available"""
    try:
        import mlflow
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(uri)
        try:
            mlflow.set_experiment(experiment_name)
        except Exception:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        logger.info(f"MLFlow connected at {uri} (experiment: '{experiment_name}')")
        return True
    except Exception as e:
        logger.warning(f"MLFlow logging not active ({e}). Training will proceed locally.")
        return False


def build_model(model_name: str, num_classes: int = 2) -> nn.Module:
    """Construct model architecture with weights"""
    name = model_name.lower().strip()

    if name in ["simple_cnn", "cnn"]:
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes: int = 2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 28 * 28, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(128, num_classes),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.features(x)
                return self.classifier(x)

        return SimpleCNN(num_classes=num_classes)

    elif name == "resnet18":
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "resnet50":
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception:
            model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif name == "mobilenet_v2":
        try:
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except Exception:
            model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    elif name == "logistic_regression":
        class LogisticRegressionModel(nn.Module):
            def __init__(self, num_classes: int = 2, input_size: int = 150528):
                super().__init__()
                self.linear = nn.Linear(input_size, num_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.view(x.size(0), -1)
                return self.linear(x)

        return LogisticRegressionModel(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: simple_cnn, resnet18, resnet50, mobilenet_v2, logistic_regression")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one single epoch with mixed precision support"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = (device.type == "cuda" and scaler is not None)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation or test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    eval_loss = running_loss / max(total, 1)
    eval_acc = (correct / max(total, 1)) * 100.0
    return eval_loss, eval_acc


def train_single_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 12,
    lr: float = 1e-4,
    use_mlflow: bool = False,
) -> Dict:
    """Train, validate, and test a single model architecture"""
    logger.info(f"\n{'='*70}\nStarting Training: {model_name.upper()} on {device}\n{'='*70}")

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()

    # Learning rate adjustments: pretrained vs custom CNN
    actual_lr = lr if "resnet" in model_name or "mobile" in model_name else lr * 5.0
    optimizer = optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.time()

    mlflow_run = None
    if use_mlflow:
        try:
            import mlflow
            mlflow_run = mlflow.start_run(run_name=f"gpu-{model_name}-{int(time.time())}")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("device", str(device))
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", train_loader.batch_size)
            mlflow.log_param("learning_rate", actual_lr)
        except Exception as e:
            logger.debug(f"MLflow start_run error: {e}")

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        ep_time = time.time() - ep_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% "
            f"{'★ BEST' if is_best else ''}"
        )

        if use_mlflow and mlflow_run:
            try:
                import mlflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
            except Exception:
                pass

    total_training_time = time.time() - start_time

    # Load best weights for final testing
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = (0.0, 0.0)
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        logger.info(f"Test Set Performance for {model_name}: Acc = {test_acc:.2f}%, Loss = {test_loss:.4f}")

    if use_mlflow and mlflow_run:
        try:
            import mlflow
            mlflow.log_metric("best_val_acc", best_val_acc)
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("test_acc", test_acc)
            mlflow.end_run()
        except Exception:
            pass

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "training_time_sec": total_training_time,
        "state_dict": best_state_dict,
        "history": history,
    }


def run_gpu_training(
    data_dir: str = "data/processed",
    output_dir: str = "src/models",
    model_names: Optional[List[str]] = None,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-4,
    preload: bool = False,
    mlflow_uri: Optional[str] = None,
):
    """
    Main orchestration function for multi-model training and saving best artifacts
    """
    if model_names is None:
        model_names = ["resnet18", "simple_cnn", "mobilenet_v2"]

    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    target_models_dir = Path(output_dir).resolve()
    target_models_dir.mkdir(parents=True, exist_ok=True)

    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"🚀 Using GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        torch.backends.cudnn.benchmark = True
    else:
        logger.warning("⚠️ CUDA is not available in current PyTorch build. Training will run on CPU.")

    use_mlflow = setup_mlflow(mlflow_uri)

    # Build DataLoaders from cached processed data
    logger.info(f"Loading datasets from {data_path}...")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=str(data_path),
        batch_size=batch_size,
        num_workers=0,  # Safe for Windows multiprocessing
        preload=preload,
    )

    all_results = {}
    best_overall = {
        "model_name": None,
        "val_acc": -1.0,
        "test_acc": -1.0,
        "val_loss": float("inf"),
    }
    best_overall_state = None

    for model_name in model_names:
        try:
            res = train_single_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=epochs,
                lr=lr,
                use_mlflow=use_mlflow,
            )

            # Save individual model checkpoint into src/models/
            model_save_path = target_models_dir / f"best_model_{model_name}.pkl"
            torch.save(res["state_dict"], model_save_path)
            logger.info(f"Saved {model_name} weights to {model_save_path}")

            all_results[model_name] = {
                "val_accuracy": round(res["best_val_acc"], 2),
                "val_loss": round(res["best_val_loss"], 4),
                "test_accuracy": round(res["test_acc"], 2),
                "test_loss": round(res["test_loss"], 4),
                "best_epoch": res["best_epoch"],
                "training_time_sec": round(res["training_time_sec"], 2),
                "model_path": str(model_save_path),
            }

            if res["best_val_acc"] > best_overall["val_acc"]:
                best_overall["model_name"] = model_name
                best_overall["val_acc"] = res["best_val_acc"]
                best_overall["test_acc"] = res["test_acc"]
                best_overall["val_loss"] = res["best_val_loss"]
                best_overall_state = res["state_dict"]

        except Exception as e:
            logger.error(f"Failed training {model_name}: {e}", exc_info=True)
            all_results[model_name] = {"error": str(e)}

    # Save best overall model directly to src/models/best_model.pkl
    if best_overall["model_name"] and best_overall_state is not None:
        best_primary_path = target_models_dir / "best_model.pkl"
        torch.save(best_overall_state, best_primary_path)
        logger.info(f"\n{'='*70}\n🏆 OVERALL BEST MODEL: {best_overall['model_name'].upper()}")
        logger.info(f"   Validation Accuracy : {best_overall['val_acc']:.2f}%")
        logger.info(f"   Test Accuracy       : {best_overall['test_acc']:.2f}%")
        logger.info(f"   Saved to            : {best_primary_path}\n{'='*70}")

    # Save comparison summary json to src/models/model_comparison.json
    comparison_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "best_model": best_overall["model_name"],
        "best_val_accuracy": round(best_overall["val_acc"], 2),
        "best_test_accuracy": round(best_overall["test_acc"], 2),
        "batch_size": batch_size,
        "epochs": epochs,
        "models": all_results,
    }

    summary_file = target_models_dir / "model_comparison.json"
    with open(summary_file, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    logger.info(f"Model comparison summary saved to {summary_file}")

    return comparison_summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Model GPU Training for Cats vs Dogs")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed dataset")
    parser.add_argument("--output-dir", type=str, default="src/models", help="Destination folder for best models")
    parser.add_argument("--models", nargs="+", default=["resnet18", "simple_cnn", "mobilenet_v2"], help="Model architectures to train")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs per model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--preload", action="store_true", help="Preload all images to memory for maximum throughput")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLFlow tracking URI")

    args = parser.parse_args()

    run_gpu_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_names=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        preload=args.preload,
        mlflow_uri=args.mlflow_uri,
    )


if __name__ == "__main__":
    main()
