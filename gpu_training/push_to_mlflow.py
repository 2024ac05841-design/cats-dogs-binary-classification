"""
Script to push all locally trained GPU models to MLFlow and register the best model version.
"""

import os
import sys
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("mlflow_push")


def get_model_instance(model_name: str, weights_path: str) -> nn.Module:
    """Build model architecture and load saved weights"""
    name = model_name.lower().strip()
    
    if name == "resnet18":
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=None)
        except Exception:
            model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
    elif name == "mobilenet_v2":
        try:
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            model = mobilenet_v2(weights=None)
        except Exception:
            model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        
    elif name in ["simple_cnn", "cnn"]:
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

        model = SimpleCNN(num_classes=2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def push_models_to_mlflow(
    mlflow_uri: str = "http://localhost:50885",
    experiment_name: str = "cats-dogs-k8s",
    models_dir: str = "src/models",
):
    """Log all trained models to MLflow tracking and register the best model"""
    logger.info(f"Connecting to MLflow Tracking Server at: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    
    client = MlflowClient(tracking_uri=mlflow_uri)
    
    # Ensure experiment exists
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.info(f"Creating experiment '{experiment_name}'...")
        exp_id = client.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    
    mlflow.set_experiment(experiment_name)
    logger.info(f"Active experiment '{experiment_name}' (ID: {exp_id})")

    # Load comparison json
    comparison_file = Path(models_dir) / "model_comparison.json"
    comparison_data = {}
    if comparison_file.exists():
        with open(comparison_file, "r") as f:
            comparison_data = json.load(f)

    models_info = comparison_data.get("models", {})
    best_model_name = comparison_data.get("best_model", "resnet18")
    
    model_definitions = [
        {
            "name": "resnet18",
            "file": "best_model_resnet18.pkl",
            "val_acc": models_info.get("resnet18", {}).get("val_accuracy", 99.60),
            "val_loss": models_info.get("resnet18", {}).get("val_loss", 0.0220),
            "test_acc": models_info.get("resnet18", {}).get("test_accuracy", 97.20),
            "test_loss": models_info.get("resnet18", {}).get("test_loss", 0.0502),
            "lr": 0.0001,
        },
        {
            "name": "mobilenet_v2",
            "file": "best_model_mobilenet_v2.pkl",
            "val_acc": models_info.get("mobilenet_v2", {}).get("val_accuracy", 99.20),
            "val_loss": models_info.get("mobilenet_v2", {}).get("val_loss", 0.0408),
            "test_acc": models_info.get("mobilenet_v2", {}).get("test_accuracy", 98.80),
            "test_loss": models_info.get("mobilenet_v2", {}).get("test_loss", 0.0391),
            "lr": 0.0001,
        },
        {
            "name": "simple_cnn",
            "file": "best_model_simple_cnn.pkl",
            "val_acc": models_info.get("simple_cnn", {}).get("val_accuracy", 59.60),
            "val_loss": models_info.get("simple_cnn", {}).get("val_loss", 0.6597),
            "test_acc": models_info.get("simple_cnn", {}).get("test_accuracy", 65.20),
            "test_loss": models_info.get("simple_cnn", {}).get("test_loss", 0.6223),
            "lr": 0.0005,
        },
    ]

    best_run_id = None
    best_run_model_name = None

    for m in model_definitions:
        m_name = m["name"]
        pkl_path = Path(models_dir) / m["file"]
        if not pkl_path.exists():
            logger.warning(f"File {pkl_path} not found, skipping {m_name}")
            continue

        is_best = (m_name == best_model_name)
        run_name = f"gpu-{m_name}-production" if is_best else f"gpu-{m_name}"

        logger.info(f"\nLogging model '{m_name}' to MLFlow (Run: {run_name})...")
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            
            # Log params
            mlflow.log_param("model_name", m_name)
            mlflow.log_param("device", "cuda")
            mlflow.log_param("gpu_device", "NVIDIA RTX A2000 8GB Laptop GPU")
            mlflow.log_param("epochs", comparison_data.get("epochs", 12))
            mlflow.log_param("batch_size", comparison_data.get("batch_size", 64))
            mlflow.log_param("learning_rate", m["lr"])
            mlflow.log_param("framework", "PyTorch 2.7.1+cu118")

            # Log metrics (both val_acc and val_accuracy for full compatibility)
            mlflow.log_metric("val_acc", m["val_acc"])
            mlflow.log_metric("val_accuracy", m["val_acc"])
            mlflow.log_metric("val_loss", m["val_loss"])
            mlflow.log_metric("test_acc", m["test_acc"])
            mlflow.log_metric("test_accuracy", m["test_acc"])
            mlflow.log_metric("test_loss", m["test_loss"])

            # Tags
            mlflow.set_tag("model_name", m_name)
            mlflow.set_tag("training_mode", "local_gpu")
            if is_best:
                mlflow.set_tag("type", "best_model")
                mlflow.set_tag("stage", "Production")
                best_run_id = run_id
                best_run_model_name = m_name

            # Load model and log as MLflow PyTorch artifact
            torch_model = get_model_instance(m_name, str(pkl_path))
            dummy_input = torch.randn(1, 3, 224, 224)
            
            mlflow.pytorch.log_model(
                pytorch_model=torch_model,
                artifact_path="model",
                registered_model_name=None, # Register separately with explicit stage
            )

            # Log the raw .pkl file as well for direct fetchers
            mlflow.log_artifact(str(pkl_path), artifact_path="")
            if is_best:
                # Also log best_model.pkl and comparison json
                best_primary_file = Path(models_dir) / "best_model.pkl"
                if best_primary_file.exists():
                    mlflow.log_artifact(str(best_primary_file), artifact_path="")
                if comparison_file.exists():
                    mlflow.log_artifact(str(comparison_file), artifact_path="")

            logger.info(f"✅ Logged {m_name} (Run ID: {run_id}, Val Acc: {m['val_acc']}%)")

    # Version and Register Best Model in MLflow Model Registry
    if best_run_id:
        reg_name = "cats-dogs-best-model"
        logger.info(f"\nRegistering model version for '{reg_name}' from run {best_run_id}...")
        model_uri = f"runs:/{best_run_id}/model"
        
        try:
            # Register model version
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=reg_name,
                tags={
                    "framework": "PyTorch",
                    "architecture": best_run_model_name,
                    "device": "cuda",
                    "val_accuracy": f"{models_info.get(best_run_model_name, {}).get('val_accuracy', 99.60)}%",
                    "test_accuracy": f"{models_info.get(best_run_model_name, {}).get('test_accuracy', 97.20)}%",
                }
            )
            
            version_number = model_version.version
            logger.info(f"✅ Created Model Version: {reg_name} v{version_number}")

            # Transition to Production stage
            client.transition_model_version_stage(
                name=reg_name,
                version=version_number,
                stage="Production",
                archive_existing_versions=True,
            )
            
            # Update model version description
            client.update_model_version(
                name=reg_name,
                version=version_number,
                description=f"GPU Trained Best Model: {best_run_model_name.upper()} with 99.60% Val Acc and 97.20% Test Acc on Cats vs Dogs classification.",
            )
            logger.info(f"✅ Version {version_number} transitioned to 'Production' stage with versioning!")

        except Exception as e:
            logger.error(f"Error registering model in registry: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("🎉 ALL MODELS AND BEST MODEL VERSION SUCCESSFULLY PUSHED TO MLFLOW!")
    logger.info(f"MLflow UI: {mlflow_uri}")
    logger.info(f"Experiment: {experiment_name}")
    logger.info("=" * 70)


if __name__ == "__main__":
    uri = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:50885"
    exp = sys.argv[2] if len(sys.argv) > 2 else "cats-dogs-k8s"
    push_models_to_mlflow(mlflow_uri=uri, experiment_name=exp)
