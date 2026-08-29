# M1: Model Development & Experiment Tracking 🏗️

**Status:** ✅ Complete
**Focus:** Building multiple models and tracking experiments with DVC and MLflow

---

## 📋 Subtasks Overview

| #   | Subtask                | Description                      | Status      |
| --- | ---------------------- | -------------------------------- | ----------- |
| 1.1 | Data & Code Versioning | Git + DVC configuration          | ✅ Complete |
| 1.2 | Model Building         | Multiple baseline models         | ✅ Complete |
| 1.3 | Experiment Tracking    | MLflow integration               | ✅ Complete |
| 1.4 | Model Artifacts        | Save and retrieve trained models | ✅ Complete |

---

## 🎯 Subtask 1.1: Data & Code Versioning

### Overview

Implement version control for both source code and datasets using Git and DVC.

### Implementation Details

#### Git Repository

```bash
# Initialize Git repository
git init
git remote add origin <your-repo-url>
git add .
git commit -m "initial commit"
git push -u origin Main
```

**Files Tracked by Git:**

- ✅ Source code (`src/`)
- ✅ Configuration files (`dvc.yaml`, `params.yaml`)
- ✅ Dockerfiles and deployment manifests
- ✅ Unit tests (`tests/`)
- ✅ GitHub Actions workflows (`.github/workflows/`)

#### DVC Pipeline Configuration

```yaml
# dvc.yaml - Define data processing and training stages

stages:
  prepare:
    cmd: python src/scripts/prepare_data.py
    deps:
      - data/raw
    outs:
      - data/processed
  
  train:
    cmd: python src/scripts/train.py --epochs 20 --batch-size 32
    deps:
      - data/processed
      - src/models/
      - src/scripts/train.py
    params:
      - train.epochs
      - train.batch_size
      - train.learning_rate
    outs:
      - models/best_model
    metrics:
      - models/best_model/model_comparison.json:
          cache: false
```

**Run DVC Pipeline:**

```bash
# Reproduce the entire pipeline
dvc repro

# View pipeline DAG
dvc dag

# Check pipeline status
dvc status
```

#### Dataset Structure

```
data/
├── raw/
│   ├── cats/
│   │   ├── cat_1.jpg
│   │   ├── cat_2.jpg
│   │   └── ...
│   └── dogs/
│       ├── dog_1.jpg
│       ├── dog_2.jpg
│       └── ...
└── processed/
    ├── train/
    │   ├── cats/
    │   └── dogs/
    ├── val/
    │   ├── cats/
    │   └── dogs/
    └── test/
        ├── cats/
        └── dogs/
```

**Ratios:** 80% train / 10% validation / 10% test

### ✅ Implementation Status

- ✅ Git repository configured on GitHub
- ✅ DVC initialized and pipeline defined
- ✅ Data splits created and versioned
- ✅ `dvc.yaml` with prepare and train stages
- ✅ Pipeline reproducible via `dvc repro`

**Files to Review:**

- [dvc.yaml](../dvc.yaml)
- [params.yaml](../params.yaml)
- [src/scripts/prepare_data.py](../src/scripts/prepare_data.py)

---

## 🎯 Subtask 1.2: Model Building

### Overview

Implement multiple baseline models for binary classification (Cats vs. Dogs).

### Model Architectures

#### 1️⃣ SimpleCNN

A lightweight 3-layer convolutional neural network.

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Conv layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
  
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
```

**Characteristics:**

- Parameters: ~2-3M
- Training time: ~10-15 minutes (GPU)
- Memory: ~1GB
- Use case: Custom lightweight baseline
- Performance: 59.60% validation accuracy

#### 2️⃣ MobileNetV2

Lightweight transfer learning model optimized for mobile and edge deployment.

```python
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load pretrained MobileNetV2
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        # Replace final layer
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
```

**Characteristics:**

- Parameters: ~3.5M
- Training time: ~10-15 minutes (GPU)
- Memory: ~512MB
- Use case: Efficient inference, edge devices
- Performance: 99.20% validation accuracy

#### 3️⃣ ResNet18

Transfer learning with ImageNet pretrained weights. Best performing model.

```python
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load pretrained ResNet18
        self.model = torchvision.models.resnet18(pretrained=True)
        # Replace final layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
```

**Characteristics:**

- Parameters: ~11.7M
- Training time: ~15-20 minutes (GPU)
- Memory: ~2GB
- Use case: High accuracy production model
- Performance: **99.60% validation accuracy** ⭐ (Best performer)

### Model Creation

```python
from src.models.cnn_model import create_model

# Create any model
model = create_model(
    model_name='SimpleCNN',
    num_classes=2,
    device='cuda',
    pretrained=False
)

# List available models
available_models = ['SimpleCNN', 'MobileNetV2', 'ResNet18']
```

### Training Process

```python
from src.models.train import train_multiple_models

# Train all models
results = train_multiple_models(
    data_dir='data/processed',
    batch_size=32,
    epochs=20,
    learning_rate=0.001,
    device='cpu'
)

# Results include:
# - Training history (loss, accuracy)
# - Validation metrics
# - Best model selection
# - Model comparison JSON
```

### ✅ Implementation Status

- ✅ SimpleCNN implemented with BatchNorm and Dropout
- ✅ MobileNetV2 transfer learning configured
- ✅ ResNet18 transfer learning configured
- ✅ Model factory pattern for easy instantiation
- ✅ Multi-model training orchestration
- ✅ GPU training with multi-model comparison

**Files to Review:**

- [src/models/cnn_model.py](../src/models/cnn_model.py) - Model definitions
- [src/models/train.py](../src/models/train.py) - Training loop
- [src/gpu_training/train_gpu.py](../src/gpu_training/train_gpu.py) - GPU training

---

## 🎯 Subtask 1.3: Experiment Tracking with MLflow

### Overview

Track all model training runs, parameters, metrics, and artifacts using MLflow.

### MLflow Setup

```bash
# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000

# Access UI at http://localhost:5000
```

### Logged Information

#### 📊 Parameters

Automatically logged from `params.yaml`:

```yaml
train:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001
  
data:
  image_size: 224
  channels: 3
```

#### 📈 Metrics (Tracked per epoch)

```python
mlflow.log_metric("train_loss", train_loss, step=epoch)
mlflow.log_metric("train_acc", train_acc, step=epoch)
mlflow.log_metric("val_loss", val_loss, step=epoch)
mlflow.log_metric("val_acc", val_acc, step=epoch)
```

#### 📁 Artifacts

- Confusion matrices (plots)
- Loss curves (plots)
- Training history (JSON)
- Best model comparison (JSON)
- Model weights (.pkl, .pt)

### MLflow Integration Code

```python
import mlflow
import mlflow.pytorch

# Start new experiment
mlflow.set_experiment("cats-dogs-classification-gpu")

with mlflow.start_run(run_name="SimpleCNN-gpu"):
    # Log parameters
    mlflow.log_params({
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.001
    })
  
    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()
  
        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, step=epoch)
  
    # Log model
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name="cats-dogs-model"
    )
  
    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("training_history.json")
```

### MLflow UI Features

```
📊 Experiment Dashboard
├── All Runs
│   ├── gpu-simple-cnn (59.60% accuracy)
│   ├── gpu-mobilenetv2 (99.20% accuracy)
│   └── gpu-resnet18-production (99.60% accuracy) ⭐ Best
├── Run Comparison
│   └── Side-by-side metrics comparison
└── Model Registry
    └── cats-dogs-best-model
        └── Version 1 (Production - ResNet18)
```

### ✅ Implementation Status

- ✅ MLflow tracking server configured
- ✅ Runs logged for all models
- ✅ Parameters automatically synced with params.yaml
- ✅ Metrics tracked per epoch
- ✅ Artifacts (models, plots) logged
- ✅ Model registry with versioning

**Files to Review:**

- [src/models/train.py](../src/models/train.py) - MLflow integration
- [src/gpu_training/train_gpu.py](../src/gpu_training/train_gpu.py) - GPU MLflow logging
- [src/gpu_training/push_to_mlflow.py](../src/gpu_training/push_to_mlflow.py) - Model registry

---

## 🎯 Subtask 1.4: Model Artifacts & Storage

### Overview

Properly save, organize, and version trained models.

### Model Storage Structure

```
src/models/
├── best_model.pkl                      # Best overall model (ResNet18 - 44.8 MB)
├── best_model_simple_cnn.pkl           # SimpleCNN baseline (103.3 MB - 59.6% accuracy)
├── best_model_mobilenet_v2.pkl         # MobileNetV2 efficient model (9.1 MB - 99.2% accuracy)
├── best_model_resnet18.pkl             # ResNet18 production best (44.8 MB - 99.6% accuracy)
├── model_comparison.json                # Performance metrics comparison
├── cnn_model.py                         # Model class definitions
├── train.py                             # Training script
├── __init__.py                          # Package initialization
└── __pycache__/                         # Python cache directory
```

### Model Saving & Loading

#### Save Model

```python
import pickle

# Save trained model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save metadata
metadata = {
    'model_type': 'SimpleCNN',
    'accuracy': 0.92,
    'timestamp': datetime.now().isoformat()
}
with open('models/model_comparison.json', 'w') as f:
    json.dump(metadata, f)
```

#### Load Model

```python
import pickle

# Load model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load metadata
with open('models/model_comparison.json', 'r') as f:
    metadata = json.load(f)
```

### Model Comparison JSON

```json
{
  "SimpleCNN": {
    "accuracy": 0.5960,
    "precision": 0.58,
    "recall": 0.61,
    "f1": 0.59,
    "training_time": 720.5,
    "model_size_mb": 12.5,
    "device": "GPU"
  },
  "MobileNetV2": {
    "accuracy": 0.9920,
    "precision": 0.99,
    "recall": 0.99,
    "f1": 0.99,
    "training_time": 840.3,
    "model_size_mb": 14.2,
    "device": "GPU"
  },
  "ResNet18": {
    "accuracy": 0.9960,
    "precision": 0.996,
    "recall": 0.996,
    "f1": 0.996,
    "training_time": 920.2,
    "model_size_mb": 47.8,
    "device": "GPU"
  },
  "best_model": "ResNet18",
  "timestamp": "2026-08-28T10:30:00"
}
```

### GPU Training with Model Versioning

```bash
# Train on GPU with multi-model comparison
python -m src.gpu_training.train_gpu \
    --data-dir data/processed \
    --output-dir src/models \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.001
```

### ✅ Implementation Status

- ✅ Models saved in .pkl format
- ✅ Model comparison JSON generated
- ✅ Training history preserved
- ✅ MLflow artifact storage configured
- ✅ Versioning metadata tracked

**Files to Review:**

- [src/gpu_training/train_gpu.py](../src/gpu_training/train_gpu.py#L300-L350) - Model saving
- [models/best_model/](../models/) - Actual model artifacts

---

## MLflow Dashboard Screenshots

### Experiment Tracking & Comparison

![MLflow Experiments](../images/mlflow-experiments.png)

**Features shown:**

- Multiple experiment runs with different hyperparameters
- Metrics comparison across GPU training runs
- Accuracy metrics: gpu-simple-cnn (59.60%), gpu-mobilenetv2 (99.20%), gpu-resnet18-production (99.60%)
- Validation loss tracking and run filtering

### Model Registry & Versioning

![MLflow Model Versions](../images/mlflow-models-versioning.png)

**Features shown:**

- Registered model: "cats-dogs-best-model"
- Version 1 in Production stage
- Architecture: ResNet18
- Performance: 99.6% val accuracy, 97.2% test accuracy
- Tags: Framework (PyTorch), Device (cuda), Training date
- Model built with GPU training: src/gpu_training/train_gpu.py

---

## Key Technologies

| Technology             | Purpose                     | Configuration                 |
| ---------------------- | --------------------------- | ----------------------------- |
| **Git**          | Source code versioning      | `.git/` + GitHub remote     |
| **DVC**          | Data versioning & pipelines | `dvc.yaml`, `params.yaml` |
| **MLflow**       | Experiment tracking         | `http://mlflow:5000`        |
| **PyTorch**      | Model training              | CUDA 11.8/12.1 (GPU) or CPU   |
| **scikit-learn** | Metrics & baselines         | `0.23.2+`                   |

---

## 🚀 Running M1 End-to-End

```bash
# 1. Prepare data using DVC
dvc repro

# 2. View DVC pipeline
dvc dag

# 3. Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# 4. Train models (CPU)
python src/scripts/train.py --epochs 20 --batch-size 32

# 5. Or train on GPU
python -m src.gpu_training.train_gpu --epochs 20 --batch-size 64

# 6. View MLflow UI
# Open http://localhost:5000 in browser

# 7. Check results
cat models/best_model/model_comparison.json
```

---

## ✨ Summary

M1 provides the **foundation** for the entire MLOps pipeline:

- ✅ **Version Control:** Git + DVC track code and data changes
- ✅ **3 Production Models:** SimpleCNN, MobileNetV2, ResNet18
- ✅ **Experiment Tracking:** MLflow logs all runs and metrics
- ✅ **GPU Training:** Optimized training on CUDA devices
- ✅ **Reproducibility:** Pipeline can be re-executed identically
- ✅ **Model Artifacts:** Best model (ResNet18 - 99.6%) selected and saved

**Next Step:** Move to [M2: Model Packaging &amp; Containerization](./02-m2-packaging.md)
