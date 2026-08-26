# MLOps Assignment 2 - Quick Reference Guide

## ✅ IMPLEMENTATION COMPLETE - READY FOR GIT PUSH

This is a complete end-to-end MLOps pipeline with all 5 modules fully implemented.

---

## 🎯 What's Been Done

### Module 1: Model Development & Experiment Tracking ✅
- **Data Preprocessing**: `src/data/preprocessing.py` - Image loading, resizing, validation
- **Data Augmentation**: `src/data/augmentation.py` - Advanced augmentation pipeline
- **Model Architecture**: `src/models/cnn_model.py` - SimpleCNN and ResNet baseline models
- **Training Pipeline**: `src/models/train.py` - Complete training with MLflow integration
- **Dataset Utilities**: Custom CatsDogsDataset class with automatic data loading

### Module 2: Model Packaging & Containerization ✅
- **FastAPI Service**: `src/inference/app.py` - REST API with 3 endpoints
  - `GET /health` - Health check
  - `POST /predict` - Image classification
  - `GET /info` - Service information
- **Model Utilities**: `src/inference/model_utils.py` - Inference functions
- **Docker Setup**: Complete Dockerfile with health checks
- **Dependencies**: Pinned versions in `requirements.txt`

### Module 3: CI Pipeline (Build, Test & Image Creation) ✅
- **Unit Tests**: 
  - `tests/test_preprocessing.py` - 10+ test cases for data preprocessing
  - `tests/test_inference.py` - 10+ test cases for inference
- **GitHub Actions**: `.github/workflows/ci.yml`
  - Checkout → Install deps → Run tests → Build Docker image
  - Pytest with coverage reporting
  - Docker image testing before push

### Module 4: CD Pipeline & Deployment ✅
- **Deployment Options**:
  - Docker Compose: `docker-compose.yml` (primary for local/Rancher Desktop)
  - Kubernetes: `k8s/deployment.yaml` + `k8s/service.yaml`
- **CD Workflow**: `.github/workflows/cd.yml`
  - Auto-deploy on main branch changes
  - Smoke tests validation
  - Service health verification
- **Smoke Tests**: `smoke_tests.sh` - Post-deployment validation

### Module 5: Monitoring, Logs & Final Submission ✅
- **Logging**: `src/monitoring/logging_config.py`
  - JSON formatted logs to `logs/app.log`
  - Request/response logging
  - Error tracking
- **Metrics**: `src/monitoring/metrics.py`
  - Request counting and latency tracking
  - Post-deployment prediction collection
  - Accuracy calculation from collected data
- **Metrics Files**:
  - `logs/metrics.jsonl` - System metrics
  - `logs/predictions.jsonl` - Prediction records

---

## 📁 Project Structure

```
MLOps_Assignment_2/
├── .venv/                                  # Virtual environment (created)
├── .git/                                   # Git repo (initialized)
├── .github/workflows/                      # CI/CD pipelines
│   ├── ci.yml                             # Test + Build pipeline
│   └── cd.yml                             # Deploy pipeline
├── src/                                    # Source code
│   ├── data/                              # Data processing
│   │   ├── preprocessing.py               # Data loading & splitting
│   │   ├── augmentation.py                # Data augmentation
│   │   └── __init__.py
│   ├── models/                            # Model training
│   │   ├── cnn_model.py                   # Model architectures
│   │   ├── train.py                       # Training script
│   │   └── __init__.py
│   ├── inference/                         # API service
│   │   ├── app.py                         # FastAPI application
│   │   ├── model_utils.py                 # Prediction utilities
│   │   └── __init__.py
│   └── monitoring/                        # Monitoring
│       ├── logging_config.py              # Logging setup
│       ├── metrics.py                     # Metrics collection
│       └── __init__.py
├── tests/                                 # Unit tests
│   ├── test_preprocessing.py              # Data tests
│   ├── test_inference.py                  # Inference tests
│   └── __init__.py
├── docker/                                # Docker setup
│   ├── Dockerfile                         # Container image
│   └── .dockerignore                      # Build exclusions
├── k8s/                                   # Kubernetes (optional)
│   ├── deployment.yaml                    # K8s deployment
│   └── service.yaml                       # K8s service
├── scripts/                               # Helper scripts
│   ├── prepare_data.py                    # Dataset preparation
│   ├── train.py                           # Training script
│   └── __init__.py
├── docker-compose.yml                     # Docker Compose deployment
├── dvc.yaml                               # DVC pipeline config
├── params.yaml                            # Training parameters
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git exclusions
├── README.md                              # Full documentation
├── setup.sh                               # Linux/Mac setup
├── setup.bat                              # Windows setup
├── smoke_tests.sh                         # Post-deploy tests
└── Assignment 2.pdf                       # Assignment brief
```

---

## 🚀 How to Proceed

### Next Steps (In Order):

1. **Prepare Dataset**
   ```bash
   # Download from Kaggle: https://www.kaggle.com/datasets/...
   # Organize in: data/raw/cats/ and data/raw/dogs/
   mkdir -p data/raw/cats data/raw/dogs
   ```

2. **Prepare Data** (Optional - can skip if using raw data directly)
   ```bash
   python scripts/prepare_data.py
   # Creates: data/processed/train/, data/processed/val/, data/processed/test/
   ```

3. **Train Model** (Optional - if you want to train before deployment)
   ```bash
   # Activate venv first
   source .venv/bin/activate  # Linux/Mac
   # OR
   .venv\Scripts\activate     # Windows
   
   # Train
   python -m src.models.train --epochs 20 --batch-size 32
   # Saves to: models/model.pkl
   ```

4. **Test Locally with Docker Compose**
   ```bash
   # Start services (requires Rancher Desktop or Docker)
   docker-compose -f docker-compose.yml up -d
   
   # Test service
   curl http://localhost:8000/health
   curl -F "file=@test_image.jpg" http://localhost:8000/predict
   
   # View logs
   docker-compose logs -f inference-service
   
   # Cleanup
   docker-compose down
   ```

5. **Push to Git** (when ready)
   ```bash
   git add .
   git commit -m "Add dataset or models"
   git push origin main
   # GitHub Actions will automatically:
   # - Run tests
   # - Build Docker image
   # - Deploy with Docker Compose
   # - Run smoke tests
   ```

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_preprocessing.py::TestCatsDogsDataset::test_dataset_len -v
```

---

## 🐳 Docker with Rancher Desktop

```bash
# Rancher Desktop is already configured for local development
# No special commands needed - just use regular docker commands

# Build
docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .

# Run
docker run -p 8000:8000 cats-dogs-classifier:latest

# Push to registry (optional)
docker tag cats-dogs-classifier:latest username/cats-dogs-classifier:latest
docker push username/cats-dogs-classifier:latest
```

---

## 📊 Testing the Inference Service

### Method 1: Using curl
```bash
# Health check
curl http://localhost:8000/health

# Info
curl http://localhost:8000/info

# Predict (with image file)
curl -F "file=@/path/to/image.jpg" http://localhost:8000/predict
```

### Method 2: Using Python
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prediction
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

### Method 3: OpenAPI/Swagger UI
```
http://localhost:8000/docs
```
(Interactive API testing in browser)

---

## 📝 File Descriptions

| File/Folder | Purpose |
|---|---|
| `src/data/preprocessing.py` | Dataset loading, splitting, validation |
| `src/data/augmentation.py` | Data augmentation transforms |
| `src/models/cnn_model.py` | CNN and ResNet model definitions |
| `src/models/train.py` | Training loop with MLflow tracking |
| `src/inference/app.py` | FastAPI REST API server |
| `src/inference/model_utils.py` | Model loading and prediction utilities |
| `src/monitoring/logging_config.py` | Logging configuration |
| `src/monitoring/metrics.py` | Metrics collection and tracking |
| `tests/test_preprocessing.py` | Unit tests for data preprocessing |
| `tests/test_inference.py` | Unit tests for inference |
| `.github/workflows/ci.yml` | CI pipeline (test + build) |
| `.github/workflows/cd.yml` | CD pipeline (deploy) |
| `docker/Dockerfile` | Docker image definition |
| `docker-compose.yml` | Local deployment with Docker Compose |
| `k8s/deployment.yaml` | Kubernetes deployment manifest |
| `k8s/service.yaml` | Kubernetes service manifest |
| `scripts/prepare_data.py` | Data preparation and splitting script |
| `scripts/train.py` | Standalone training script |
| `requirements.txt` | Python package dependencies (pinned versions) |
| `params.yaml` | Training hyperparameters |
| `dvc.yaml` | DVC pipeline configuration |
| `README.md` | Full project documentation |
| `smoke_tests.sh` | Post-deployment validation tests |

---

## ✅ Deliverables Ready

- ✅ Complete source code with all modules
- ✅ Configuration files (DVC, CI/CD, Docker, K8s)
- ✅ Unit tests (10+ test cases)
- ✅ Docker containerization
- ✅ GitHub Actions CI/CD
- ✅ Deployment manifests (Docker Compose + Kubernetes)
- ✅ Monitoring and logging
- ✅ Smoke tests
- ✅ Comprehensive README

---

## 🎓 Key Technologies Used

- **ML Framework**: PyTorch
- **API Framework**: FastAPI
- **Experiment Tracking**: MLflow
- **Container**: Docker + Rancher Desktop
- **CI/CD**: GitHub Actions
- **Deployment**: Docker Compose (primary), Kubernetes (optional)
- **Data Management**: DVC
- **Testing**: pytest
- **Logging**: Python logging + JSON formatter
- **Monitoring**: Custom metrics collection

---

## 📞 Support Notes

- **Port 8000**: Inference API service
- **Port 5000**: MLflow UI (optional, if running separately)
- **Virtual Environment**: `.venv/` (created, dependencies installed)
- **Git Status**: Repository initialized, first commit done
- **Ready to Push**: Yes ✅

---

**Status: IMPLEMENTATION COMPLETE ✅**

Everything is set up and ready for you to:
1. Add your dataset
2. Train the model (optional)
3. Test locally with Docker Compose
4. Push to Git to trigger CI/CD pipeline
