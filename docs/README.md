# MLOps Assignment 2 - Complete Documentation

## 📚 Overview

This documentation provides a comprehensive guide to all **5 modules** of the MLOps Assignment. Each module is designed to build a complete machine learning operations pipeline for a **Cats vs. Dogs Binary Classification System**.

---

## 📋 Module Guide

### [**M1: Model Development & Experiment Tracking**](./01-m1-model-development.md)

🎯 **Focus:** Building models and tracking experiments
📊 **Key Technologies:** Git, DVC, MLflow, PyTorch
✅ **Status:** Complete

### [**M2: Model Packaging & Containerization**](./02-m2-packaging.md)

🎯 **Focus:** Creating REST API and Docker containers
📦 **Key Technologies:** FastAPI, Docker, Python
✅ **Status:** Complete

### [**M3: CI Pipeline for Build, Test & Image Creation**](./03-m3-ci-pipeline.md)

🎯 **Focus:** Automated testing and Docker image building
🔄 **Key Technologies:** GitHub Actions, pytest, Docker Registry
✅ **Status:** Complete

### [**M4: CD Pipeline & Deployment**](./04-m4-cd-deployment.md)

🎯 **Focus:** Deploying models to production environments
🚀 **Key Technologies:** Docker Compose, Kubernetes, Helm
✅ **Status:** Complete

### [**M5: Monitoring, Logs & Final Submission**](./05-m5-monitoring.md)

🎯 **Focus:** Production monitoring and deliverables
📈 **Key Technologies:** Prometheus, ELK Stack, JSON Logging
✅ **Status:** 95% Complete (needs screen recording)

---

## 🚀 Quick Start

```bash
# 1. Set up environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Prepare data (M1 - Stage 1)
python src/scripts/prepare_data.py

# 3. Train model (M1 - Stage 2)
python src/scripts/train.py

# 4. Start inference service (M2)
python -m src.inference.app

# 5. Deploy with Docker Compose (M4)
docker-compose up

# 6. Run smoke tests (M4)
bash smoke_tests.sh
```

---

## 📊 Project Structure

```
MLOps Assignment 2/
├── src/                              # Source code
│   ├── data/                         # Data preprocessing
│   ├── models/                       # Model definitions & training
│   ├── inference/                    # REST API service
│   ├── monitoring/                   # Logging & metrics
│   ├── gpu_training/                 # GPU-accelerated training
│   ├── scripts/                      # CLI entry points
│   └── utils/                        # Utility functions
├── tests/                            # Unit tests
├── docker/                           # Dockerfile templates
│   ├── Dockerfile.inference
│   └── Dockerfile.training
├── k8s/                              # Kubernetes manifests
├── docs/                             # This documentation
├── .github/workflows/                # GitHub Actions CI/CD
├── dvc.yaml                          # DVC pipeline
├── docker-compose.yml                # Local deployment
└── params.yaml                       # Configuration parameters
```

---

## 🎯 Learning Outcomes

By completing this assignment, you will understand:

- ✅ **Data Versioning:** DVC for managing datasets
- ✅ **Experiment Tracking:** MLflow for reproducible ML workflows
- ✅ **Model Containerization:** Docker for consistent deployments
- ✅ **CI/CD Automation:** GitHub Actions for automated testing & deployment
- ✅ **Cloud Deployment:** Kubernetes for scalable production systems
- ✅ **Production Monitoring:** Metrics collection and performance tracking
- ✅ **GitOps Workflow:** Infrastructure as Code principles

## 🔗 Navigation

- [M1: Model Development](./01-m1-model-development.md) - Data versioning, multiple models, experiment tracking
- [M2: Packaging](./02-m2-packaging.md) - REST API, Docker containers
- [M3: CI Pipeline](./03-m3-ci-pipeline.md) - Automated tests, image building
- [M4: CD &amp; Deployment](./04-m4-cd-deployment.md) - Kubernetes, Docker Compose
- [M5: Monitoring](./05-m5-monitoring.md) - Logging, metrics, final submission

---

## Support & Resources

- 📖 [DVC Documentation](https://dvc.org/doc)
- 📖 [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- 📖 [FastAPI Documentation](https://fastapi.tiangolo.com/)
- 📖 [GitHub Actions Documentation](https://docs.github.com/en/actions)
- 📖 [Kubernetes Documentation](https://kubernetes.io/docs/)


---

**Last Updated:** August 27, 2026
