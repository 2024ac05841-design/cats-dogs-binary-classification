# Kubernetes Deployment Guide - Cats vs Dogs Classification

## Overview

This guide covers deploying the Cats vs Dogs classifier to Rancher Desktop using a dedicated Kubernetes namespace with:
- Persistent storage for data and models
- Training as a CronJob
- MLFlow for model tracking and versioning
- Inference API service
- Ingress for web access

## Prerequisites

1. **Rancher Desktop installed** with Kubernetes enabled
2. **kubectl configured** to access your local cluster
3. **Docker image built**: `docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .`
4. **PersistentVolume directories created** on host machine

## Step 1: Create Required Host Directories

```bash
# Create data directories on your local machine
mkdir -p /mnt/data/cats-dogs/data/processed/train
mkdir -p /mnt/data/cats-dogs/data/processed/val
mkdir -p /mnt/data/cats-dogs/data/processed/test
mkdir -p /mnt/data/cats-dogs/models/best_model
mkdir -p /mnt/data/cats-dogs/mlflow

# On Windows with Rancher Desktop
# Use WSL2 to create these directories
# Or manually create C:\mnt\data\cats-dogs\... structure
```

## Step 2: Prepare Your Dataset

```bash
# Copy your prepared dataset to the PV location
# Train data
cp -r data/processed/train/* /mnt/data/cats-dogs/data/processed/train/

# Validation data
cp -r data/processed/val/* /mnt/data/cats-dogs/data/processed/val/

# Test data
cp -r data/processed/test/* /mnt/data/cats-dogs/data/processed/test/
```

## Step 3: Deploy to Kubernetes

### Option A: Deploy All at Once (Recommended)

```bash
# From project root
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-persistent-volumes.yaml
kubectl apply -f k8s/02-training-configmap.yaml
kubectl apply -f k8s/03-training-cronjob.yaml
kubectl apply -f k8s/04-mlflow-deployment.yaml
kubectl apply -f k8s/05-inference-deployment.yaml
kubectl apply -f k8s/06-ingress.yaml
```

### Option B: Deploy Step by Step

```bash
# 1. Create namespace
kubectl apply -f k8s/00-namespace.yaml
kubectl get namespace cats-dogs-classification

# 2. Create persistent volumes
kubectl apply -f k8s/01-persistent-volumes.yaml
kubectl get pv -n cats-dogs-classification
kubectl get pvc -n cats-dogs-classification

# 3. Create training config
kubectl apply -f k8s/02-training-configmap.yaml

# 4. Create training job
kubectl apply -f k8s/03-training-cronjob.yaml
kubectl get cronjob -n cats-dogs-classification

# 5. Deploy MLFlow
kubectl apply -f k8s/04-mlflow-deployment.yaml
kubectl get deployment -n cats-dogs-classification

# 6. Deploy inference service
kubectl apply -f k8s/05-inference-deployment.yaml

# 7. Setup ingress
kubectl apply -f k8s/06-ingress.yaml
kubectl get ingress -n cats-dogs-classification
```

## Step 4: Verify Deployments

```bash
# Check all resources in namespace
kubectl get all -n cats-dogs-classification

# Check pods
kubectl get pods -n cats-dogs-classification

# Check volumes
kubectl get pv
kubectl get pvc -n cats-dogs-classification

# View pod logs
kubectl logs -f deployment/mlflow -n cats-dogs-classification
kubectl logs -f deployment/inference-service -n cats-dogs-classification
```

## Step 5: Running Training

### Manual Training Job Trigger

```bash
# Create a one-time training job from the CronJob template
kubectl create job --from=cronjob/cats-dogs-training manual-training-run \
  -n cats-dogs-classification

# Monitor training job
kubectl get jobs -n cats-dogs-classification
kubectl logs -f job/manual-training-run -n cats-dogs-classification

# Check training results
kubectl exec -it deployment/inference-service -n cats-dogs-classification -- \
  ls -la /models/best_model/
```

### Automatic Scheduled Training

The CronJob runs automatically at **2 AM every Sunday** (configurable in `03-training-cronjob.yaml`).

To modify schedule:
```bash
# Edit the cronjob
kubectl edit cronjob cats-dogs-training -n cats-dogs-classification

# Change the schedule field (cron format):
# minute hour day month day-of-week
# Examples:
# "0 2 * * 0"   = 2 AM every Sunday
# "0 */6 * * *" = Every 6 hours
# "0 0 * * *"   = Daily at midnight
```

## Step 6: Access MLFlow Web UI

### Local Machine Access

1. **Update your hosts file** to enable DNS resolution:

**Windows**: Edit `C:\Windows\System32\drivers\etc\hosts`
```
127.0.0.1 mlflow.local
127.0.0.1 inference.local
```

**Linux/Mac**: Edit `/etc/hosts`
```
127.0.0.1 mlflow.local
127.0.0.1 inference.local
```

2. **Port Forward** (if using Ingress doesn't work):
```bash
# MLFlow UI
kubectl port-forward -n cats-dogs-classification svc/mlflow 5000:5000
# Access: http://localhost:5000

# Inference API
kubectl port-forward -n cats-dogs-classification svc/inference-service 8000:8000
# Access: http://localhost:8000/docs
```

3. **Access Services**:
- **MLFlow UI**: http://mlflow.local (or http://localhost:5000)
- **Inference API Docs**: http://inference.local/docs (or http://localhost:8000/docs)
- **Inference Health**: http://inference.local/health

## Step 7: Monitor Training & Models

### Check Training Progress

```bash
# Watch training job
kubectl logs -f job/manual-training-run -n cats-dogs-classification

# Check if models were saved
kubectl get pvc -n cats-dogs-classification
kubectl describe pvc training-models-pvc -n cats-dogs-classification
```

### Access MLFlow Dashboard

Once training completes, MLFlow shows:
1. **Experiments**: cats-dogs-k8s (all runs)
2. **Runs**: Individual model training runs
3. **Best Model**: Tagged as "best_model" with highest validation accuracy
4. **Metrics**: 
   - val_accuracy
   - val_loss
   - train_accuracy
   - train_loss

### Query Models via MLFlow API

```bash
# Get best model info
curl http://localhost:5000/api/2.0/mlflow/registered-models/search

# Get experiment details
curl http://localhost:5000/api/2.0/mlflow/experiments/search

# Get runs in experiment
curl http://localhost:5000/api/2.0/mlflow/runs/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["1"]}'
```

## Step 8: Scaling & Monitoring

### Scale Inference Service

```bash
# Increase replicas
kubectl scale deployment inference-service --replicas=4 \
  -n cats-dogs-classification

# View replicas
kubectl get deployment inference-service -n cats-dogs-classification
```

### Monitor Resource Usage

```bash
# View resource consumption
kubectl top nodes
kubectl top pod -n cats-dogs-classification

# Describe resources
kubectl describe nodes
```

### Check Pod Events

```bash
# Watch pod events
kubectl get events -n cats-dogs-classification --sort-by='.lastTimestamp'

# Detailed pod info
kubectl describe pod <pod-name> -n cats-dogs-classification
```

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────┐
│          Data Preparation                           │
│    (Local: data/processed/)                         │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│     PersistentVolume: training-data-pv              │
│    /mnt/data/cats-dogs/data/processed               │
│    ├── train/                                       │
│    ├── val/                                         │
│    └── test/                                        │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│          Training CronJob                           │
│   - Runs training with 3 models                     │
│   - Selects best model                              │
│   - Saves to models PV                              │
└────────────┬────────────────────────────────────────┘
             │
             ▼ (logs models)
┌─────────────────────────────────────────────────────┐
│     MLFlow Server (Experiment Tracking)             │
│   - Tracks all 3 model metrics                      │
│   - Versions best model                             │
│   - Stores artifacts in PV                          │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│     PersistentVolume: training-models-pv            │
│    /mnt/data/cats-dogs/models/                      │
│    ├── best_model/                                  │
│    │   ├── best_model_resnet18.pkl                  │
│    │   └── model_comparison.json                    │
│    └── (temporary models cleaned up)                │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│      Inference Service (2+ replicas)                │
│   - Copies best model on startup                    │
│   - Serves predictions via REST API                 │
│   - Health checks every 10 seconds                  │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│     Ingress (Web Access)                            │
│   - MLFlow: mlflow.local:5000                       │
│   - Inference: inference.local:8000                 │
│   - Assignment details visible in browser           │
└─────────────────────────────────────────────────────┘
```

## Troubleshooting

### Issue: Pods stuck in Pending state

```bash
# Check events
kubectl describe pod <pod-name> -n cats-dogs-classification

# Solution: Ensure PersistentVolumes exist and are mounted
kubectl get pv
# Verify /mnt/data/cats-dogs/ directories exist on host
```

### Issue: Training job fails

```bash
# Check logs
kubectl logs -f job/manual-training-run -n cats-dogs-classification

# Common causes:
# - Data not in /data/processed/train, val, test
# - Insufficient memory (increase limits in cronjob)
# - Model import errors
```

### Issue: MLFlow not accessible

```bash
# Verify deployment
kubectl get deployment mlflow -n cats-dogs-classification
kubectl logs deployment/mlflow -n cats-dogs-classification

# Port forward if ingress not working
kubectl port-forward svc/mlflow 5000:5000 -n cats-dogs-classification
```

### Issue: Inference pod can't find model

```bash
# Check if model was saved
kubectl exec -it deployment/inference-service -n cats-dogs-classification -- \
  ls -la /models/best_model/

# If empty, run training job first
kubectl create job --from=cronjob/cats-dogs-training training-init \
  -n cats-dogs-classification
```

## Cleanup

```bash
# Delete everything in namespace
kubectl delete namespace cats-dogs-classification

# Delete namespace only (keep resources)
kubectl delete -f k8s/00-namespace.yaml

# Delete specific resource
kubectl delete deployment inference-service -n cats-dogs-classification
```

## Assignment Details in Browser

Once deployed, view assignment details:

1. **MLFlow Dashboard** (http://mlflow.local):
   - Experiment: "cats-dogs-k8s"
   - Shows all training runs
   - Best model highlighted
   - Metrics comparison

2. **API Documentation** (http://inference.local/docs):
   - Swagger UI showing all endpoints
   - `/health` - Service health
   - `/predict` - Image prediction endpoint
   - `/info` - Service information

3. **Model Comparison**:
   - Access MLFlow artifacts
   - View `model_comparison.json`
   - All 3 models' metrics

## Next Steps

1. Copy prepared dataset to `/mnt/data/cats-dogs/data/processed/`
2. Deploy all manifests: `kubectl apply -f k8s/`
3. Trigger training: `kubectl create job ... --from=cronjob/...`
4. Monitor MLFlow at http://mlflow.local
5. Test inference at http://inference.local/docs
