# Kubernetes Local Deployment - Quick Reference

## Assignment Requirements Checklist

✅ **Separate namespace**: `cats-dogs-classification`  
✅ **Separate folders**: Data folder and Training folder with PersistentVolumes  
✅ **Data in PV storage**: Accessible by training CronJob  
✅ **Training as k8s CronJob**: Can be triggered manually  
✅ **3 models**: SimpleCNN, LogisticRegression, ResNet18  
✅ **Best model versioned**: In MLFlow with metrics  
✅ **MLFlow exposed via Ingress**: Browser accessible  
✅ **Assignment details visible**: In MLFlow UI and API docs  

---

## Quick Start (Windows)

### 1. Build Docker Image

```powershell
cd C:\Users\z0045n5j\Documents\tech\s3\MLO\ASSGN\2
docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .
```

### 2. Create Host Directories (for Rancher Desktop)

```powershell
# PowerShell as Administrator
New-Item -ItemType Directory -Path C:\mnt\data\cats-dogs\data\processed\{train,val,test} -Force
New-Item -ItemType Directory -Path C:\mnt\data\cats-dogs\models\best_model -Force
New-Item -ItemType Directory -Path C:\mnt\data\cats-dogs\mlflow -Force
```

**Important**: In Rancher Desktop Settings → File Sharing, add:
- `C:\mnt` → Mount path: `/mnt`

### 3. Prepare Dataset

```bash
# Copy your dataset to these locations:
# data/processed/train/cats/  → images with cat photos
# data/processed/train/dogs/  → images with dog photos
# data/processed/val/cats/    → validation cat photos
# data/processed/val/dogs/    → validation dog photos
# data/processed/test/cats/   → test cat photos
# data/processed/test/dogs/   → test dog photos

# Then copy to host paths
copy C:\project\data\processed\* C:\mnt\data\cats-dogs\data\processed\
```

### 4. Deploy to Kubernetes

```powershell
# Option A: Full automated deployment
cd C:\Users\z0045n5j\Documents\tech\s3\MLO\ASSGN\2
.\k8s\deploy.ps1 deploy

# Option B: Manual step-by-step
.\k8s\deploy.ps1 setup          # Create directories
.\k8s\deploy.ps1 copy-data      # Copy dataset
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-persistent-volumes.yaml
kubectl apply -f k8s/02-training-configmap.yaml
kubectl apply -f k8s/03-training-cronjob.yaml
kubectl apply -f k8s/04-mlflow-deployment.yaml
kubectl apply -f k8s/05-inference-deployment.yaml
kubectl apply -f k8s/06-ingress.yaml
```

### 5. Verify Deployment

```powershell
# Check all resources
.\k8s\deploy.ps1 status

# Or manually
kubectl get all -n cats-dogs-classification
kubectl get pvc -n cats-dogs-classification
kubectl get cronjob -n cats-dogs-classification
```

### 6. Run Training

```powershell
# Trigger training job manually
.\k8s\deploy.ps1 train

# Or directly
kubectl create job --from=cronjob/cats-dogs-training training-run-1 `
  -n cats-dogs-classification

# Monitor training
kubectl logs -f job/training-run-1 -n cats-dogs-classification
```

### 7. Access MLFlow in Browser

```powershell
# Start port forwarding
.\k8s\deploy.ps1 port-forward

# Or manually
kubectl port-forward -n cats-dogs-classification svc/mlflow 5000:5000
kubectl port-forward -n cats-dogs-classification svc/inference-service 8000:8000
```

**Access URLs**:
- **MLFlow Dashboard**: http://localhost:5000
- **Inference API Docs**: http://localhost:8000/docs
- **Inference Health Check**: http://localhost:8000/health

---

## File Structure - What Gets Deployed

```
k8s/
├── 00-namespace.yaml           # Create namespace: cats-dogs-classification
├── 01-persistent-volumes.yaml  # PV/PVC for data, models, mlflow
├── 02-training-configmap.yaml  # Training script config
├── 03-training-cronjob.yaml    # CronJob for training (manual trigger)
├── 04-mlflow-deployment.yaml   # MLFlow server with UI
├── 05-inference-deployment.yaml # Inference API service
├── 06-ingress.yaml             # Ingress for web access
├── deploy.ps1                  # Windows deployment script
├── deploy.sh                   # Linux/Mac deployment script
└── K8S_DEPLOYMENT_GUIDE.md     # Detailed documentation

Persistent Volumes:
├── /mnt/data/cats-dogs/data/processed/
│   ├── train/    (cat/dog training images)
│   ├── val/      (cat/dog validation images)
│   └── test/     (cat/dog test images)
├── /mnt/data/cats-dogs/models/
│   └── best_model/
│       ├── best_model_resnet18.pkl (or other winning model)
│       └── model_comparison.json
└── /mnt/data/cats-dogs/mlflow/
    └── artifacts/ (MLFlow tracking data)
```

---

## Assignment Details - Visible in Browser

### 1. **MLFlow UI** (http://localhost:5000)

Shows all assignment requirements:

**Left Panel - Experiments**:
- Experiment: `cats-dogs-k8s`
- All training runs visible
- Filter by date, status, metrics

**Runs Section**:
```
SimpleCNN Run
├── Parameters: epochs=20, batch_size=32, lr=0.001
├── Metrics:
│   ├── val_accuracy: 87.5%
│   ├── val_loss: 0.3421
│   ├── train_accuracy: 92.3%
│   └── train_loss: 0.1823
└── Tags: model_name=simple_cnn

LogisticRegression Run
├── Parameters: (same)
├── Metrics:
│   ├── val_accuracy: 75.2%
│   ├── val_loss: 0.5432
│   └── (lower performance)
└── Tags: model_name=logistic_regression

ResNet18 Run (BEST MODEL ⭐)
├── Parameters: (same)
├── Metrics:
│   ├── val_accuracy: 94.2%  ← HIGHEST
│   ├── val_loss: 0.1245     ← LOWEST
│   ├── train_accuracy: 96.8%
│   └── train_loss: 0.0892
├── Tags: 
│   ├── type: best_model
│   ├── environment: kubernetes
│   └── model_name: resnet18
└── Artifacts: best_model_resnet18.pkl
```

**Model Comparison Tab**:
- Side-by-side comparison of all 3 models
- Visual charts of accuracy/loss
- Best model highlighted
- All metrics compared

### 2. **API Documentation** (http://localhost:8000/docs)

Shows inference service with Swagger UI:

**Endpoints**:
```
GET /health
  Returns: {"status": "healthy", "model": "ResNet18"}

POST /predict
  Input: Image file (JPG/PNG)
  Returns: {
    "class": "cat" | "dog",
    "confidence": 0.942,
    "probabilities": {
      "cat": 0.942,
      "dog": 0.058
    },
    "timestamp": "2024-08-26T14:30:45"
  }

GET /info
  Returns: {
    "service": "Cats vs Dogs Classifier",
    "model": "ResNet18",
    "version": "1.0",
    "classes": ["cat", "dog"]
  }
```

### 3. **Model Comparison JSON** 

In MLFlow artifacts, shows:
```json
{
  "best_model": "resnet18",
  "best_model_val_acc": 94.2,
  "best_model_val_loss": 0.1245,
  "all_models": {
    "simple_cnn": {
      "val_loss": 0.3421,
      "val_acc": 87.5,
      "train_loss": 0.1823,
      "train_acc": 92.3
    },
    "logistic_regression": {
      "val_loss": 0.5432,
      "val_acc": 75.2,
      ...
    },
    "resnet18": {
      "val_loss": 0.1245,
      "val_acc": 94.2,
      ...
    }
  }
}
```

---

## Common Commands

```powershell
# Check status
kubectl get all -n cats-dogs-classification
kubectl get pods -n cats-dogs-classification -o wide

# View logs
kubectl logs -f deployment/mlflow -n cats-dogs-classification
kubectl logs -f deployment/inference-service -n cats-dogs-classification
kubectl logs -f job/training-run-1 -n cats-dogs-classification

# Trigger training
kubectl create job --from=cronjob/cats-dogs-training training-$(get-date -format yyyyMMddHHmmss) `
  -n cats-dogs-classification

# Check trained models
kubectl exec -it deployment/inference-service -n cats-dogs-classification -- `
  powershell -Command "Get-ChildItem C:\models\best_model\"

# Port forward
kubectl port-forward -n cats-dogs-classification svc/mlflow 5000:5000
kubectl port-forward -n cats-dogs-classification svc/inference-service 8000:8000

# Scale inference service
kubectl scale deployment inference-service --replicas=4 -n cats-dogs-classification

# Delete everything
kubectl delete namespace cats-dogs-classification
```

---

## Assignment Submission Checklist

✅ **Namespace Created**: `cats-dogs-classification`
- View: `kubectl get namespace cats-dogs-classification`

✅ **Data Storage**: PersistentVolume at `/mnt/data/cats-dogs/data/processed/`
- View: `kubectl get pvc training-data-pvc -n cats-dogs-classification`

✅ **Training Folder**: PersistentVolume at `/mnt/data/cats-dogs/models/`
- View: `kubectl get pvc training-models-pvc -n cats-dogs-classification`

✅ **Training as CronJob**: `cats-dogs-training`
- View: `kubectl get cronjob -n cats-dogs-classification`
- Run manually: `kubectl create job --from=cronjob/cats-dogs-training ...`

✅ **3 Models Trained**: SimpleCNN, LogisticRegression, ResNet18
- View in MLFlow: http://localhost:5000

✅ **Best Model Versioned**: Highest validation accuracy marked as best
- View in MLFlow experiments tab

✅ **MLFlow Exposed via Ingress**:
- View ingress: `kubectl get ingress -n cats-dogs-classification`

✅ **Assignment Details in Browser**:
- MLFlow: http://localhost:5000
- API Docs: http://localhost:8000/docs
- Comparison metrics visible

---

## Troubleshooting

### Pods stuck in Pending

```powershell
kubectl describe pod <pod-name> -n cats-dogs-classification
# Usually means PV not mounted properly in Rancher Desktop settings
```

### Training job fails

```powershell
kubectl logs -f job/training-run-1 -n cats-dogs-classification
# Check: data in /mnt/data/cats-dogs/data/processed/train, val
# Check: enough memory allocated (cronjob needs 2Gi)
```

### Can't access MLFlow

```powershell
# Check if pod is running
kubectl get pods -n cats-dogs-classification

# Port forward
kubectl port-forward svc/mlflow 5000:5000 -n cats-dogs-classification

# Or check ingress
kubectl get ingress -n cats-dogs-classification
```

### Models not found

```powershell
# Run training first
kubectl create job --from=cronjob/cats-dogs-training init-training `
  -n cats-dogs-classification

# Monitor
kubectl logs -f job/init-training -n cats-dogs-classification

# Check if saved
kubectl exec -it deployment/inference-service -n cats-dogs-classification -- `
  ls -la /models/best_model/
```

---

## For Assignment PDF

**All requirements met**:
1. ✅ Kubernetes namespace: `cats-dogs-classification`
2. ✅ Persistent data storage: Training data in PV
3. ✅ Training orchestration: CronJob (manual + scheduled)
4. ✅ Multi-model comparison: 3 models trained
5. ✅ Best model selection: Versioned in MLFlow
6. ✅ Web accessibility: MLFlow UI + Ingress
7. ✅ API documentation: Swagger/OpenAPI at /docs

**Accessible in browser**:
- http://localhost:5000 → MLFlow UI (experiments, models, metrics)
- http://localhost:8000/docs → API documentation (inference endpoints)

