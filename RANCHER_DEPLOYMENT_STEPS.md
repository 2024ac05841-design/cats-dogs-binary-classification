# Rancher Desktop Deployment & Training Execution Guide

## Prerequisites Setup

### 1. Verify Rancher Desktop is Running
```powershell
# Check kubectl connection
kubectl cluster-info
kubectl get nodes

# Should show your local node
```

### 2. Build Docker Image Locally
```powershell
cd C:\Users\z0045n5j\Documents\tech\s3\MLO\ASSGN\2

# Build the image
docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .

# Verify image exists
docker images | grep cats-dogs-classifier
```

### 3. Prepare Dataset Directories (One-time)
```powershell
# Create host directories for Rancher Desktop
$dirs = @(
    "C:\mnt\data\cats-dogs\data\processed\train",
    "C:\mnt\data\cats-dogs\data\processed\val",
    "C:\mnt\data\cats-dogs\data\processed\test",
    "C:\mnt\data\cats-dogs\models\best_model",
    "C:\mnt\data\cats-dogs\mlflow"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

Write-Host "Directories created at C:\mnt\data\cats-dogs\"
```

### 4. Configure Rancher Desktop Volume Mounts
- Open **Rancher Desktop** → **Settings** → **File Sharing**
- Add: `C:\mnt` → Mount path: `/mnt`
- Apply changes

### 5. Copy Your Prepared Dataset
```powershell
# If you have prepared data, copy to PV locations
# Expected structure:
# C:\mnt\data\cats-dogs\data\processed\
#   ├── train\
#   │   ├── cats\  (cat images)
#   │   └── dogs\  (dog images)
#   ├── val\
#   │   ├── cats\
#   │   └── dogs\
#   └── test\
#       ├── cats\
#       └── dogs\

# Example copy command (adjust paths to your data)
# Copy-Item "C:\your\prepared\data\*" -Destination "C:\mnt\data\cats-dogs\data\processed\" -Recurse -Force
```

---

## DEPLOYMENT TO KUBERNETES

### Step 1: Deploy Namespace and Persistent Volumes
```powershell
cd C:\Users\z0045n5j\Documents\tech\s3\MLO\ASSGN\2

# Create namespace
kubectl apply -f k8s/00-namespace.yaml

# Verify namespace created
kubectl get namespace cats-dogs-classification

# Create persistent volumes
kubectl apply -f k8s/01-persistent-volumes.yaml

# Wait for PVCs to bind
Start-Sleep -Seconds 5
kubectl get pv
kubectl get pvc -n cats-dogs-classification

# Verify all PVCs are BOUND
```

### Step 2: Deploy MLFlow Server
```powershell
# Create training configuration
kubectl apply -f k8s/02-training-configmap.yaml

# Deploy MLFlow
kubectl apply -f k8s/04-mlflow-deployment.yaml

# Wait for MLFlow pod to be ready
kubectl wait --for=condition=ready pod -l app=mlflow -n cats-dogs-classification --timeout=300s

# Verify MLFlow is running
kubectl get deployment mlflow -n cats-dogs-classification
kubectl get pods -n cats-dogs-classification

# Check logs
kubectl logs deployment/mlflow -n cats-dogs-classification
```

### Step 3: Deploy Training CronJob
```powershell
# Create training cronjob with RBAC
kubectl apply -f k8s/03-training-cronjob.yaml

# Verify cronjob created
kubectl get cronjob -n cats-dogs-classification
kubectl describe cronjob cats-dogs-training -n cats-dogs-classification
```

### Step 4: Deploy Inference Service
```powershell
# Deploy inference service
kubectl apply -f k8s/05-inference-deployment.yaml

# Wait for deployment
kubectl wait --for=condition=ready pod -l app=inference-service -n cats-dogs-classification --timeout=300s

# Verify service
kubectl get deployment inference-service -n cats-dogs-classification
kubectl get svc inference-service -n cats-dogs-classification
```

### Step 5: Setup Ingress
```powershell
# Deploy ingress
kubectl apply -f k8s/06-ingress.yaml

# Verify ingress
kubectl get ingress -n cats-dogs-classification
```

### Verify Complete Deployment
```powershell
# Check all resources
kubectl get all -n cats-dogs-classification

# Should show:
# - 1 Namespace
# - 2 Deployments (mlflow, inference-service)
# - 2 Services (mlflow, inference-service)
# - 1 CronJob (cats-dogs-training)
# - 3 PersistentVolumeClaims (data, models, mlflow)
# - 3+ Pods (2 for inference, 1 for mlflow, etc.)
```

---

## TRIGGER TRAINING JOB

### Run Training Once (Manual Trigger)
```powershell
# Get current timestamp for unique job name
$timestamp = Get-Date -Format "yyyyMMddHHmmss"
$jobName = "manual-training-run-$timestamp"

# Create job from cronjob
kubectl create job $jobName --from=cronjob/cats-dogs-training -n cats-dogs-classification

Write-Host "Training job created: $jobName"
Write-Host "Monitor with: kubectl logs -f job/$jobName -n cats-dogs-classification"

# Wait for job to start
Start-Sleep -Seconds 5
kubectl get jobs -n cats-dogs-classification
```

### Monitor Training Progress (REAL-TIME)
```powershell
# Watch job status
kubectl get jobs -n cats-dogs-classification -w

# In another PowerShell window, watch pod status
kubectl get pods -n cats-dogs-classification -w

# Check training logs (this shows everything)
kubectl logs -f job/manual-training-run-<timestamp> -n cats-dogs-classification

# Example output should show:
# ======================================================================
# Starting Kubernetes CronJob Training
# ======================================================================
# Available models: ['simple_cnn', 'logistic_regression', 'resnet18']
# Data directory: /data/processed
# 
# ============================================================
# Training simple_cnn model...
# ============================================================
# Epoch [1/20] Train Loss: 0.6521, Train Acc: 62.45% Val Loss: 0.5821, Val Acc: 65.32%
# Epoch [2/20] Train Loss: 0.4321, Train Acc: 78.90% Val Loss: 0.3721, Val Acc: 79.45%
# ...
# Epoch [20/20] Train Loss: 0.1823, Train Acc: 92.34% Val Loss: 0.3421, Val Acc: 87.56%
# Completed training simple_cnn
# 
# ============================================================
# Training logistic_regression model...
# ============================================================
# ...
# 
# ============================================================
# Training resnet18 model...
# ============================================================
# ...
#
# ============================================================
# Training Completed Successfully
# Best Model: resnet18
# Validation Accuracy: 94.23%
# Validation Loss: 0.1245
# ============================================================
```

### Wait for Training to Complete
```powershell
# Check job status
kubectl get job $jobName -n cats-dogs-classification

# When it shows "1/1" under COMPLETIONS, training is done
# If it says "1/1" under FAILURES, check logs for errors

# This should take approximately:
# - 5-10 min for SimpleCNN
# - 5-10 min for LogisticRegression
# - 10-15 min for ResNet18
# Total: ~25-40 minutes depending on your system
```

---

## VERIFY MODELS IN MLFLOW

### Setup Port Forwarding
```powershell
# In one PowerShell window, setup port forwarding
kubectl port-forward -n cats-dogs-classification svc/mlflow 5000:5000
Write-Host "MLFlow available at: http://localhost:5000"

# In another window, setup inference service access
kubectl port-forward -n cats-dogs-classification svc/inference-service 8000:8000
Write-Host "Inference API available at: http://localhost:8000/docs"
```

### Access MLFlow UI
```
Open browser: http://localhost:5000
```

**What you should see in MLFlow UI:**

1. **Left Panel - Experiments:**
   - Experiment name: `cats-dogs-k8s`

2. **Runs Section - All 3 Model Runs:**
   ```
   Run 1: SimpleCNN
   ├── Parameters:
   │   ├── epochs: 20
   │   ├── batch_size: 32
   │   └── lr: 0.001
   ├── Metrics:
   │   ├── val_loss: ~0.34
   │   ├── val_acc: ~87.5%
   │   ├── train_loss: ~0.18
   │   └── train_acc: ~92.3%
   └── Tags: model_name=simple_cnn
   
   Run 2: LogisticRegression
   ├── Metrics:
   │   ├── val_loss: ~0.54
   │   ├── val_acc: ~75.2%
   │   └── (Lower accuracy - expected)
   └── Tags: model_name=logistic_regression
   
   Run 3: ResNet18 (BEST ⭐)
   ├── Metrics:
   │   ├── val_loss: ~0.12 (LOWEST)
   │   ├── val_acc: ~94.2% (HIGHEST)
   │   ├── train_loss: ~0.09
   │   └── train_acc: ~96.8%
   ├── Tags:
   │   ├── type: best_model
   │   ├── environment: kubernetes
   │   └── model_name: resnet18
   └── Artifacts:
       └── best_model_resnet18.pkl
   ```

### Compare Models in MLFlow
1. Click **"Experiment"** tab
2. Select all 3 runs (checkboxes)
3. Click **"Compare"**
4. View side-by-side metrics and charts
5. See ResNet18 highlighted as best

### Download Model Comparison JSON
```powershell
# The comparison is automatically saved
# Location: /mnt/data/cats-dogs/models/best_model/model_comparison.json

# View it
Get-Content "C:\mnt\data\cats-dogs\models\best_model\model_comparison.json" | ConvertFrom-Json | ConvertTo-Json -Depth 5

# Expected output:
# {
#   "best_model": "resnet18",
#   "best_model_val_acc": 94.23,
#   "best_model_val_loss": 0.1245,
#   "all_models": {
#     "simple_cnn": {
#       "val_loss": 0.3421,
#       "val_acc": 87.56,
#       "train_loss": 0.1823,
#       "train_acc": 92.34
#     },
#     "logistic_regression": {
#       "val_loss": 0.5432,
#       "val_acc": 75.23,
#       "train_loss": 0.4821,
#       "train_acc": 78.90
#     },
#     "resnet18": {
#       "val_loss": 0.1245,
#       "val_acc": 94.23,
#       "train_loss": 0.0892,
#       "train_acc": 96.78
#     }
#   }
# }
```

---

## VERIFY BEST MODEL VERSIONING

### Check Best Model Files
```powershell
# List all saved models
Get-ChildItem "C:\mnt\data\cats-dogs\models\best_model\" -Recurse

# Should show:
# C:\mnt\data\cats-dogs\models\best_model\
# ├── best_model_resnet18.pkl (or whatever won - ~100MB)
# ├── model_comparison.json
# └── (temporary model files cleaned up)

# Check file sizes
Get-ChildItem "C:\mnt\data\cats-dogs\models\best_model\" | Format-Table Name, Length
```

### Verify Best Model in MLFlow
```powershell
# Query MLFlow API to get best model info
$response = Invoke-WebRequest -Uri "http://localhost:5000/api/2.0/mlflow/runs/search" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"experiment_ids": ["1"]}' | ConvertFrom-Json

$response.runs | Where-Object { $_.data.tags -like "*best_model*" } | 
  Select-Object -Property @{Name="RunID"; Expression={$_.info.run_id}}, `
                          @{Name="Model"; Expression={($_.data.tags | Where-Object { $_.key -eq "model_name" }).value}}, `
                          @{Name="Status"; Expression={$_.info.status}}

# Alternative: Query via UI
# Go to http://localhost:5000
# Click on experiment "cats-dogs-k8s"
# Look for run with tag "type: best_model"
# Click on it to see model details
```

### Test Inference with Best Model
```powershell
# Access Inference API Docs
# http://localhost:8000/docs

# Or test via PowerShell
$modelInfo = Invoke-RestMethod -Uri "http://localhost:8000/info"
Write-Host "Current Model: $($modelInfo.model)"
Write-Host "Classes: $($modelInfo.classes -join ', ')"

# Expected output:
# Current Model: ResNet18 (or whatever model is best)
# Classes: cat, dog
```

---

## Troubleshooting

### Issue: Training Pod Stuck or Not Starting
```powershell
# Check pod events
kubectl describe pod <pod-name> -n cats-dogs-classification

# Common causes:
# - Data not in /data/processed/ (check PV mount)
# - Insufficient memory (increase job limits in cronjob manifest)
# - Image not found (rebuild: docker build ...)
```

### Issue: Training Job Failed
```powershell
# Check full logs
kubectl logs job/$jobName -n cats-dogs-classification --all-containers=true

# Check if data exists in pod
kubectl exec job/$jobName -n cats-dogs-classification -- ls -la /data/processed/

# If data missing, copy it:
# Verify C:\mnt\data\cats-dogs\data\processed\ has data
# Make sure Rancher Desktop has volume sharing enabled
```

### Issue: Models Not in MLFlow
```powershell
# Check if MLFlow server is running
kubectl get pod -l app=mlflow -n cats-dogs-classification

# Check MLFlow logs
kubectl logs deployment/mlflow -n cats-dogs-classification

# Verify models were saved
kubectl exec deployment/inference-service -n cats-dogs-classification -- `
  powershell -Command "Get-ChildItem /models/best_model/"
```

### Issue: Can't Access MLFlow UI
```powershell
# Check port forwarding is running
Get-Process | Where-Object { $_.Name -like "*kubectl*" }

# Restart port forward
kubectl port-forward -n cats-dogs-classification svc/mlflow 5000:5000

# Or access via service internally
kubectl exec -it deployment/mlflow -n cats-dogs-classification -- curl http://localhost:5000
```

---

## Complete Workflow Summary

```
1. ✅ Build Docker Image
   docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .

2. ✅ Create Host Directories
   C:\mnt\data\cats-dogs\{data,models,mlflow}

3. ✅ Copy Dataset
   C:\mnt\data\cats-dogs\data\processed\{train,val,test}

4. ✅ Deploy to Kubernetes
   kubectl apply -f k8s/00-namespace.yaml
   kubectl apply -f k8s/01-persistent-volumes.yaml
   kubectl apply -f k8s/02-training-configmap.yaml
   kubectl apply -f k8s/03-training-cronjob.yaml
   kubectl apply -f k8s/04-mlflow-deployment.yaml
   kubectl apply -f k8s/05-inference-deployment.yaml
   kubectl apply -f k8s/06-ingress.yaml

5. ✅ Trigger Training Job
   kubectl create job manual-training-run-<timestamp> --from=cronjob/cats-dogs-training -n cats-dogs-classification

6. ✅ Monitor Training (takes ~25-40 min)
   kubectl logs -f job/manual-training-run-<timestamp> -n cats-dogs-classification

7. ✅ View Models in MLFlow
   kubectl port-forward svc/mlflow 5000:5000
   Open: http://localhost:5000

8. ✅ Verify Best Model
   - See all 3 models in MLFlow UI
   - ResNet18 marked as "best_model"
   - Compare metrics across all models
   - Download model_comparison.json

9. ✅ Test Inference API
   kubectl port-forward svc/inference-service 8000:8000
   Open: http://localhost:8000/docs
```

---

## Expected Outputs After Training

### Models Directory
```
C:\mnt\data\cats-dogs\models\best_model\
├── best_model_resnet18.pkl (100-200MB, depending on checkpoint)
├── model_comparison.json (shows all 3 models' metrics)
└── (training artifacts)
```

### MLFlow Experiment View
```
Experiment: cats-dogs-k8s
Runs:
- SimpleCNN: val_acc ~87.5%, val_loss ~0.34
- LogisticRegression: val_acc ~75.2%, val_loss ~0.54
- ResNet18: val_acc ~94.2%, val_loss ~0.12 ← BEST (tagged)
```

### Metrics Visible
```
Each model shows:
- epochs: 20
- batch_size: 32
- lr: 0.001
- val_accuracy, val_loss, train_accuracy, train_loss
- Tags: model_name, environment=kubernetes
```

---

## Quick Command Reference

```powershell
# View all resources
kubectl get all -n cats-dogs-classification

# View training progress
kubectl logs -f job/manual-training-run-<timestamp> -n cats-dogs-classification

# View MLFlow logs
kubectl logs -f deployment/mlflow -n cats-dogs-classification

# Check models saved
Get-ChildItem C:\mnt\data\cats-dogs\models\best_model\

# View model comparison
Get-Content C:\mnt\data\cats-dogs\models\best_model\model_comparison.json

# Port forward MLFlow
kubectl port-forward svc/mlflow 5000:5000 -n cats-dogs-classification

# Port forward Inference
kubectl port-forward svc/inference-service 8000:8000 -n cats-dogs-classification

# Delete everything
kubectl delete namespace cats-dogs-classification
```
