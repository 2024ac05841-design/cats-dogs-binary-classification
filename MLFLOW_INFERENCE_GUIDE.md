# MLFlow-Based Inference Model Fetching

## Overview

The inference container now automatically fetches the latest best model from MLFlow on startup with intelligent caching. This ensures that:

1. **Always Latest Model**: Inference pods always use the newest best model without manual updates
2. **Automatic Versioning**: Model versions are tracked and updated automatically
3. **Smart Caching**: Models are cached locally and only re-downloaded if version changes
4. **Zero-Downtime Updates**: Pod restarts automatically fetch new models without interruption
5. **MLFlow Fallback**: If MLFlow is unavailable, falls back to cached model gracefully

## Architecture

```
Inference Pod Startup
    ↓
MLFlowModelFetcher.fetch_best_model()
    ↓
    ├─→ Check MLFlow Connection
    │   ├─→ Success: Continue
    │   └─→ Failed: Use cached model or untrained fallback
    │
    ├─→ Get Experiment (cats-dogs-k8s)
    │   ├─→ Found: Look for best model
    │   └─→ Not Found: Use cached model
    │
    ├─→ Find Best Model Run
    │   │   (Priority: tagged "best_model" → highest val_acc)
    │   │
    │   ├─→ Found: Continue
    │   └─→ Not Found: Use cached model
    │
    ├─→ Load Version Information
    │   ├─→ Cached version exists?
    │   │   ├─→ YES: Version changed?
    │   │   │   ├─→ YES: Download new model
    │   │   │   └─→ NO: Use cached model (SKIP DOWNLOAD)
    │   │   └─→ NO: Download model
    │   │
    │   ├─→ Download from MLFlow
    │   ├─→ Save model_version.json (metadata)
    │   └─→ Load model into memory
    │
    ↓
Load Model Weights
    ↓
Ready for Inference
```

## Key Features

### 1. Smart Version Tracking

Each cached model includes version information in `model_version.json`:

```json
{
  "run_id": "abc123def456",
  "model_name": "resnet18",
  "val_accuracy": 94.23,
  "fetched_at": "2024-08-26T10:30:45.123456",
  "mlflow_uri": "http://mlflow:5000",
  "experiment_name": "cats-dogs-k8s"
}
```

### 2. Version Change Detection

The fetcher compares current best model with cached version:

- **Different run_id**: New training detected → Download
- **Different accuracy**: Data/params changed → Download
- **Same run & accuracy**: No download needed (use cache)

Benefits:
- ✅ Network bandwidth saved
- ✅ Pod startup time reduced (no HTTP transfer)
- ✅ MLFlow server load reduced
- ✅ Faster horizontal scaling of inference pods

### 3. Fallback Chain

If any step fails, gracefully fall back:

```
1. MLFlow + Fresh Download
   ↓ (fails)
2. MLFlow + Cached Version
   ↓ (fails)
3. Local Cached Model
   ↓ (fails)
4. Untrained Default Model (simple_cnn)
```

## Configuration

### Environment Variables

In Kubernetes deployment or Docker container:

```bash
# MLFlow tracking server URI
MLFLOW_TRACKING_URI=http://mlflow:5000

# Experiment name containing trained models
EXPERIMENT_NAME=cats-dogs-k8s

# Where to cache downloaded models
MODEL_PATH=/app-models/model.pkl
```

### In Kubernetes (05-inference-deployment.yaml)

```yaml
env:
- name: MODEL_PATH
  value: "/app-models/model.pkl"
- name: MLFLOW_TRACKING_URI
  value: "http://mlflow:5000"
- name: EXPERIMENT_NAME
  value: "cats-dogs-k8s"
```

## Model Selection Logic

### Best Model Identification

MLFlow is queried with this priority:

1. **Tagged Best Model**: Run with `type=best_model` tag (set by training script)
2. **Highest Accuracy**: If no tag, use run with highest `val_acc` metric

### Model Name Detection

From MLFlow run parameters/tags:
- `resnet*` → ResNet18 architecture
- `logistic*` → Logistic Regression architecture
- Default → Simple CNN architecture

## Usage in Code

### Automatic (Recommended)

Inference container handles everything on startup:

```python
# In src/inference/app.py
def load_model_on_startup():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_name = os.getenv("EXPERIMENT_NAME", "cats-dogs-k8s")
    
    success, model_path, info = load_model_with_mlflow(
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
    )
    
    # Use model_path and check info["source"] and info["message"]
```

### Manual Usage

For testing or scripting:

```python
from src.inference.mlflow_model_fetcher import MLFlowModelFetcher

# Create fetcher
fetcher = MLFlowModelFetcher(
    mlflow_uri="http://mlflow:5000",
    experiment_name="cats-dogs-k8s",
    cache_dir="/app-models",
)

# Fetch best model
success, model_path, info = fetcher.fetch_best_model()

if success:
    print(f"Model loaded from {info['source']}")
    print(f"Model: {info['model_name']}")
    print(f"Accuracy: {info['val_accuracy']:.4f}")
    print(f"Cached: {info['cached']}")
else:
    print(f"Failed: {info['message']}")
```

## Monitoring

### Check Model Source

Use `/info` endpoint to verify which model is loaded:

```bash
# Get model information
curl http://localhost:8000/info

# Response:
{
  "service": "Cats vs Dogs Classifier",
  "version": "1.0.0",
  "device": "cuda",
  "model_loaded": true,
  "classes": ["cat", "dog"],
  "model_info": {
    "source": "mlflow_cached",      # or mlflow_fresh, local_fallback, untrained_fallback
    "model_name": "resnet18",
    "val_accuracy": 94.23,
    "run_id": "abc123def456",
    "cached": true,                 # true = no download needed
    "message": "Using cached ResNet18 model"
  },
  "endpoints": {...}
}
```

### Check Container Logs

```bash
# View startup logs
kubectl logs -f deployment/inference-service -n cats-dogs-classification

# Expected output:
# [INFO] Using device: cuda
# [INFO] Attempting to fetch best model from MLFlow: http://mlflow:5000
# [INFO] MLFlow client initialized with URI: http://mlflow:5000
# [INFO] Found experiment 'cats-dogs-k8s' with ID: 0
# [INFO] Found best model: resnet18 (run: abc123, val_acc: 0.9423)
# [INFO] Model version unchanged, using cached version
# [INFO] Model loaded successfully from mlflow_cached: resnet18 (val_acc: 94.23) - Using cached resnet18 model
```

### View Cached Version Info

```bash
# Inside container or from host
kubectl exec -it pod/inference-service-xyz -n cats-dogs-classification -- \
  cat /app-models/model_version.json

# Output:
{
  "run_id": "abc123def456",
  "model_name": "resnet18",
  "val_accuracy": 94.23,
  "fetched_at": "2024-08-26T10:30:45.123456",
  "mlflow_uri": "http://mlflow:5000",
  "experiment_name": "cats-dogs-k8s"
}
```

## Workflow: New Training Deployment

When you train new models and want inference to use them:

### Step 1: Complete Training
```bash
kubectl create job manual-training-run-$(date +%s) \
  --from=cronjob/cats-dogs-training \
  -n cats-dogs-classification
```

### Step 2: Monitor Training
```bash
kubectl logs -f job/manual-training-run-<timestamp> -n cats-dogs-classification
```

### Step 3: Verify Best Model in MLFlow
```bash
# Port forward MLFlow
kubectl port-forward svc/mlflow 5000:5000 -n cats-dogs-classification

# Open browser: http://localhost:5000
# Verify ResNet18 is marked with "best_model" tag
```

### Step 4: Restart Inference Pods
```bash
# Trigger model update by restarting inference pods
kubectl rollout restart deployment/inference-service \
  -n cats-dogs-classification

# Pods will automatically fetch new best model on startup
```

### Step 5: Verify New Model Loaded
```bash
# Check model info endpoint
kubectl port-forward svc/inference-service 8000:8000 \
  -n cats-dogs-classification

curl http://localhost:8000/info | jq '.model_info'

# Should show:
# - source: mlflow_fresh (if downloaded) or mlflow_cached (if version unchanged)
# - Updated run_id, val_accuracy
# - cached: false (if freshly downloaded)
```

## Performance Characteristics

### Startup Time

- **First time**: ~3-5 seconds (download model + load)
- **Subsequent restarts**: ~1-2 seconds (cached, no download)
- **With version change**: ~3-5 seconds (re-download + load)

### Network Usage

- **Typical model size**: 100-300 MB (ResNet18: ~104MB)
- **With caching**: Network used only when version changes
- **Bandwidth saved**: ~300MB per pod × N replicas for unchanged versions

### Storage Usage

- **Per pod**: ~300MB for cached model + ~2KB for metadata
- **2 replica pods**: ~600MB total cache
- **emptyDir volume**: Ephemeral, cleared on pod restart

## Troubleshooting

### Issue: Pod stuck in ImagePullBackOff or CrashLoopBackOff

```bash
# Check logs
kubectl logs -f deployment/inference-service -n cats-dogs-classification

# Common causes:
# 1. MLFlow not ready - Wait for MLFlow pod to be running
# 2. Network issue - Check pod can reach mlflow service
# 3. Bad credentials - Check MLFLOW_TRACKING_URI env var
```

**Solution**:
```bash
# Verify MLFlow is running
kubectl get pods -n cats-dogs-classification | grep mlflow

# Test connectivity from inference pod
kubectl run -it --rm debug --image=curlimages/curl --restart=Never \
  -n cats-dogs-classification -- \
  curl http://mlflow:5000/

# If MLFlow not responding, check its logs
kubectl logs deployment/mlflow -n cats-dogs-classification
```

### Issue: Using "untrained_fallback" instead of trained model

```bash
# Check MLFlow has the experiment
kubectl logs deployment/mlflow -n cats-dogs-classification | grep "experiment"

# Verify training completed successfully
kubectl logs job/<training-job-name> -n cats-dogs-classification | tail -20

# Check best model is tagged in MLFlow
# Go to MLFlow UI and verify run has "type=best_model" tag
```

**Solution**:
```bash
# Re-run training
kubectl create job manual-training-run-$(date +%s) \
  --from=cronjob/cats-dogs-training \
  -n cats-dogs-classification

# Wait for training to complete
# Restart inference pods
kubectl rollout restart deployment/inference-service -n cats-dogs-classification
```

### Issue: Model not updating after new training

**Cause**: Version detected as unchanged, using old cached model

```bash
# Verify new model was actually saved with different val_acc
kubectl logs job/<new-training-job> -n cats-dogs-classification | grep "val_acc"

# Check cached version
kubectl exec -it pod/inference-service-xyz -n cats-dogs-classification -- \
  cat /app-models/model_version.json
```

**Solution**: If metrics are truly different:
```bash
# Force re-download by clearing cache
kubectl delete pods -l app=inference-service -n cats-dogs-classification

# Pods recreate and fetch fresh model from MLFlow
# Check model_version.json is updated
```

### Issue: MLFlow connection refused

```bash
# Verify MLFlow is running
kubectl get svc mlflow -n cats-dogs-classification

# Test from another pod
kubectl run -it --rm debug --image=python:3.11 --restart=Never \
  -n cats-dogs-classification -- \
  python -c "import requests; print(requests.get('http://mlflow:5000/').text)"
```

**Solution**:
```bash
# Wait for MLFlow to be ready
kubectl wait --for=condition=ready pod -l app=mlflow \
  -n cats-dogs-classification --timeout=300s

# Inference pods will auto-recover after MLFlow ready
```

## Implementation Details

### MLFlowModelFetcher Class

Located in `src/inference/mlflow_model_fetcher.py`:

**Key Methods**:
- `fetch_best_model()`: Main entry point - returns (success, path, info)
- `_find_best_model()`: Query MLFlow for best run
- `_get_model_artifact()`: Download model from MLFlow
- `_version_changed()`: Compare versions to detect updates
- `_load_cached_version()`: Read model_version.json
- `_save_version()`: Write model_version.json

**Logging**: Comprehensive INFO/WARNING logs for debugging

## Best Practices

1. **Always tag best model during training**
   - Training script automatically tags with `type=best_model`
   - Easier MLFlow querying and model identification

2. **Monitor model_info endpoint**
   - Check `source` field to confirm model loading source
   - Track `cached` field to understand download patterns

3. **Plan pod updates**
   - Pod restart = automatic model refresh
   - No deployment changes needed for new model
   - Plan restarts during low-traffic periods

4. **Test before production**
   - Verify new model in MLFlow first
   - Check /info endpoint shows correct model
   - Run test predictions before routing traffic

5. **Keep MLFlow accessible**
   - Inference pods depend on MLFlow availability
   - Use appropriate resource limits to prevent outages
   - Monitor MLFlow logs for errors

## Summary

The MLFlow-based inference system provides:
- ✅ Automatic best model selection
- ✅ Intelligent version caching (no re-download if unchanged)
- ✅ Graceful fallback chain
- ✅ Transparent model updates via pod restarts
- ✅ Comprehensive logging and monitoring
- ✅ Production-ready reliability

Deploy with confidence knowing your inference pods always serve the latest and greatest trained models!
