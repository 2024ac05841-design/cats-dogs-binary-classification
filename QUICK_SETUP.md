# Quick Setup Guide: Local Rancher Deployment

Deploy the complete MLOps pipeline on Rancher Desktop in minutes.

---

## 📋 Prerequisites

### **Required Software**

- ✅ Rancher Desktop (v1.5+) - [Download](https://rancherdesktop.io/)
- ✅ Docker CLI (included with Rancher Desktop)
- ✅ kubectl (included with Rancher Desktop)
- ✅ Git
- ✅ Python 3.11+ (for running locally before Docker)

### **System Requirements**

- 8GB+ RAM (recommended)
- 20GB+ disk space
- Windows, macOS, or Linux

### **GitHub Access**

- GitHub account with access to [cats-dogs-binary-classification](https://github.com/2024ac05841-design/cats-dogs-binary-classification)
- Personal Access Token (for GHCR if needed)

---

## 🚀 Step 1: Setup Rancher Desktop

### **1.1 Install Rancher Desktop**

```bash
# Download from https://rancherdesktop.io/
# Install following the official guide for your OS
```

### **1.2 Enable Kubernetes**

1. Open Rancher Desktop
2. Go to **Settings** → **Kubernetes**
3. Enable **Kubernetes** checkbox
4. Select **v1.28+** (latest stable)
5. Click **Apply**
6. Wait for initialization (~2-3 minutes)

### **1.3 Verify Installation**

```bash
# Check Rancher Desktop status
kubectl cluster-info

# Output should show:
# Kubernetes master is running at https://...
# CoreDNS is running at https://...
```

### **1.4 Switch to Rancher Context**

```powershell
# Windows PowerShell:
# Rancher Desktop automatically sets context
# Verify with:
kubectl config current-context
# Should show: rancher-desktop
```

---

## 📥 Step 2: Clone Repository

```bash
# Clone the repository
git clone https://github.com/2024ac05841-design/cats-dogs-binary-classification.git

# Navigate to project
cd cats-dogs-binary-classification

# Verify you're on Main branch
git branch
# Should show: * Main
```

---

## 🔧 Step 3: Prepare Kubernetes Namespace & Secrets

### **3.1 Create Namespace**

```bash
# Create the namespace for the project
kubectl create namespace cats-dogs-classification

# Verify namespace created
kubectl get namespace
```

### **3.2 Create Docker Registry Secret (Optional - for private GHCR)**

```bash
# If using private GHCR, create secret:
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=YOUR_GITHUB_USERNAME \
  --docker-password=YOUR_GITHUB_PAT \
  --docker-email=your.email@example.com \
  -n cats-dogs-classification

# Verify secret created
kubectl get secrets -n cats-dogs-classification
```

### **3.3 Create ConfigMap for Model Path (Optional)**

```bash
# ConfigMap for model location
kubectl create configmap model-config \
  --from-literal=MODEL_PATH=/models/model.pkl \
  --from-literal=LOG_LEVEL=INFO \
  -n cats-dogs-classification

# Verify ConfigMap
kubectl get configmap -n cats-dogs-classification
```

---

## 📦 Step 4: Deploy Services

### **4.1 Create Persistent Volumes (for model & logs)**

```bash
# Apply PV configuration
kubectl apply -f k8s/01-persistent-volumes.yaml

# Verify PVs created
kubectl get pv
```

### **4.2 Deploy Inference Service**

```bash
# Apply inference service deployment
kubectl apply -f k8s/05-inference-deployment.yaml

# Watch deployment progress
kubectl rollout status deployment/inference-service \
  -n cats-dogs-classification

# Should complete with: deployment "inference-service" successfully rolled out
```

### **4.3 Deploy MLflow Server (Optional but Recommended)**

```bash
# Apply MLflow deployment
kubectl apply -f k8s/04-mlflow-deployment.yaml

# Verify MLflow is running
kubectl get pods -n cats-dogs-classification | grep mlflow
```

### **4.4 Deploy Prometheus (for metrics)**

```bash
# Apply Prometheus deployment
kubectl apply -f k8s/01-prometheus.yaml

# Verify Prometheus is running
kubectl get pods -n cats-dogs-classification | grep prometheus
```

### **4.5 Deploy Grafana (for dashboards)**

```bash
# Apply Grafana deployment
kubectl apply -f k8s/02-grafana.yaml

# Verify Grafana is running
kubectl get pods -n cats-dogs-classification | grep grafana
```

### **4.6 Deploy Training CronJob (Optional)**

```bash
# Apply training cronjob (runs on schedule)
kubectl apply -f k8s/03-training-cronjob.yaml

# Verify cronjob created
kubectl get cronjob -n cats-dogs-classification
```

---

## 5️⃣ Step 5: Verify All Services Are Running

```bash
# Check all pods status
kubectl get pods -n cats-dogs-classification

# Expected output (all should be Running):
# NAME                                READY   STATUS    RESTARTS   AGE
# inference-service-xxxxx             1/1     Running   0          2m
# mlflow-xxxxx                        1/1     Running   0          2m
# prometheus-xxxxx                    1/1     Running   0          2m
# grafana-xxxxx                       1/1     Running   0          2m

# Check services
kubectl get svc -n cats-dogs-classification

# Check persistent volumes
kubectl get pv,pvc -n cats-dogs-classification
```

---

## 🌐 Step 6: Access Services

### **6.1 Inference Service**

```bash
# Port forward to access API
kubectl port-forward svc/inference-service 8000:8000 \
  -n cats-dogs-classification

# Access in browser or curl:
curl http://localhost:8000/health

# Swagger UI (interactive documentation):
# Open: http://localhost:8000/docs
```

### **6.2 MLflow Server**

```bash
# Port forward MLflow
kubectl port-forward svc/mlflow 5000:5000 \
  -n cats-dogs-classification

# Access MLflow UI:
# Open: http://localhost:5000
```

### **6.3 Prometheus**

```bash
# Port forward Prometheus
kubectl port-forward svc/prometheus 9090:9090 \
  -n cats-dogs-classification

# Access Prometheus UI:
# Open: http://localhost:9090
```

### **6.4 Grafana**

```bash
# Port forward Grafana
kubectl port-forward svc/grafana 3000:3000 \
  -n cats-dogs-classification

# Access Grafana UI:
# Open: http://localhost:3000
# Default credentials: admin / admin
```

---

## ✅ Step 7: Test Inference Service

### **7.1 Health Check**

```bash
# Simple health check
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","model_loaded":true}
```

### **7.2 Make Prediction**

```bash
# Upload a test image for prediction
curl -X POST -F "file=@path/to/test_image.jpg" \
  http://localhost:8000/predict

# Expected response (example):
# {
#   "class_name": "dog",
#   "confidence": 0.9999,
#   "processing_time_ms": 145.32
# }
```

### **7.3 View Metrics**

```bash
# Get Prometheus metrics
curl http://localhost:8000/metrics

# Expected: Prometheus format metrics
# request_count_total{endpoint="/predict"} 5
# request_latency_seconds_bucket{...} 0.145
```

---

## 🛑 Step 8: Cleanup & Tear Down

### **Remove All Services**

```bash
# Delete entire namespace (removes all resources)
kubectl delete namespace cats-dogs-classification

# Or selectively delete:
kubectl delete deployment inference-service \
  -n cats-dogs-classification
kubectl delete deployment mlflow -n cats-dogs-classification
kubectl delete deployment prometheus -n cats-dogs-classification
kubectl delete deployment grafana -n cats-dogs-classification
kubectl delete cronjob training-job -n cats-dogs-classification
```

### **Stop Port Forwarding**

```bash
# Press Ctrl+C in terminal windows running port-forward
# Or manually close the terminals
```

---

## 🐛 Troubleshooting

### **Pods Not Starting**

```bash
# Check pod logs for errors
kubectl logs <pod-name> -n cats-dogs-classification

# Example:
kubectl logs inference-service-xxxxx \
  -n cats-dogs-classification --tail=50

# Check events for cluster issues
kubectl describe pod <pod-name> -n cats-dogs-classification
```

### **Image Pull Errors**

```bash
# If using GHCR, verify secret is attached to deployment
kubectl get deployment inference-service -o yaml \
  -n cats-dogs-classification | grep imagePullSecrets

# If not present, patch the deployment:
kubectl patch serviceaccount default \
  -n cats-dogs-classification \
  -p '{"imagePullSecrets": [{"name": "ghcr-secret"}]}'
```

### **Port Forward Not Working**

```bash
# Check if port is already in use
# Windows PowerShell:
netstat -ano | findstr :8000

# If port is in use, use different port:
kubectl port-forward svc/inference-service 8001:8000 \
  -n cats-dogs-classification
```

### **Persistent Volume Issues**

```bash
# Check PV and PVC status
kubectl get pv,pvc -n cats-dogs-classification

# Check storage class
kubectl get storageclass

# For Rancher Desktop, should use 'local-path' storage class
```

### **Out of Memory**

```bash
# Check node resources
kubectl top nodes
kubectl top pods -n cats-dogs-classification

# If low memory:
# 1. Increase Rancher Desktop memory in Settings
# 2. Reduce replica count in deployment manifests
# 3. Adjust resource limits in k8s/*.yaml files
```

---

## 📊 Monitoring Deployment

### **Real-Time Pod Monitoring**

```bash
# Watch pods continuously
kubectl get pods -n cats-dogs-classification --watch

# Check resource usage
kubectl top pods -n cats-dogs-classification
```

### **View Service Status**

```bash
# Get detailed service information
kubectl describe svc inference-service \
  -n cats-dogs-classification

# Get service IP (internal)
kubectl get svc -n cats-dogs-classification
```

### **Check Application Logs**

```bash
# Inference service logs
kubectl logs deployment/inference-service \
  -n cats-dogs-classification -f

# Follow logs in real-time (add -f flag)
```

---

## 🎯 Quick Commands Reference

```bash
# Create namespace
kubectl create namespace cats-dogs-classification

# Deploy all services
kubectl apply -f k8s/ -n cats-dogs-classification

# Check status
kubectl get pods -n cats-dogs-classification

# Port forward inference API
kubectl port-forward svc/inference-service 8000:8000 \
  -n cats-dogs-classification

# Port forward Grafana
kubectl port-forward svc/grafana 3000:3000 \
  -n cats-dogs-classification

# View logs
kubectl logs deployment/inference-service \
  -n cats-dogs-classification

# Delete all
kubectl delete namespace cats-dogs-classification

# Switch to namespace (optional, for shorter commands)
kubectl config set-context --current \
  --namespace=cats-dogs-classification
```

---

## 🎬 Complete Setup in One Command (Advanced)

```bash
# Create namespace and deploy everything
kubectl create namespace cats-dogs-classification && \
kubectl apply -f k8s/ -n cats-dogs-classification && \
kubectl rollout status deployment/inference-service \
  -n cats-dogs-classification

# Then access services:
# Terminal 1: kubectl port-forward svc/inference-service 8000:8000 -n cats-dogs-classification
# Terminal 2: kubectl port-forward svc/grafana 3000:3000 -n cats-dogs-classification
# Terminal 3: kubectl port-forward svc/prometheus 9090:9090 -n cats-dogs-classification
```

---

## ✨ What You Should See After Setup

✅ **All pods running** (inference, mlflow, prometheus, grafana)
✅ **Inference API responding** at http://localhost:8000/health
✅ **Swagger UI working** at http://localhost:8000/docs
✅ **Grafana dashboards** at http://localhost:3000
✅ **Prometheus metrics** at http://localhost:9090
✅ **MLflow experiments** at http://localhost:5000

---

## 📞 Support

- **GitHub Issues:** [Project Repository](https://github.com/2024ac05841-design/cats-dogs-binary-classification)
- **Rancher Desktop Docs:** https://docs.rancherdesktop.io/
- **Kubernetes Docs:** https://kubernetes.io/docs/
- **kubectl Cheatsheet:** https://kubernetes.io/docs/reference/kubectl/cheatsheet/

---

## 🔄 Useful Development Workflow

```bash
# Terminal 1: Monitor pods
kubectl get pods -n cats-dogs-classification --watch

# Terminal 2: Stream logs
kubectl logs deployment/inference-service \
  -n cats-dogs-classification -f

# Terminal 3: Port forward
kubectl port-forward svc/inference-service 8000:8000 \
  -n cats-dogs-classification

# Terminal 4: Test API
# while true; do curl http://localhost:8000/health; sleep 2; done
```

---
