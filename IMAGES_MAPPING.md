# Image to Documentation Mapping

Complete mapping of all images in `/images` folder to their corresponding document sections and use cases.

## 📊 Summary Statistics

- **Total Images:** 15 screenshots
- **Documents Updated:** 4 (M1, M2, M4, M5)
- **Categories:** MLflow, Swagger API, Kubernetes/Rancher, Prometheus, Grafana

---

## 📸 Image Mapping Guide

### 🔬 M1: Model Development & Experiment Tracking

**File:** `docs/01-m1-model-development.md`

#### 1. MLflow Experiments Tracking
- **Image:** `mlflow-experiments.png`
- **Location:** Section "📸 MLflow Dashboard Screenshots → Experiment Tracking & Comparison"
- **Purpose:** Show experiment runs, hyperparameter comparison, and metrics visualization
- **Key Features Shown:**
  - Multiple GPU training runs
  - Accuracy comparison: 59.60% → 99.20% → 99.60%
  - Run filtering and analysis
  - Validation loss tracking

#### 2. MLflow Model Registry & Versioning
- **Image:** `mlflow-models-versioning.png`
- **Location:** Section "📸 MLflow Dashboard Screenshots → Model Registry & Versioning"
- **Purpose:** Display registered model versions and production promotion
- **Key Features Shown:**
  - Registered model: "cats-dogs-best-model"
  - Version 1 (Production stage)
  - ResNet18 architecture
  - Performance metrics: 99.6% val accuracy
  - Model tags (Framework, Device, Training date)

---

### 📡 M2: Model Packaging & Containerization

**File:** `docs/02-m2-packaging.md`

#### 3. Swagger UI - API Endpoint Documentation
- **Image:** `swagger-endpoints.png`
- **Location:** Section "📸 API Documentation & Testing → Swagger UI - Endpoint Documentation"
- **Purpose:** Show interactive API documentation interface
- **Key Features Shown:**
  - Complete endpoint listing
  - Swagger UI at `/docs`
  - Request/response schema documentation
  - Try-it-out interface for testing
  - All endpoints: /health, /predict, /info, /metrics, /logs, /stats

#### 4. Swagger Sample Prediction Response (Dog Classification)
- **Image:** `swagger-sample-dog-response.png`
- **Location:** Section "📸 API Documentation & Testing → Sample Prediction Response"
- **Purpose:** Demonstrate prediction endpoint with actual response
- **Key Features Shown:**
  - POST /predict request with image upload
  - Response: Dog classification with 99.99% confidence
  - Probability breakdown (cat: 0.01%, dog: 99.99%)
  - HTTP 200 status code
  - Curl command for API testing

---

### ☸️ M4: CD Pipeline & Deployment

**File:** `docs/04-m4-cd-deployment.md`

#### 5. Rancher - Kubernetes Pods Overview
- **Image:** `rancher-pods.png`
- **Location:** Section "📸 Kubernetes Cluster Management via Rancher → Cluster Pods Overview"
- **Purpose:** Show all running pods in the namespace
- **Key Features Shown:**
  - All service pods running (Ready: 1/1)
  - cats-dogs-training pod
  - grafana pod (monitoring)
  - inference-service pod with replicas
  - mlflow pod (model registry)
  - prometheus pod (metrics)
  - Namespace: cats-dogs-classification

#### 6. Rancher - Deployments & Rollouts
- **Image:** `rancher-deployments.png`
- **Location:** Section "📸 Kubernetes Cluster Management via Rancher → Deployments & Rollouts"
- **Purpose:** Display deployment status and image management
- **Key Features Shown:**
  - Inference service deployment (1/1 ready)
  - Image registry: ghcr.io/2024ac05841-design/*
  - Pod age and restart tracking
  - Deployment metadata
  - Service load balancing status

#### 7. Rancher - Training CronJob Schedule
- **Image:** `rancher-training-job.png`
- **Location:** Section "📸 Kubernetes Cluster Management via Rancher → Training CronJob Schedule"
- **Purpose:** Display automated training job configuration
- **Key Features Shown:**
  - CronJob: cats-dogs-training
  - Schedule: 0 2 * * 0 (2 AM every Sunday)
  - Status: Suspended (toggleable)
  - Docker image: ghcr.io/2024ac05841-design/cats-dogs-classifier-training:latest
  - Training command execution

---

### 📈 M5: Monitoring, Logs & Model Performance

**File:** `docs/05-m5-monitoring.md`

#### 8. Prometheus - Service Discovery Configuration
- **Image:** `prometheus-service-discovery.png`
- **Location:** Section "📸 Monitoring Dashboards & Visualizations → Prometheus Target Health & Service Discovery"
- **Purpose:** Show Kubernetes service discovery setup
- **Key Features Shown:**
  - Discovered service labels
  - Target labels configuration
  - kubernetes_sd_configs setup
  - Scrape endpoint configuration
  - Service discovery rules

#### 9. Prometheus - Query Results & Metrics
- **Image:** `prometheus-query-results.png`
- **Location:** Section "📸 Monitoring Dashboards & Visualizations → Prometheus Query Results"
- **Purpose:** Display metric query execution and visualization
- **Key Features Shown:**
  - Time series query execution
  - Uptime metric: up{job="inference-service"} = 1
  - Query graph with time range
  - Metric value: 1.0 (healthy)
  - Query interface and history

#### 10. Prometheus - Target Health Status
- **Image:** `prometheus-target-health.png`
- **Location:** Section "📸 Monitoring Dashboards & Visualizations → Prometheus Target Status"
- **Purpose:** Show target scraping status and configuration
- **Key Features Shown:**
  - Target status: All healthy (1/1 up)
  - Endpoint: 10.42.0.2:8000/metrics
  - Scrape interval: 5s
  - Last scrape: Success
  - Target metadata

#### 11. Grafana - Main Dashboard List
- **Image:** `graphana-dashboards.png`
- **Location:** Section "📸 Monitoring Dashboards & Visualizations → Grafana Dashboard Visualizations → Main Dashboard Overview"
- **Purpose:** Show available dashboards and navigation
- **Key Features Shown:**
  - Dashboard creation interface
  - "Inference & Model Telemetry" dashboard
  - "Request & Response Log Audit" dashboard
  - Model status indicators
  - Key KPI widgets

#### 12. Grafana - Inference & Model Telemetry (Part 1)
- **Image:** `graphana-inference-model-metrics-1.png`
- **Location:** Section "📸 Monitoring Dashboards & Visualizations → Grafana Dashboard Visualizations → Inference & Model Telemetry - Part 1"
- **Purpose:** Display model performance and request metrics
- **Key Features Shown:**
  - Model Status: ResNet18 (Production, v1)
  - Total Predictions: 10
  - Average Latency: 1.59s
  - Inference Errors: 0
  - Traffic breakdown by endpoint
  - Latency percentiles (p50, p95, p99)

#### 13. Grafana - Inference & Model Telemetry (Part 2)
- **Image:** `graphana-inference-model-metrics-2.png`
- **Location:** Section "📸 Monitoring Dashboards & Visualizations → Grafana Dashboard Visualizations → Inference & Model Telemetry - Part 2"
- **Purpose:** Show prediction distribution and confidence analysis
- **Key Features Shown:**
  - Latency percentiles visualization
  - Model forward pass execution time
  - Prediction by class: Cat (6), Dog (4)
  - Prediction confidence distribution
  - 90%-100% confidence range dominance

#### 14. Grafana - Request/Response Audit (Part 1)
- **Image:** `graphana-request-response-audit-1.png`
- **Location:** Section "📸 Monitoring Dashboards & Visualizations → Grafana Dashboard Visualizations → Request/Response Log Audit - Part 1"
- **Purpose:** Display request audit trail and error tracking
- **Key Features Shown:**
  - Recent Prediction Requests: 10
  - Successful Predictions: 10
  - Failed/Errored Requests: 0
  - Incoming Prediction Rate (req/sec)
  - HTTP Status Code Breakdown
  - Success rate: 100%

#### 15. Grafana - Request/Response Audit (Part 2)
- **Image:** `graphana-request-response-audit-2.png`
- **Location:** Section "📸 Monitoring Dashboards & Visualizations → Grafana Dashboard Visualizations → Request/Response Log Audit - Part 2"
- **Purpose:** Show real-time monitoring of inference service
- **Key Features Shown:**
  - Same metrics as Part 1
  - Request rate: ~0.08 req/s
  - All HTTP 200 responses (0 errors)
  - Consistent successful prediction stream
  - Real-time health monitoring

---

## 🎯 How Images Enhance Documentation

### For M1 (Model Development)
✅ **Visual Proof:** Shows actual MLflow runs and model performance progression  
✅ **Experiment Comparison:** Demonstrates how different models were evaluated  
✅ **Production Promotion:** Shows which model version is active in production  

### For M2 (Packaging)
✅ **API Testing:** Shows real prediction examples and confidence scores  
✅ **Documentation:** Proves Swagger UI is working and accessible  
✅ **Response Format:** Demonstrates exact API response structure  

### For M4 (Deployment)
✅ **Cluster Health:** Shows all components running successfully  
✅ **Image Management:** Proves GHCR images are pulled and deployed  
✅ **Automation:** Demonstrates CronJob scheduling for model retraining  

### For M5 (Monitoring)
✅ **Metrics Collection:** Shows Prometheus successfully scraping inference service  
✅ **Dashboard Visualization:** Demonstrates real-time monitoring capabilities  
✅ **Audit Trail:** Shows request/response logging working end-to-end  

---

## 📋 File Organization

```
/images/                                    # Image folder (15 images)
├── mlflow-experiments.png                 # M1: Experiment tracking
├── mlflow-models-versioning.png           # M1: Model registry
├── swagger-endpoints.png                  # M2: API documentation
├── swagger-sample-dog-response.png        # M2: Prediction example
├── rancher-deployments.png                # M4: K8s deployments
├── rancher-pods.png                       # M4: Running pods
├── rancher-training-job.png               # M4: CronJob schedule
├── prometheus-query-results.png           # M5: Metric queries
├── prometheus-service-discovery.png       # M5: Service discovery
├── prometheus-target-health.png           # M5: Target status
├── graphana-dashboards.png                # M5: Dashboard list
├── graphana-inference-model-metrics-1.png # M5: Model metrics (1)
├── graphana-inference-model-metrics-2.png # M5: Model metrics (2)
├── graphana-request-response-audit-1.png  # M5: Request audit (1)
└── graphana-request-response-audit-2.png  # M5: Request audit (2)
```

---

## ✅ Verification Checklist

- [x] All 15 images placed in /images folder
- [x] M1 doc: MLflow experiment tracking screenshot added
- [x] M1 doc: MLflow model registry screenshot added
- [x] M2 doc: Swagger endpoints screenshot added
- [x] M2 doc: Swagger prediction response screenshot added
- [x] M4 doc: Rancher pods overview screenshot added
- [x] M4 doc: Rancher deployments screenshot added
- [x] M4 doc: Rancher training job screenshot added
- [x] M5 doc: Prometheus service discovery screenshot added
- [x] M5 doc: Prometheus query results screenshot added
- [x] M5 doc: Prometheus target health screenshot added
- [x] M5 doc: Grafana dashboard overview screenshot added
- [x] M5 doc: Grafana inference metrics part 1 screenshot added
- [x] M5 doc: Grafana inference metrics part 2 screenshot added
- [x] M5 doc: Grafana audit trail part 1 screenshot added
- [x] M5 doc: Grafana audit trail part 2 screenshot added

---

## 🎓 Documentation Structure Benefits

### Before (ASCII Art)
- Conceptual understanding only
- No visual proof of working system
- Difficult to verify implementation

### After (Real Screenshots)
✅ **Visual Proof:** Demonstrates actual working system  
✅ **Confidence:** Shows metrics, dashboards, and logs functioning  
✅ **Professionalism:** Professional documentation with evidence  
✅ **Troubleshooting:** Real data helps debug issues  
✅ **Team Alignment:** Everyone sees same operational state  

---

## 🚀 Next Steps

1. **README Update:** Consider adding image gallery to main README
2. **Additional Screenshots:** 
   - Training logs and output
   - Local Docker compose setup
   - GitHub Actions CI/CD pipeline execution
3. **Image Captions:** Each screenshot includes detailed feature descriptions
4. **Cross-References:** All images linked from text and indexed for quick access

---

**Created:** 2026-08-28  
**Purpose:** Complete mapping of 15 dashboard/API screenshots to documentation  
**Status:** ✅ All images mapped and documentation updated
