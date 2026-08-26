#!/usr/bin/env powershell
# Rancher Desktop Deployment & Training Script with Monitoring
# Run: .\rancher-deploy.ps1
# Builds separate images, copies data, deploys monitoring, and triggers training

param(
    [switch]$SkipBuild = $false,
    [switch]$SkipData = $false
)

$ErrorActionPreference = "Stop"
$NAMESPACE = "cats-dogs-classification"
$PROJECT_DIR = "C:\Users\z0045n5j\Documents\tech\s3\MLO\ASSGN\2"
$LOCAL_DATA_DIR = "$PROJECT_DIR\data\processed"
$PV_DATA_DIR = "C:\mnt\data\cats-dogs\data\processed"
$PV_MODELS_DIR = "C:\mnt\data\cats-dogs\models\best_model"
$PV_MLFLOW_DIR = "C:\mnt\data\cats-dogs\mlflow"

# Color functions
function Write-Info { param([string]$msg) Write-Host "[INFO] $msg" -ForegroundColor Blue }
function Write-Success { param([string]$msg) Write-Host "[SUCCESS] $msg" -ForegroundColor Green }
function Write-ErrorMsg { param([string]$msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }
function Write-Warning { param([string]$msg) Write-Host "[WARNING] $msg" -ForegroundColor Yellow }
function Write-Header { param([string]$msg) Write-Host "`n========== $msg ==========" -ForegroundColor Cyan }

try {
    Write-Header "Cats vs Dogs - Full Deployment with Monitoring"
    
    # STEP 1: Build Docker Images
    if (-not $SkipBuild) {
        Write-Header "STEP 1: Building Docker Images"
        Push-Location $PROJECT_DIR
        
        Write-Info "Building training image (cats-dogs-trainer:latest)..."
        docker build -f docker/Dockerfile.training -t cats-dogs-trainer:latest .
        if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to build training image" }
        Write-Success "✓ Training image built successfully"
        
        Write-Info "Building inference image (cats-dogs-inference:latest)..."
        docker build -f docker/Dockerfile.inference -t cats-dogs-inference:latest .
        if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to build inference image" }
        Write-Success "✓ Inference image built successfully"
        
        Pop-Location
    } else {
        Write-Warning "Skipping Docker image build"
    }

    # STEP 2: Create PersistentVolume Directories
    Write-Header "STEP 2: Creating PersistentVolume Directories"
    $dirs = @(
        $PV_DATA_DIR,
        "$PV_DATA_DIR\train",
        "$PV_DATA_DIR\val",
        "$PV_DATA_DIR\test",
        $PV_MODELS_DIR,
        $PV_MLFLOW_DIR
    )
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Info "✓ Created: $dir"
        } else {
            Write-Info "  Exists: $dir"
        }
    }
    Write-Success "✓ All directories ready at C:\mnt\data\cats-dogs\"

    # STEP 3: Copy Training Data
    if (-not $SkipData) {
        Write-Header "STEP 3: Preparing and Copying Training Data"
        
        if (Test-Path $LOCAL_DATA_DIR) {
            $dataCount = @(Get-ChildItem $LOCAL_DATA_DIR -Recurse -File -ErrorAction SilentlyContinue).Count
            if ($dataCount -eq 0) {
                Write-Warning "⚠ No data found at $LOCAL_DATA_DIR"
                Write-Warning "Expected structure:"
                Write-Warning "  $LOCAL_DATA_DIR\train\{cats,dogs}\"
                Write-Warning "  $LOCAL_DATA_DIR\val\{cats,dogs}\"
                Write-Warning "  $LOCAL_DATA_DIR\test\{cats,dogs}\"
                Read-Host "Press Enter after data is ready"
            }
        } else {
            Write-Warning "⚠ Data directory not found"
            Write-Warning "Create or prepare with: python src/scripts/prepare_data.py"
            Read-Host "Press Enter after data is ready"
        }
        
        if (Test-Path $LOCAL_DATA_DIR) {
            Write-Info "Copying training data to PersistentVolume..."
            $filesToCopy = @(Get-ChildItem $LOCAL_DATA_DIR -Recurse -File -ErrorAction SilentlyContinue)
            $fileCount = $filesToCopy.Count
            
            if ($fileCount -gt 0) {
                Write-Info "Found $fileCount files to copy..."
                Copy-Item "$LOCAL_DATA_DIR\*" -Destination $PV_DATA_DIR -Recurse -Force
                
                $copiedFiles = @(Get-ChildItem $PV_DATA_DIR -Recurse -File -ErrorAction SilentlyContinue)
                $copiedCount = $copiedFiles.Count
                
                if ($copiedCount -eq $fileCount) {
                    Write-Success "✓ Successfully copied all $copiedCount files"
                } else {
                    Write-Warning "⚠ Copied $copiedCount of $fileCount files"
                }
            } else {
                Write-Warning "⚠ No files found to copy"
            }
        }
    }

    # STEP 4: Deploy to Kubernetes with Explicit Namespace
    Write-Header "STEP 4: Deploying to Kubernetes (Namespace: $NAMESPACE)"
    Push-Location $PROJECT_DIR
    
    Write-Info "1. Creating namespace: $NAMESPACE..."
    kubectl apply -f k8s/00-namespace.yaml
    if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to create namespace" }
    Start-Sleep -Seconds 2
    Write-Success "✓ Namespace created"

    Write-Info "2. Creating PersistentVolumes in namespace: $NAMESPACE..."
    kubectl apply -f k8s/01-persistent-volumes.yaml -n $NAMESPACE
    if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to create PersistentVolumes" }
    Start-Sleep -Seconds 3
    Write-Success "✓ PersistentVolumes created"

    Write-Info "3. Applying training configuration in namespace: $NAMESPACE..."
    kubectl apply -f k8s/02-training-configmap.yaml -n $NAMESPACE
    if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to apply training config" }
    Write-Success "✓ Training configuration applied"

    Write-Info "4. Deploying training CronJob (cats-dogs-trainer image) in namespace: $NAMESPACE..."
    kubectl apply -f k8s/03-training-cronjob.yaml -n $NAMESPACE
    if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to deploy training cronjob" }
    Write-Success "✓ Training CronJob deployed"

    Write-Info "5. Deploying MLFlow server in namespace: $NAMESPACE..."
    kubectl apply -f k8s/04-mlflow-deployment.yaml -n $NAMESPACE
    if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to deploy MLFlow" }
    Write-Success "✓ MLFlow server deployed"

    Write-Info "6. Deploying inference service (cats-dogs-inference image) in namespace: $NAMESPACE..."
    kubectl apply -f k8s/05-inference-deployment.yaml -n $NAMESPACE
    if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to deploy inference service" }
    Write-Success "✓ Inference service deployed"

    Write-Info "7. Setting up ingress in namespace: $NAMESPACE..."
    kubectl apply -f k8s/06-ingress.yaml -n $NAMESPACE
    if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to setup ingress" }
    Write-Success "✓ Ingress configured"

    Write-Info "8. Deploying Prometheus monitoring in namespace: $NAMESPACE..."
    kubectl apply -f monitoring/01-prometheus.yaml -n $NAMESPACE
    if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to deploy Prometheus" }
    Write-Success "✓ Prometheus deployed"

    Write-Info "9. Deploying Grafana dashboards in namespace: $NAMESPACE..."
    kubectl apply -f monitoring/02-grafana.yaml -n $NAMESPACE
    if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to deploy Grafana" }
    Write-Success "✓ Grafana deployed"

    Pop-Location
    Write-Success "✓ All Kubernetes resources deployed to namespace: $NAMESPACE"

    # STEP 5: Verify Deployment
    Write-Header "STEP 5: Verifying Deployment in Namespace: $NAMESPACE"
    
    Write-Info "Checking namespaces..."
    kubectl get ns | Select-String $NAMESPACE | Write-Host -ForegroundColor Green
    
    Write-Info ""
    Write-Info "Checking PersistentVolumeClaims in $NAMESPACE..."
    kubectl get pvc -n $NAMESPACE --no-headers
    
    Write-Info ""
    Write-Info "Checking Deployments in $NAMESPACE..."
    kubectl get deployment -n $NAMESPACE --no-headers
    
    Write-Info ""
    Write-Info "Checking Pods in $NAMESPACE..."
    kubectl get pods -n $NAMESPACE --no-headers
    
    Write-Info ""
    Write-Info "Checking CronJobs in $NAMESPACE..."
    kubectl get cronjob -n $NAMESPACE --no-headers
    
    Write-Info ""
    Write-Info "Waiting for pods to be ready (max 60 seconds)..."
    Start-Sleep -Seconds 5

    # STEP 6: Trigger Training Job
    Write-Header "STEP 6: Triggering Training Job"
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $jobName = "manual-training-run-$timestamp"
    
    Write-Info "Creating training job: $jobName in namespace: $NAMESPACE"
    kubectl create job $jobName --from=cronjob/cats-dogs-training -n $NAMESPACE
    if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to create training job" }
    
    Start-Sleep -Seconds 2
    kubectl get jobs -n $NAMESPACE --no-headers
    Write-Success "✓ Training job created and will run with cats-dogs-trainer image"

    # STEP 7: Monitor Training
    Write-Header "STEP 7: Real-Time Training Monitoring"
    Write-Info "Waiting for pod to start (may take 10-30 seconds)..."
    Start-Sleep -Seconds 10
    
    Write-Info "Starting real-time log monitoring of training job..."
    Write-Info "Training will take approximately 25-40 minutes"
    Write-Info "Press Ctrl+C to stop monitoring (logs will continue in background)"
    Write-Info ""
    
    kubectl logs -f job/$jobName -n $NAMESPACE

} catch {
    Write-ErrorMsg "Deployment failed: $_"
}

# STEP 8: After Training Completes
Write-Header "STEP 8: Training Complete - Next Steps"

Write-Info ""
Write-Info "1️⃣  View Models in MLFlow:"
Write-Info "   kubectl port-forward svc/mlflow 5000:5000 -n $NAMESPACE"
Write-Info "   Open: http://localhost:5000"

Write-Info ""
Write-Info "2️⃣  View Grafana Dashboards:"
Write-Info "   kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE"
Write-Info "   Open: http://localhost:3000"
Write-Info "   Login: admin / admin"

Write-Info ""
Write-Info "3️⃣  View Prometheus Metrics:"
Write-Info "   kubectl port-forward svc/prometheus 9090:9090 -n $NAMESPACE"
Write-Info "   Open: http://localhost:9090"

Write-Info ""
Write-Info "4️⃣  Test Inference API (uses cats-dogs-inference image):"
Write-Info "   kubectl port-forward svc/inference-service 8000:8000 -n $NAMESPACE"
Write-Info "   Open: http://localhost:8000/docs"

Write-Info ""
Write-Info "5️⃣  Check Training Metrics:"
Write-Info "   Get-Content '$PV_MODELS_DIR\model_comparison.json' | ConvertFrom-Json"

Write-Info ""
Write-Info "6️⃣  View Training Logs:"
Write-Info "   Get-ChildItem '$PV_MODELS_DIR'"

Write-Info ""
Write-Info "7️⃣  Check Docker Images Built:"
Write-Info "   docker images | Select-String 'cats-dogs'"

Write-Success "✓ Deployment complete with monitoring!"
Write-Success "✓ Training using cats-dogs-trainer image"
Write-Success "✓ Inference using cats-dogs-inference image"
Write-Success "✓ Monitoring available via Prometheus & Grafana"
Write-Info ""
Write-Info "All operations performed in namespace: $NAMESPACE"
