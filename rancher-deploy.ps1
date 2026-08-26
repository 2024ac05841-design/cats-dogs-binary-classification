#!/usr/bin/env powershell
# Quick Rancher Desktop Deployment & Training Script
# Run all steps with: .\rancher-deploy.ps1

param(
    [switch]$SkipBuild = $false,
    [switch]$SkipData = $false
)

$ErrorActionPreference = "Stop"
$NAMESPACE = "cats-dogs-classification"
$PROJECT_DIR = "C:\Users\z0045n5j\Documents\tech\s3\MLO\ASSGN\2"

# Colors
function Write-Info { param([string]$msg) Write-Host "[INFO] $msg" -ForegroundColor Blue }
function Write-Success { param([string]$msg) Write-Host "[SUCCESS] $msg" -ForegroundColor Green }
function Write-Error { param([string]$msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }
function Write-Warning { param([string]$msg) Write-Host "[WARNING] $msg" -ForegroundColor Yellow }

try {
    # STEP 1: Build Docker Image
    if (-not $SkipBuild) {
        Write-Info "=== STEP 1: Building Docker Image ==="
        Push-Location $PROJECT_DIR
        docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .
        Pop-Location
        Write-Success "Docker image built"
    }

    # STEP 2: Create directories
    Write-Info "=== STEP 2: Creating host directories ==="
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
    Write-Success "Directories created at C:\mnt\data\cats-dogs\"

    # STEP 3: Verify dataset
    if (-not $SkipData) {
        Write-Info "=== STEP 3: Checking dataset ==="
        $dataPath = "C:\mnt\data\cats-dogs\data\processed\train"
        if ((Get-ChildItem $dataPath -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
            Write-Warning "No data found at $dataPath"
            Write-Warning "Please copy your prepared dataset to: C:\mnt\data\cats-dogs\data\processed\"
            Write-Warning "Expected structure:"
            Write-Warning "  - train/cats/ and train/dogs/"
            Write-Warning "  - val/cats/ and val/dogs/"
            Write-Warning "  - test/cats/ and test/dogs/"
            Read-Host "Press Enter to continue anyway"
        } else {
            Write-Success "Dataset found"
        }
    }

    # STEP 4: Deploy to Kubernetes
    Write-Info "=== STEP 4: Deploying to Kubernetes ==="
    Push-Location $PROJECT_DIR

    Write-Info "Creating namespace..."
    kubectl apply -f k8s/00-namespace.yaml
    Start-Sleep -Seconds 2

    Write-Info "Creating persistent volumes..."
    kubectl apply -f k8s/01-persistent-volumes.yaml
    Start-Sleep -Seconds 5

    Write-Info "Applying training configuration..."
    kubectl apply -f k8s/02-training-configmap.yaml

    Write-Info "Deploying training cronjob..."
    kubectl apply -f k8s/03-training-cronjob.yaml

    Write-Info "Deploying MLFlow server..."
    kubectl apply -f k8s/04-mlflow-deployment.yaml

    Write-Info "Deploying inference service..."
    kubectl apply -f k8s/05-inference-deployment.yaml

    Write-Info "Setting up ingress..."
    kubectl apply -f k8s/06-ingress.yaml

    Pop-Location
    Write-Success "Kubernetes deployment completed"

    # STEP 5: Verify deployment
    Write-Info "=== STEP 5: Verifying deployment ==="
    Write-Info "Checking PersistentVolumeClaims..."
    kubectl get pvc -n $NAMESPACE

    Write-Info "Checking Deployments..."
    kubectl get deployment -n $NAMESPACE

    Write-Info "Checking Pods..."
    kubectl get pods -n $NAMESPACE

    Write-Info "Checking CronJob..."
    kubectl get cronjob -n $NAMESPACE

    # STEP 6: Trigger training job
    Write-Info "=== STEP 6: Triggering Training Job ==="
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $jobName = "manual-training-run-$timestamp"
    
    Write-Info "Creating training job: $jobName"
    kubectl create job $jobName --from=cronjob/cats-dogs-training -n $NAMESPACE

    Start-Sleep -Seconds 3
    kubectl get jobs -n $NAMESPACE
    Write-Success "Training job created!"

    # STEP 7: Monitor training
    Write-Info "=== STEP 7: Monitoring Training ==="
    Write-Info "Training is starting. This will take ~25-40 minutes."
    Write-Info "You can monitor the progress with:"
    Write-Info ""
    Write-Info "  kubectl logs -f job/$jobName -n $NAMESPACE"
    Write-Info ""
    Write-Info "Press Enter to start monitoring..."
    Read-Host

    Write-Info "Starting real-time log monitoring..."
    kubectl logs -f job/$jobName -n $NAMESPACE

} catch {
    Write-Error "Deployment failed: $_"
}

# STEP 8: After training completes
Write-Info ""
Write-Info "=== STEP 8: Training Complete ==="
Write-Info ""
Write-Info "Next steps:"
Write-Info "1. View models in MLFlow:"
Write-Info "   kubectl port-forward svc/mlflow 5000:5000 -n $NAMESPACE"
Write-Info "   Open: http://localhost:5000"
Write-Info ""
Write-Info "2. Test inference API:"
Write-Info "   kubectl port-forward svc/inference-service 8000:8000 -n $NAMESPACE"
Write-Info "   Open: http://localhost:8000/docs"
Write-Info ""
Write-Info "3. Check saved models:"
Write-Info "   Get-ChildItem 'C:\mnt\data\cats-dogs\models\best_model\'"
Write-Info ""
Write-Info "4. View model comparison:"
Write-Info "   Get-Content 'C:\mnt\data\cats-dogs\models\best_model\model_comparison.json' | ConvertFrom-Json"
Write-Info ""
Write-Success "Deployment and training complete!"
