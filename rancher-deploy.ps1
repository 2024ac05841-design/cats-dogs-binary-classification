#!/usr/bin/env powershell
# Quick Rancher Desktop Deployment & Training Script
# Run all steps with: .\rancher-deploy.ps1
# Builds separate images for training and inference with automated data copying

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

# Colors
function Write-Info { param([string]$msg) Write-Host "[INFO] $msg" -ForegroundColor Blue }
function Write-Success { param([string]$msg) Write-Host "[SUCCESS] $msg" -ForegroundColor Green }
function Write-ErrorMsg { param([string]$msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }
function Write-Warning { param([string]$msg) Write-Host "[WARNING] $msg" -ForegroundColor Yellow }

try {
    # STEP 1: Build Separate Docker Images
    if (-not $SkipBuild) {
        Write-Info "=== STEP 1: Building Docker Images ==="
        Push-Location $PROJECT_DIR
        
        # Build training image
        Write-Info "Building training image (cats-dogs-trainer:latest)..."
        docker build -f docker/Dockerfile.training -t cats-dogs-trainer:latest .
        if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to build training image" }
        Write-Success "✓ Training image built"
        
        # Build inference image
        Write-Info "Building inference image (cats-dogs-inference:latest)..."
        docker build -f docker/Dockerfile.inference -t cats-dogs-inference:latest .
        if ($LASTEXITCODE -ne 0) { Write-ErrorMsg "Failed to build inference image" }
        Write-Success "✓ Inference image built"
        
        Pop-Location
        Write-Success "Both Docker images built successfully"
    }

    # STEP 2: Create directories
    Write-Info "=== STEP 2: Creating PersistentVolume directories ==="
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
            Write-Info "Created: $dir"
        } else {
            Write-Info "Already exists: $dir"
        }
    }
    Write-Success "Directories ready at C:\mnt\data\cats-dogs\"

    # STEP 3: Prepare and copy training data
    if (-not $SkipData) {
        Write-Info "=== STEP 3: Preparing and copying training data ==="
        
        # Check if local data exists
        if (Test-Path $LOCAL_DATA_DIR) {
            $dataCount = @(Get-ChildItem $LOCAL_DATA_DIR -Recurse -File -ErrorAction SilentlyContinue).Count
            if ($dataCount -eq 0) {
                Write-Warning "⚠ No data found at $LOCAL_DATA_DIR"
                Write-Warning "Please copy your prepared dataset first:"
                Write-Warning "  1. Run: python src/scripts/prepare_data.py (to prepare from raw data)"
                Write-Warning "  2. Or copy existing prepared data to: $LOCAL_DATA_DIR"
                Write-Warning "Expected structure:"
                Write-Warning "  $LOCAL_DATA_DIR\train\cats\"
                Write-Warning "  $LOCAL_DATA_DIR\train\dogs\"
                Write-Warning "  $LOCAL_DATA_DIR\val\cats\"
                Write-Warning "  $LOCAL_DATA_DIR\val\dogs\"
                Write-Warning "  $LOCAL_DATA_DIR\test\cats\"
                Write-Warning "  $LOCAL_DATA_DIR\test\dogs\"
                Read-Host "Press Enter after data is ready to continue"
            }
        } else {
            Write-Warning "⚠ Data directory not found at $LOCAL_DATA_DIR"
            Write-Warning "Please prepare data first:"
            Write-Warning "  python src/scripts/prepare_data.py"
            Read-Host "Press Enter after data is ready to continue"
        }
        
        # Copy data to PV
        if (Test-Path $LOCAL_DATA_DIR) {
            Write-Info "Copying training data to PersistentVolume..."
            Write-Info "Source: $LOCAL_DATA_DIR"
            Write-Info "Destination: $PV_DATA_DIR"
            
            # Count files for progress
            $filesToCopy = @(Get-ChildItem $LOCAL_DATA_DIR -Recurse -File -ErrorAction SilentlyContinue)
            $fileCount = $filesToCopy.Count
            
            if ($fileCount -gt 0) {
                Write-Info "Copying $fileCount files..."
                Copy-Item "$LOCAL_DATA_DIR\*" -Destination $PV_DATA_DIR -Recurse -Force
                
                # Verify copy
                $copiedFiles = @(Get-ChildItem $PV_DATA_DIR -Recurse -File -ErrorAction SilentlyContinue)
                $copiedCount = $copiedFiles.Count
                
                if ($copiedCount -eq $fileCount) {
                    Write-Success "✓ Successfully copied $copiedCount files to PersistentVolume"
                } else {
                    Write-Warning "⚠ Expected $fileCount files but got $copiedCount (possible partial copy)"
                }
            } else {
                Write-Warning "⚠ No data files found to copy"
            }
        } else {
            Write-Warning "⚠ Data directory not available - training may fail"
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

    Write-Info "Deploying training cronjob (uses cats-dogs-trainer image)..."
    kubectl apply -f k8s/03-training-cronjob.yaml

    Write-Info "Deploying MLFlow server..."
    kubectl apply -f k8s/04-mlflow-deployment.yaml

    Write-Info "Deploying inference service (uses cats-dogs-inference image)..."
    kubectl apply -f k8s/05-inference-deployment.yaml

    Write-Info "Setting up ingress..."
    kubectl apply -f k8s/06-ingress.yaml

    Pop-Location
    Write-Success "Kubernetes deployment completed"

    # STEP 5: Verify deployment
    Write-Info "=== STEP 5: Verifying deployment ==="
    Write-Info "Checking PersistentVolumeClaims..."
    kubectl get pvc -n $NAMESPACE

    Write-Info ""
    Write-Info "Checking Deployments..."
    kubectl get deployment -n $NAMESPACE

    Write-Info ""
    Write-Info "Checking Pods..."
    kubectl get pods -n $NAMESPACE

    Write-Info ""
    Write-Info "Checking CronJob..."
    kubectl get cronjob -n $NAMESPACE

    # STEP 6: Trigger training job
    Write-Info "=== STEP 6: Triggering Training Job ==="
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $jobName = "manual-training-run-$timestamp"
    
    Write-Info "Creating training job: $jobName (uses cats-dogs-trainer image)"
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
    Write-ErrorMsg "Deployment failed: $_"
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
Write-Info "2. Test inference API (uses cats-dogs-inference image):"
Write-Info "   kubectl port-forward svc/inference-service 8000:8000 -n $NAMESPACE"
Write-Info "   Open: http://localhost:8000/docs"
Write-Info ""
Write-Info "3. Check saved models:"
Write-Info "   Get-ChildItem '$PV_MODELS_DIR\'"
Write-Info ""
Write-Info "4. View model comparison:"
Write-Info "   Get-Content '$PV_MODELS_DIR\model_comparison.json' | ConvertFrom-Json"
Write-Info ""
Write-Info "5. Check Docker images:"
Write-Info "   docker images | Select-String 'cats-dogs'"
Write-Info ""
Write-Success "Deployment complete with separate training and inference images!"
Write-Info "   Get-Content 'C:\mnt\data\cats-dogs\models\best_model\model_comparison.json' | ConvertFrom-Json"
Write-Info ""
Write-Success "Deployment and training complete!"
