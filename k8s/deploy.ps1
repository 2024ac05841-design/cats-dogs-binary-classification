# Kubernetes Deployment Helper Script for Windows
# Automates deployment and management of Cats vs Dogs classifier in Rancher Desktop

param(
    [Parameter(Position=0)]
    [string]$Command = "",
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"

# Configuration
$NAMESPACE = "cats-dogs-classification"
$PROJECT_DIR = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# Color output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check prerequisites
function Check-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    $missing = @()
    
    if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
        $missing += "kubectl"
    }
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        $missing += "docker"
    }
    
    if ($missing.Count -gt 0) {
        Write-Error "Missing required tools: $($missing -join ', ')"
        exit 1
    }
    
    Write-Success "Prerequisites check passed"
}

# Build Docker image
function Build-Image {
    Write-Info "Building Docker image..."
    
    Push-Location $PROJECT_DIR
    try {
        docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker image built successfully"
        } else {
            Write-Error "Failed to build Docker image"
            exit 1
        }
    } finally {
        Pop-Location
    }
}

# Create host directories
function Create-HostDirs {
    Write-Info "Creating host directories for persistent volumes..."
    
    $dirs = @(
        "C:\mnt\data\cats-dogs\data\processed\train",
        "C:\mnt\data\cats-dogs\data\processed\val",
        "C:\mnt\data\cats-dogs\data\processed\test",
        "C:\mnt\data\cats-dogs\models\best_model",
        "C:\mnt\data\cats-dogs\mlflow"
    )
    
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Success "Host directories created at C:\mnt\data\cats-dogs\"
    Write-Info "Mount these paths in Rancher Desktop settings for persistent storage"
}

# Copy data to PV
function Copy-Data {
    Write-Info "Copying dataset to persistent volume..."
    
    $source = "$PROJECT_DIR\data\processed"
    $dest = "C:\mnt\data\cats-dogs\data\processed"
    
    if (-not (Test-Path $source)) {
        Write-Error "Prepared data not found at $source"
        Write-Info "Run: python src\scripts\prepare_data.py first"
        return $false
    }
    
    try {
        Copy-Item "$source\*" $dest -Recurse -Force -ErrorAction SilentlyContinue
        Write-Success "Data copied to persistent volume"
        return $true
    } catch {
        Write-Warning "Could not copy data automatically"
        Write-Info "Please manually copy $source to $dest"
        return $false
    }
}

# Deploy to Kubernetes
function Deploy-Kubernetes {
    Write-Info "Deploying to Kubernetes..."
    
    Push-Location $PROJECT_DIR
    try {
        Write-Info "Applying namespace..."
        kubectl apply -f k8s/00-namespace.yaml
        
        Write-Info "Creating persistent volumes..."
        kubectl apply -f k8s/01-persistent-volumes.yaml
        
        Start-Sleep -Seconds 5
        Write-Info "Verifying PVCs..."
        kubectl get pvc -n $NAMESPACE
        
        Write-Info "Applying training configuration..."
        kubectl apply -f k8s/02-training-configmap.yaml
        
        Write-Info "Deploying training cronjob..."
        kubectl apply -f k8s/03-training-cronjob.yaml
        
        Write-Info "Deploying MLFlow..."
        kubectl apply -f k8s/04-mlflow-deployment.yaml
        
        Write-Info "Deploying inference service..."
        kubectl apply -f k8s/05-inference-deployment.yaml
        
        Write-Info "Setting up ingress..."
        kubectl apply -f k8s/06-ingress.yaml
        
        Write-Success "Kubernetes deployment completed"
    } finally {
        Pop-Location
    }
}

# Wait for deployments
function Wait-ForDeployments {
    Write-Info "Waiting for deployments to be ready..."
    
    try {
        kubectl wait --for=condition=available --timeout=300s `
            deployment/mlflow -n $NAMESPACE -ErrorAction SilentlyContinue
    } catch {
        # Ignore errors, just continue
    }
    
    try {
        kubectl wait --for=condition=available --timeout=300s `
            deployment/inference-service -n $NAMESPACE -ErrorAction SilentlyContinue
    } catch {
        # Ignore errors, just continue
    }
    
    Write-Success "Deployments are ready"
}

# Check deployment status
function Check-Status {
    Write-Info "Checking deployment status..."
    
    Write-Host ""
    Write-Host "=== Namespace ===" -ForegroundColor Cyan
    kubectl get namespace $NAMESPACE
    
    Write-Host ""
    Write-Host "=== Pods ===" -ForegroundColor Cyan
    kubectl get pods -n $NAMESPACE
    
    Write-Host ""
    Write-Host "=== Services ===" -ForegroundColor Cyan
    kubectl get svc -n $NAMESPACE
    
    Write-Host ""
    Write-Host "=== PersistentVolumeClaims ===" -ForegroundColor Cyan
    kubectl get pvc -n $NAMESPACE
    
    Write-Host ""
    Write-Host "=== CronJobs ===" -ForegroundColor Cyan
    kubectl get cronjob -n $NAMESPACE
    
    Write-Host ""
    Write-Host "=== Ingress ===" -ForegroundColor Cyan
    kubectl get ingress -n $NAMESPACE
}

# Run training job
function Run-Training {
    Write-Info "Creating training job..."
    
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $jobName = "manual-training-run-$timestamp"
    
    kubectl create job --from=cronjob/cats-dogs-training $jobName -n $NAMESPACE
    
    Write-Success "Training job created: $jobName"
    Write-Info "Monitor with: kubectl logs -f job/$jobName -n $NAMESPACE"
}

# Port forward for local access
function Port-Forward {
    Write-Info "Setting up port forwarding..."
    Write-Info "MLFlow: http://localhost:5000"
    Write-Info "Inference API: http://localhost:8000"
    Write-Info "Press Ctrl+C to stop port forwarding"
    
    # Run both in separate processes
    $job1 = Start-Process powershell -ArgumentList "kubectl port-forward -n $NAMESPACE svc/mlflow 5000:5000" -PassThru
    $job2 = Start-Process powershell -ArgumentList "kubectl port-forward -n $NAMESPACE svc/inference-service 8000:8000" -PassThru
    
    Write-Success "Port forwarding active"
    
    try {
        $job1.WaitForExit()
    } finally {
        Stop-Process -Id $job1.Id -ErrorAction SilentlyContinue
        Stop-Process -Id $job2.Id -ErrorAction SilentlyContinue
    }
}

# View logs
function View-Logs {
    param([string]$PodType)
    
    if ([string]::IsNullOrEmpty($PodType)) {
        Write-Error "Usage: deploy.ps1 logs <mlflow|inference|training>"
        exit 1
    }
    
    switch ($PodType.ToLower()) {
        "mlflow" {
            kubectl logs -f deployment/mlflow -n $NAMESPACE
        }
        "inference" {
            kubectl logs -f deployment/inference-service -n $NAMESPACE
        }
        "training" {
            $jobs = kubectl get jobs -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}'
            if ($jobs) {
                $job = ($jobs -split ' ' | Select-Object -Last 1)
                kubectl logs -f job/$job -n $NAMESPACE
            } else {
                Write-Error "No training jobs found"
            }
        }
        default {
            Write-Error "Unknown pod type: $PodType"
            exit 1
        }
    }
}

# Cleanup
function Cleanup {
    Write-Warning "This will delete namespace $NAMESPACE and all resources"
    $confirm = Read-Host "Continue? (yes/no)"
    
    if ($confirm -eq "yes") {
        kubectl delete namespace $NAMESPACE
        Write-Success "Cleanup completed"
    } else {
        Write-Info "Cleanup cancelled"
    }
}

# Display usage
function Show-Usage {
    @"
Usage: .\deploy.ps1 <command> [options]

Commands:
  build              Build Docker image
  setup              Create host directories for persistent volumes
  copy-data          Copy prepared dataset to persistent volume
  deploy             Full deployment (build, setup, deploy k8s)
  status             Check deployment status
  train              Trigger training job
  logs <type>        View logs (mlflow|inference|training)
  port-forward       Setup port forwarding for local access
  cleanup            Delete namespace and all resources
  help               Show this message

Examples:
  .\deploy.ps1 deploy              # Full deployment
  .\deploy.ps1 train               # Run training job
  .\deploy.ps1 logs mlflow         # View MLFlow logs
  .\deploy.ps1 port-forward        # Enable local access

"@ | Write-Host
}

# Main
try {
    Check-Prerequisites
    
    switch ($Command.ToLower()) {
        "build" {
            Build-Image
        }
        "setup" {
            Create-HostDirs
        }
        "copy-data" {
            Copy-Data
        }
        "deploy" {
            Build-Image
            Create-HostDirs
            Copy-Data
            Deploy-Kubernetes
            Wait-ForDeployments
            Check-Status
        }
        "status" {
            Check-Status
        }
        "train" {
            Run-Training
        }
        "logs" {
            View-Logs $Arguments[0]
        }
        "port-forward" {
            Port-Forward
        }
        "cleanup" {
            Cleanup
        }
        "help" {
            Show-Usage
        }
        "" {
            Show-Usage
        }
        default {
            Write-Error "Unknown command: $Command"
            Show-Usage
            exit 1
        }
    }
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
