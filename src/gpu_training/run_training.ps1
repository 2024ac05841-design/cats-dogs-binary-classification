# PowerShell script to run GPU training locally
param (
    [int]$Epochs = 12,
    [int]$BatchSize = 64,
    [double]$LR = 0.0001,
    [string]$DataDir = "data/processed",
    [string]$OutputDir = "src/models"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "🐾 CATS VS DOGS - LOCAL GPU TRAINING PIPELINE 🐾" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Activate .venv if it exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "[1/3] Activating virtual environment (.venv)..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "[1/3] Using current Python environment..." -ForegroundColor Yellow
}

# 2. Check PyTorch CUDA availability
Write-Host "[2/3] Checking PyTorch GPU/CUDA status..." -ForegroundColor Yellow
$cudaCheck = python -c "import torch; print(torch.cuda.is_available())" 2>$null

if ($cudaCheck -ne "True") {
    Write-Host "⚠️ PyTorch with CUDA is not currently enabled." -ForegroundColor DarkYellow
    Write-Host "Installing PyTorch with CUDA support into .venv..." -ForegroundColor Yellow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-deps
} else {
    $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))"
    Write-Host "✅ PyTorch CUDA is ready! GPU: $gpuName" -ForegroundColor Green
}

# 3. Run training
Write-Host "[3/3] Starting multi-model training with GPU..." -ForegroundColor Yellow
Write-Host "  - Data Directory : $DataDir" -ForegroundColor Gray
Write-Host "  - Output Path    : $OutputDir" -ForegroundColor Gray
Write-Host "  - Epochs         : $Epochs" -ForegroundColor Gray
Write-Host "  - Batch Size     : $BatchSize" -ForegroundColor Gray
Write-Host "  - Learning Rate  : $LR" -ForegroundColor Gray
Write-Host "----------------------------------------------------------" -ForegroundColor Cyan

python -m src.gpu_training.train_gpu `
    --data-dir $DataDir `
    --output-dir $OutputDir `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --lr $LR `
    --models resnet18 simple_cnn mobilenet_v2

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host "🎉 TRAINING COMPLETE! Best models saved to $OutputDir" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Host "`n❌ Training process exited with code $LASTEXITCODE" -ForegroundColor Red
}
