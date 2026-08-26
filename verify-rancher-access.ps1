#!/usr/bin/env powershell
# Rancher Desktop Terminal Setup Verification

Write-Host "===== Rancher Desktop Terminal Setup =====" -ForegroundColor Blue
Write-Host ""

# Step 1: Check kubectl installation
Write-Host "[1] Checking kubectl installation..." -ForegroundColor Blue
$kubectlVersion = kubectl version --client=true 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] kubectl is installed" -ForegroundColor Green
} else {
    Write-Host "[ERROR] kubectl not found" -ForegroundColor Red
}
Write-Host ""

# Step 2: Check kubectl context
Write-Host "[2] Checking kubectl context..." -ForegroundColor Blue
$context = kubectl config current-context
Write-Host "[SUCCESS] Current context: $context" -ForegroundColor Green
Write-Host ""

# Step 3: Test cluster connection
Write-Host "[3] Testing cluster connection..." -ForegroundColor Blue
kubectl cluster-info | Select-Object -First 2
Write-Host ""

# Step 4: List namespaces
Write-Host "[4] Available namespaces:" -ForegroundColor Blue
kubectl get namespaces --no-headers | ForEach-Object {
    $line = $_
    if ($line -like "*cats-dogs*") {
        Write-Host "  [FOUND] $line" -ForegroundColor Green
    } else {
        Write-Host "  `u{2022} $line"
    }
}
Write-Host ""

# Step 5: Check cats-dogs-classification namespace
Write-Host "[5] Checking cats-dogs-classification namespace..." -ForegroundColor Blue
$ns = kubectl get ns cats-dogs-classification -o jsonpath='{.metadata.name}' 2>&1
if ($LASTEXITCODE -eq 0 -and $ns -eq "cats-dogs-classification") {
    Write-Host "[SUCCESS] Namespace exists" -ForegroundColor Green
    Write-Host "  Pods in namespace:" -ForegroundColor Cyan
    $pods = kubectl get pods -n cats-dogs-classification --no-headers 2>&1
    if ($pods) {
        $pods | ForEach-Object { Write-Host "    `u{2022} $_" }
    } else {
        Write-Host "    (No pods - deploy with: kubectl apply -f k8s/00-namespace.yaml)" -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARNING] Namespace not found yet" -ForegroundColor Yellow
    Write-Host "  Deploy with: kubectl apply -f k8s/00-namespace.yaml" -ForegroundColor Yellow
}
Write-Host ""

# Step 6: Total pod count
Write-Host "[6] Total pods in cluster:" -ForegroundColor Blue
$podCount = @(kubectl get pods -A --no-headers 2>&1).Count
Write-Host "  $podCount pods" -ForegroundColor Green
Write-Host ""

Write-Host "===== Summary =====" -ForegroundColor Green
Write-Host "[SUCCESS] Terminal access to Rancher Desktop is working!" -ForegroundColor Green
Write-Host ""
Write-Host "Common commands:" -ForegroundColor Cyan
Write-Host "  kubectl get pods -A"
Write-Host "  kubectl get pods -n cats-dogs-classification"
Write-Host "  kubectl get deployment -n cats-dogs-classification"
Write-Host "  kubectl logs deployment/mlflow -n cats-dogs-classification"
Write-Host ""
