#!/bin/bash
# Kubernetes Deployment Helper Script
# Automates deployment and management of Cats vs Dogs classifier in Rancher Desktop

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

NAMESPACE="cats-dogs-classification"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_DIR"
    docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .
    
    if [ $? -eq 0 ]; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Create host directories
create_host_dirs() {
    log_info "Creating host directories for persistent volumes..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo mkdir -p /mnt/data/cats-dogs/data/processed/{train,val,test}
        sudo mkdir -p /mnt/data/cats-dogs/models/best_model
        sudo mkdir -p /mnt/data/cats-dogs/mlflow
        sudo chmod -R 777 /mnt/data/cats-dogs
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        mkdir -p /mnt/data/cats-dogs/data/processed/{train,val,test}
        mkdir -p /mnt/data/cats-dogs/models/best_model
        mkdir -p /mnt/data/cats-dogs/mlflow
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Windows with WSL
        log_warning "Windows detected. Using WSL2 paths"
        echo "Please run in WSL2 terminal:"
        echo "  mkdir -p /mnt/data/cats-dogs/data/processed/{train,val,test}"
        echo "  mkdir -p /mnt/data/cats-dogs/models/best_model"
        echo "  mkdir -p /mnt/data/cats-dogs/mlflow"
        echo "  chmod -R 777 /mnt/data/cats-dogs"
    fi
    
    log_success "Host directories created"
}

# Copy data to PV
copy_data() {
    log_info "Copying dataset to persistent volume..."
    
    if [ ! -d "$PROJECT_DIR/data/processed" ]; then
        log_error "Prepared data not found at $PROJECT_DIR/data/processed"
        log_info "Run: python src/scripts/prepare_data.py first"
        return 1
    fi
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo cp -r "$PROJECT_DIR/data/processed"/* /mnt/data/cats-dogs/data/processed/ 2>/dev/null || true
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        cp -r "$PROJECT_DIR/data/processed"/* /mnt/data/cats-dogs/data/processed/ 2>/dev/null || true
    else
        log_warning "Please manually copy data to host PV path"
    fi
    
    log_success "Data copied to persistent volume"
}

# Deploy namespace and resources
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    cd "$PROJECT_DIR"
    
    log_info "Applying namespace..."
    kubectl apply -f k8s/00-namespace.yaml
    
    log_info "Creating persistent volumes..."
    kubectl apply -f k8s/01-persistent-volumes.yaml
    
    log_info "Waiting for PVCs to bind..."
    sleep 5
    kubectl get pvc -n $NAMESPACE
    
    log_info "Applying training configuration..."
    kubectl apply -f k8s/02-training-configmap.yaml
    
    log_info "Deploying training cronjob..."
    kubectl apply -f k8s/03-training-cronjob.yaml
    
    log_info "Deploying MLFlow..."
    kubectl apply -f k8s/04-mlflow-deployment.yaml
    
    log_info "Deploying inference service..."
    kubectl apply -f k8s/05-inference-deployment.yaml
    
    log_info "Setting up ingress..."
    kubectl apply -f k8s/06-ingress.yaml
    
    log_success "Kubernetes deployment completed"
}

# Wait for deployments
wait_for_deployments() {
    log_info "Waiting for deployments to be ready..."
    
    kubectl wait --for=condition=available --timeout=300s \
        deployment/mlflow -n $NAMESPACE 2>/dev/null || true
    
    kubectl wait --for=condition=available --timeout=300s \
        deployment/inference-service -n $NAMESPACE 2>/dev/null || true
    
    log_success "Deployments are ready"
}

# Check deployment status
check_status() {
    log_info "Checking deployment status..."
    
    echo ""
    echo "=== Namespace ==="
    kubectl get namespace $NAMESPACE
    
    echo ""
    echo "=== Pods ==="
    kubectl get pods -n $NAMESPACE
    
    echo ""
    echo "=== Services ==="
    kubectl get svc -n $NAMESPACE
    
    echo ""
    echo "=== PersistentVolumeClaims ==="
    kubectl get pvc -n $NAMESPACE
    
    echo ""
    echo "=== CronJobs ==="
    kubectl get cronjob -n $NAMESPACE
    
    echo ""
    echo "=== Ingress ==="
    kubectl get ingress -n $NAMESPACE
}

# Run training job
run_training() {
    log_info "Creating training job..."
    
    kubectl create job \
        --from=cronjob/cats-dogs-training \
        manual-training-run-$(date +%s) \
        -n $NAMESPACE
    
    log_success "Training job created"
    log_info "Monitor with: kubectl logs -f job/<job-name> -n $NAMESPACE"
}

# Port forward for local access
port_forward() {
    log_info "Setting up port forwarding..."
    log_info "MLFlow: http://localhost:5000"
    log_info "Inference API: http://localhost:8000"
    
    kubectl port-forward -n $NAMESPACE svc/mlflow 5000:5000 &
    MLFLOW_PID=$!
    
    kubectl port-forward -n $NAMESPACE svc/inference-service 8000:8000 &
    INFERENCE_PID=$!
    
    log_success "Port forwarding active (PIDs: $MLFLOW_PID, $INFERENCE_PID)"
    
    trap "kill $MLFLOW_PID $INFERENCE_PID 2>/dev/null" EXIT
    
    wait
}

# View logs
view_logs() {
    local pod_type=$1
    
    if [ -z "$pod_type" ]; then
        log_error "Usage: $0 logs <mlflow|inference|training>"
        exit 1
    fi
    
    case $pod_type in
        mlflow)
            kubectl logs -f deployment/mlflow -n $NAMESPACE
            ;;
        inference)
            kubectl logs -f deployment/inference-service -n $NAMESPACE
            ;;
        training)
            local job=$(kubectl get jobs -n $NAMESPACE --sort-by=.metadata.creationTimestamp | tail -1 | awk '{print $1}')
            if [ -z "$job" ]; then
                log_error "No training jobs found"
                return 1
            fi
            kubectl logs -f job/$job -n $NAMESPACE
            ;;
        *)
            log_error "Unknown pod type: $pod_type"
            exit 1
            ;;
    esac
}

# Cleanup
cleanup() {
    log_warning "Cleaning up deployment..."
    
    read -p "Delete namespace $NAMESPACE and all resources? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace $NAMESPACE
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Display usage
usage() {
    cat << EOF
Usage: $0 <command> [options]

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
  $0 deploy              # Full deployment
  $0 train               # Run training job
  $0 logs mlflow         # View MLFlow logs
  $0 port-forward        # Enable local access

EOF
}

# Main
main() {
    if [ $# -eq 0 ]; then
        usage
        exit 1
    fi
    
    local command=$1
    shift
    
    check_prerequisites
    
    case $command in
        build)
            build_image
            ;;
        setup)
            create_host_dirs
            ;;
        copy-data)
            copy_data
            ;;
        deploy)
            build_image
            create_host_dirs
            copy_data
            deploy_kubernetes
            wait_for_deployments
            check_status
            ;;
        status)
            check_status
            ;;
        train)
            run_training
            ;;
        logs)
            view_logs "$@"
            ;;
        port-forward)
            port_forward
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
