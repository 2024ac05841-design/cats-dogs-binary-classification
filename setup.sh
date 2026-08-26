#!/bin/bash

# Setup script for MLOps project

set -e

echo "================================"
echo "MLOps Pipeline Setup"
echo "================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | grep -oP '\d+\.\d+')
if [[ $(echo "$python_version < 3.11" | bc) -eq 1 ]]; then
    echo "❌ Python 3.11 or higher is required"
    exit 1
fi
echo "✓ Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || . .venv/Scripts/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ Pip upgraded"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
    echo "✓ DVC initialized"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{raw,processed}
mkdir -p models
mkdir -p logs
mkdir -p mlruns
echo "✓ Directories created"

# Make smoke_tests.sh executable
chmod +x smoke_tests.sh

# Run tests
echo ""
echo "Running unit tests..."
pytest tests/ -v --tb=short || echo "⚠️ Some tests may have failed - check output above"

echo ""
echo "================================"
echo "✓ Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Place dataset in data/raw/ (organize as cats/ and dogs/ subdirectories)"
echo "2. Train model: python -m src.models.train"
echo "3. Run inference service: uvicorn src.inference.app:app --host 0.0.0.0 --port 8000"
echo "4. Test locally: docker-compose -f docker-compose.yml up -d"
echo "5. Run smoke tests: bash smoke_tests.sh"
echo "6. Commit and push: git add . && git commit -m 'message' && git push"
