@echo off
REM Setup script for MLOps project on Windows

echo ================================
echo MLOps Pipeline Setup
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

echo ✓ Python found
python --version

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    echo ✓ Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo ✓ Virtual environment activated

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo ✓ Pip upgraded

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo ✓ Dependencies installed

REM Create necessary directories
echo Creating directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "mlruns" mkdir mlruns
echo ✓ Directories created

REM Run tests
echo.
echo Running unit tests...
pytest tests\ -v --tb=short
if %errorlevel% neq 0 (
    echo ⚠️ Some tests may have failed - check output above
)

echo.
echo ================================
echo ✓ Setup Complete!
echo ================================
echo.
echo Next steps:
echo 1. Place dataset in data\raw (organize as cats\ and dogs\ subdirectories)
echo 2. Train model: python -m src.models.train
echo 3. Run inference service: uvicorn src.inference.app:app --host 0.0.0.0 --port 8000
echo 4. Test locally: docker-compose -f docker-compose.yml up -d
echo 5. Run smoke tests: bash smoke_tests.sh
echo 6. Commit and push: git add . ^&^& git commit -m "message" ^&^& git push
echo.
pause
