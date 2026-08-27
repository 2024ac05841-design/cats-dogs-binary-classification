"""FastAPI inference service for Cats vs Dogs classifier"""

import logging
import os
import json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn as nn

from ..models.cnn_model import create_model
from .model_utils import predict, preprocess_image, CLASS_NAMES
from .mlflow_model_fetcher import load_model_with_mlflow

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global model and model info
model = None
device = None
model_info = {}


class PredictionResponse(BaseModel):
    """Response schema for predictions"""

    class_name: str
    confidence: float
    probabilities: dict
    message: str = "Prediction successful"


class HealthResponse(BaseModel):
    """Response schema for health check"""

    status: str
    message: str


def load_model_on_startup():
    """Load model on startup, fetching from MLFlow Model Registry with volume caching"""
    global model, device, model_info

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Get MLFlow configuration from environment
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    registered_model_name = os.getenv("REGISTERED_MODEL_NAME", "cats-dogs-best-model")
    experiment_name = os.getenv("EXPERIMENT_NAME", "cats-dogs-k8s")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    cache_dir = os.getenv("MODEL_PATH", "/app-models/model.pkl").rsplit("/", 1)[0]

    logger.info(
        f"Attempting to fetch best model '{registered_model_name}' (stage: {model_stage}) "
        f"from MLFlow: {mlflow_uri}"
    )

    # Fetch model from MLFlow registry or volume cache
    success, model_path, info = load_model_with_mlflow(
        mlflow_uri=mlflow_uri,
        registered_model_name=registered_model_name,
        experiment_name=experiment_name,
        model_stage=model_stage,
        cache_dir=cache_dir,
    )

    model_info = info

    if not success or not os.path.exists(model_path):
        logger.error(f"Failed to fetch model from MLFlow, creating untrained model")
        model_info["source"] = "untrained_fallback"
        model_info["message"] = "Using default untrained model"
        model = create_model(model_name="simple_cnn", device=device)
        return

    # Load the model architecture and weights
    try:
        raw_name = str(model_info.get("model_name", "resnet18")).lower()
        if "resnet" in raw_name:
            model_name = "resnet18"
        elif "mobile" in raw_name:
            model_name = "resnet18"  # fallback or create resnet architecture
        elif "logistic" in raw_name:
            model_name = "logistic_regression"
        else:
            model_name = "simple_cnn"

        logger.info(f"Instantiating {model_name} architecture...")
        model = create_model(model_name=model_name, device=device)

        # Load weights
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()

        logger.info(
            f"✅ Model loaded successfully from {model_info.get('source')}: {model_name} "
            f"(val_acc: {model_info.get('val_accuracy', 'N/A')}, version: {model_info.get('version', 'N/A')}) "
            f"- {model_info.get('message', '')}"
        )

    except Exception as e:
        logger.error(f"Error loading model weights: {e}", exc_info=True)
        logger.warning("Creating untrained model as fallback")
        model_info["source"] = "untrained_fallback"
        model_info["message"] = f"Failed to load weights: {str(e)}"
        model = create_model(model_name="simple_cnn", device=device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting up inference service...")
    load_model_on_startup()
    yield
    # Shutdown
    logger.info("Shutting down inference service...")


# Create FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="REST API for binary image classification",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        if model is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "message": "Model not loaded"},
            )

        return {
            "status": "healthy",
            "message": "Service is running and model is loaded",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": f"Error: {str(e)}"},
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict class of uploaded image

    Args:
        file: Image file (jpg, png)

    Returns:
        Prediction response with class, confidence, and probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Preprocess and predict
        image_tensor = preprocess_image(temp_path)
        class_name, confidence, probs = predict(model, image_tensor, device)

        # Clean up
        os.remove(temp_path)

        logger.info(f"Prediction completed: {class_name} ({confidence:.4f})")

        return {
            "class_name": class_name,
            "confidence": float(confidence),
            "probabilities": probs,
            "message": "Prediction successful",
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/info")
async def info():
    """Get information about the model and service"""
    return {
        "service": "Cats vs Dogs Classifier",
        "version": "1.0.0",
        "device": device,
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
        "model_info": model_info,
        "endpoints": {"health": "/health", "predict": "/predict", "info": "/info"},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
