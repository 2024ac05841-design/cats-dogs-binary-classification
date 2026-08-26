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

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global model
model = None
device = None


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
    """Load model on startup"""
    global model, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model_path = os.getenv("MODEL_PATH", "models/model.pkl")
    
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}, using untrained model")
        model = create_model(model_name="simple_cnn", device=device)
    else:
        try:
            model = create_model(model_name="simple_cnn", device=device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        if model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "message": "Model not loaded"
                }
            )
        
        return {
            "status": "healthy",
            "message": "Service is running and model is loaded"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": f"Error: {str(e)}"
            }
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
            "message": "Prediction successful"
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
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "info": "/info"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
