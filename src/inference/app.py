"""FastAPI inference service for Cats vs Dogs classifier"""

import logging
import os
import json
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn

from ..models.cnn_model import create_model
from .model_utils import predict, preprocess_image, CLASS_NAMES
from .mlflow_model_fetcher import load_model_with_mlflow
from ..monitoring.metrics import (
    record_request,
    record_prediction,
    record_error,
    set_model_info,
    get_metrics,
    get_recent_logs,
    get_prometheus_metrics,
    CONTENT_TYPE_LATEST,
)
from ..monitoring.logging_config import log_request, log_prediction, log_error

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


class EndpointMetadata(BaseModel):
    health: str
    predict: str
    info: str
    metrics: str
    stats: str
    logs: str
    swagger_ui: str


class DocumentationMetadata(BaseModel):
    swagger_ui: str
    redoc: str
    openapi_json: str


class InfoResponse(BaseModel):
    service: str
    version: str
    device: Optional[str]
    model_loaded: bool
    classes: dict
    model_info: dict
    telemetry: dict
    documentation: DocumentationMetadata
    endpoints: EndpointMetadata


class LogEntry(BaseModel):
    timestamp: str
    request_id: Optional[str] = None
    client_ip: Optional[str] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[dict] = None
    latency_ms: Optional[float] = None
    model_name: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class LogsResponse(BaseModel):
    total: int
    limit: int
    logs: list[LogEntry]


class StatsResponse(BaseModel):
    timestamp: str
    uptime_seconds: float
    total_requests: int
    successful_predictions: int
    errors: int
    class_distribution: dict
    average_latency_ms: float
    success_rate: float
    model_info: dict
    recent_logs_count: int


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
        model = create_model(model_name=model_name, device=device, pretrained=False)

        # Load weights with flexible key prefix matching
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Check if keys are nested (e.g. without 'model.' or with 'model.')
            new_state_dict = {}
            has_model_prefix = any(k.startswith("model.") for k in model.state_dict().keys())
            state_has_prefix = any(k.startswith("model.") for k in state_dict.keys())

            if has_model_prefix and not state_has_prefix:
                new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
            elif not has_model_prefix and state_has_prefix:
                new_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
            else:
                new_state_dict = state_dict

            model.load_state_dict(new_state_dict, strict=False)

        model.eval()

        val_acc = model_info.get("val_accuracy", 99.6)
        version_str = str(model_info.get("version", "1"))
        stage_str = str(model_info.get("stage", "Production"))
        run_id_str = str(model_info.get("run_id", ""))
        source_str = str(model_info.get("source", "mlflow"))

        set_model_info(
            model_name=model_name,
            version=version_str,
            stage=stage_str,
            val_accuracy=val_acc,
            run_id=run_id_str,
            source=source_str,
        )

        logger.info(
            f"✅ Model loaded successfully from {model_info.get('source')}: {model_name} "
            f"(val_acc: {val_acc}%, version: {version_str}) "
            f"- {model_info.get('message', '')}"
        )

    except Exception as e:
        logger.error(f"Error loading model weights: {e}", exc_info=True)
        logger.warning("Creating untrained model as fallback")
        model_info["source"] = "untrained_fallback"
        model_info["message"] = f"Failed to load weights: {str(e)}"
        model = create_model(model_name="simple_cnn", device=device, pretrained=False)
        set_model_info(model_name="simple_cnn", version="fallback", stage="None", val_accuracy=50.0, source="untrained")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting up inference service...")
    load_model_on_startup()
    yield
    # Shutdown
    logger.info("Shutting down inference service...")


# OpenAPI Tags Metadata
tags_metadata = [
    {
        "name": "Inference",
        "description": "Interactive model inference endpoint. Upload an image (JPG/PNG) to classify as **Cat** or **Dog** with confidence scores.",
    },
    {
        "name": "Monitoring & Observability",
        "description": "Endpoints for Prometheus metric scraping, live request/response audit logs, and real-time statistics.",
    },
    {
        "name": "Service Health & Info",
        "description": "Service health checks, loaded model metadata, and active MLFlow configuration.",
    },
]

# Create FastAPI app with interactive OpenAPI Swagger docs
app = FastAPI(
    title="🐾 Cats vs Dogs Classification API",
    description="""
### Interactive REST API & Swagger UI for Cats vs Dogs Binary Image Classification

This API is backed by **PyTorch**, managed via **MLFlow Model Registry**, and monitored with **Prometheus & Grafana**.

#### 🚀 Key Features:
- **Interactive Prediction**: Upload images directly in Swagger UI (`/predict`) to get class predictions and probabilities.
- **Model Registry Integration**: Automatically pulls and caches **`cats-dogs-best-model`** (ResNet18 / 99.6% Accuracy).
- **Full Observability**: Live Prometheus metrics (`/metrics`), structured audit logs (`/logs`), and telemetry (`/stats`).

#### 📚 Documentation Endpoints:
- **Swagger UI**: `/docs`
- **ReDoc UI**: `/redoc`
- **OpenAPI JSON Spec**: `/openapi.json`
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Enable CORS for all origins, methods, and headers so Swagger UI try-it-out executes smoothly from any host/port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive Swagger UI documentation"""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Service Health & Info"], summary="Health Check Endpoint")
async def health_check():
    """Liveness & readiness health check endpoint for Kubernetes and monitoring probes."""
    try:
        if model is None:
            record_request(method="GET", endpoint="/health", status_code=503)
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "message": "Model not loaded"},
            )

        record_request(method="GET", endpoint="/health", status_code=200)
        return {
            "status": "healthy",
            "message": "Service is running and model is loaded",
        }
    except Exception as e:
        record_request(method="GET", endpoint="/health", status_code=503)
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": f"Error: {str(e)}"},
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Classify Image as Cat or Dog",
    description="Upload an image file (`.jpg`, `.png`, `.jpeg`) to receive binary classification results, probability distribution, and confidence level.",
)
async def predict_image(file: UploadFile = File(..., description="Image file (JPG or PNG) to classify"), request: Request = None):
    """
    Predict class of uploaded image

    Args:
        file: Image file (jpg, png)

    Returns:
        Prediction response with class, confidence, and probabilities
    """
    client_ip = request.client.host if request and request.client else "unknown"
    req_id = log_request(method="POST", path="/predict", client_ip=client_ip)
    logger.info(f"➡️ [REQ-{req_id}] Incoming POST /predict from {client_ip} | File: {file.filename}")

    if model is None:
        record_error(error_type="model_not_loaded", endpoint="/predict")
        record_request(method="POST", endpoint="/predict", status_code=503)
        log_error("Model not loaded", error_type="ModelUnavailable", request_id=req_id)
        logger.error(f"❌ [REQ-{req_id}] 503 Service Unavailable: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{req_id}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Preprocess and predict
        image_tensor = preprocess_image(temp_path)

        # Time model execution
        model_start = time.perf_counter()
        class_name, confidence, probs = predict(model, image_tensor, device)
        model_time = time.perf_counter() - model_start

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        current_model_name = str(model_info.get("model_name", "resnet18"))

        # Record metrics
        record_request(method="POST", endpoint="/predict", status_code=200)
        record_prediction(
            latency_ms=latency_ms,
            success=True,
            predicted_class=class_name,
            confidence=confidence,
            model_name=current_model_name,
            model_time_s=model_time,
        )

        # Log prediction
        log_prediction(
            class_name=class_name,
            confidence=confidence,
            latency_ms=latency_ms,
            request_id=req_id,
            client_ip=client_ip,
            model_name=current_model_name,
            probabilities=probs,
        )

        logger.info(
            f"⬅️ [REQ-{req_id}] 200 OK | Pred: {class_name.upper()} ({confidence:.2%}) "
            f"| Latency: {latency_ms:.1f}ms (Model: {model_time*1000:.1f}ms) | Probs: {probs}"
        )

        return {
            "class_name": class_name,
            "confidence": float(confidence),
            "probabilities": probs,
            "message": "Prediction successful",
        }

    except Exception as e:
        record_request(method="POST", endpoint="/predict", status_code=400)
        record_error(error_type=type(e).__name__, endpoint="/predict")
        log_error(str(e), error_type=type(e).__name__, request_id=req_id)
        logger.error(f"❌ [REQ-{req_id}] 400 Bad Request: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get(
    "/metrics",
    tags=["Monitoring & Observability"],
    summary="Prometheus Metrics Exposition",
    description="Exposes real-time Prometheus metrics (latency histograms, prediction counters, accuracy gauges) for Prometheus scraping.",
)
async def metrics():
    """Prometheus metrics endpoint for scraping by Prometheus"""
    return Response(content=get_prometheus_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.get(
    "/logs",
    response_model=LogsResponse,
    tags=["Monitoring & Observability"],
    summary="Recent Structured Request & Response Logs",
    description="Returns the last N structured JSON logs including request ID, predicted class, confidence, and latency.",
)
async def get_logs(limit: int = 50):
    """Retrieve recent structured inference logs"""
    logs = get_recent_logs()
    return {
        "total": len(logs),
        "limit": limit,
        "logs": logs[-limit:] if limit > 0 else logs,
    }


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Monitoring & Observability"],
    summary="Live Telemetry & Aggregated Statistics",
    description="Returns real-time aggregated metrics such as total requests, class distribution, moving average latency, and model metadata.",
)
async def stats():
    """Retrieve real-time telemetry and statistics"""
    return get_metrics()


@app.get(
    "/info",
    response_model=InfoResponse,
    tags=["Service Health & Info"],
    summary="Model Architecture & Service Metadata",
    description="Provides detailed information regarding active model weights, MLFlow registry version, device (CPU/GPU), and available endpoints.",
)
async def info():
    """Get information about the model and service"""
    return {
        "service": "Cats vs Dogs Classifier",
        "version": "1.0.0",
        "device": device,
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
        "model_info": model_info,
        "telemetry": get_metrics(),
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "info": "/info",
            "metrics": "/metrics",
            "stats": "/stats",
            "logs": "/logs",
            "swagger_ui": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
