"""FastAPI inference service for Cats vs Dogs classifier"""

import logging
import os
import json
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
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
    model_config = {"protected_namespaces": ()}
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
    model_config = {"protected_namespaces": ()}
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
    model_config = {"protected_namespaces": ()}
    total: int
    limit: int
    logs: list[LogEntry]


class StatsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
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
            has_model_prefix = any(
                k.startswith("model.") for k in model.state_dict().keys()
            )
            state_has_prefix = any(k.startswith("model.") for k in state_dict.keys())

            if has_model_prefix and not state_has_prefix:
                new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
            elif not has_model_prefix and state_has_prefix:
                new_state_dict = {
                    k.replace("model.", "", 1): v for k, v in state_dict.items()
                }
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
        set_model_info(
            model_name="simple_cnn",
            version="fallback",
            stage="None",
            val_accuracy=50.0,
            source="untrained",
        )


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
    swagger_ui_parameters={
        "persistAuthorization": True,
        "displayRequestDuration": True,
    },
    lifespan=lifespan,
)

# Enable CORS FIRST (before all other middleware) for all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight for 1 hour
)


@app.middleware("http")
async def request_telemetry_middleware(request: Request, call_next):
    """Global request/response logging and latency tracking middleware"""
    start_time = time.perf_counter()
    req_id = str(int(time.time() * 1000))
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    method = request.method

    is_probe = path in ["/health", "/metrics"]
    if not is_probe:
        logger.info(f"➡️ [REQ-{req_id}] Incoming {method} {path} from {client_ip}")

    try:
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        status_code = response.status_code

        if not is_probe:
            logger.info(
                f"⬅️ [REQ-{req_id}] {status_code} {method} {path} completed in {latency_ms:.1f}ms"
            )

        return response
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(
            f"❌ [REQ-{req_id}] 500 Server Error on {method} {path} ({latency_ms:.1f}ms): {e}",
            exc_info=True,
        )
        raise e


# Exception handler for HTTP exceptions (ensures CORS headers are included)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper CORS headers"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        },
    )


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive API testing UI"""
    return RedirectResponse(url="/docs")


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def interactive_ui():
    """Self-contained interactive API testing console (zero CDN dependencies)"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐾 Cats vs Dogs Classification - API Testing Console</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-color: #c9d1d9;
            --text-bright: #ffffff;
            --accent-color: #58a6ff;
            --success-color: #2ea043;
            --warning-color: #d29922;
            --error-color: #f85149;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            padding: 24px;
            line-height: 1.5;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
        }
        h1 { color: var(--text-bright); font-size: 24px; }
        .links a {
            color: var(--accent-color);
            text-decoration: none;
            margin-left: 16px;
            font-size: 14px;
        }
        .links a:hover { text-decoration: underline; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        @media(max-width: 850px) { .grid { grid-template-columns: 1fr; } }
        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .card h2 { color: var(--text-bright); font-size: 18px; margin-bottom: 14px; }
        .btn-group { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px; }
        button {
            background-color: #21262d;
            color: var(--accent-color);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        button:hover {
            background-color: #30363d;
            border-color: #8b949e;
        }
        button.primary {
            background-color: var(--success-color);
            color: white;
            border-color: rgba(240, 246, 252, 0.1);
        }
        button.primary:hover { background-color: #2c974b; }
        input[type="file"] {
            display: block;
            margin-bottom: 15px;
            color: var(--text-color);
        }
        .preview {
            max-width: 100%;
            max-height: 200px;
            border-radius: 6px;
            margin-bottom: 15px;
            display: none;
            border: 1px solid var(--border-color);
        }
        pre {
            background-color: #090c10;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 14px;
            color: #7ee787;
            font-size: 13px;
            overflow-x: auto;
            max-height: 400px;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .badge-success { background: rgba(46,160,67,0.15); color: #3fb950; border: 1px solid rgba(46,160,67,0.4); }
        .badge-error { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid rgba(248,81,73,0.4); }
        .status-line { font-size: 13px; margin-bottom: 8px; color: var(--text-color); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🐾 Cats vs Dogs Classification API</h1>
                <p style="font-size: 14px; color: #8b949e;">Interactive Test Console & Live Inference Engine</p>
            </div>
            <div class="links">
                <a href="/docs" target="_blank">📖 Swagger UI</a>
                <a href="/redoc" target="_blank">📚 ReDoc</a>
                <a href="/openapi.json" target="_blank">📜 OpenAPI JSON</a>
            </div>
        </header>

        <div class="grid">
            <!-- Left Column: GET Endpoints -->
            <div>
                <div class="card">
                    <h2>🔍 Quick GET Endpoints</h2>
                    <div class="btn-group">
                        <button onclick="callGet('/health')">GET /health</button>
                        <button onclick="callGet('/info')">GET /info</button>
                        <button onclick="callGet('/stats')">GET /stats</button>
                        <button onclick="callGet('/logs?limit=10')">GET /logs</button>
                        <button onclick="callGet('/metrics')">GET /metrics</button>
                    </div>
                    <div class="status-line" id="getStatus">Select an endpoint above to execute.</div>
                    <pre id="getResult">{\n  "message": "Click any button above to test the endpoint."\n}</pre>
                </div>
            </div>

            <!-- Right Column: Prediction Upload -->
            <div>
                <div class="card">
                    <h2>🚀 Try Image Prediction (/predict)</h2>
                    <input type="file" id="imageInput" accept="image/jpeg, image/png" onchange="previewImage()">
                    <img id="imagePreview" class="preview" alt="Image preview">
                    <button class="primary" onclick="uploadAndPredict()">Execute Prediction</button>
                    <div class="status-line" id="predStatus" style="margin-top: 15px;">Upload a cat or dog image to test.</div>
                    <pre id="predResult">{\n  "message": "Awaiting image upload..."\n}</pre>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function callGet(path) {
            const statusEl = document.getElementById('getStatus');
            const resultEl = document.getElementById('getResult');
            statusEl.innerHTML = `Fetching <code>${path}</code>...`;
            const t0 = performance.now();
            try {
                const res = await fetch(path);
                const dt = (performance.now() - t0).toFixed(1);
                const isJson = res.headers.get('content-type')?.includes('application/json');
                const body = isJson ? await res.json() : await res.text();
                statusEl.innerHTML = `<span class="badge ${res.ok ? 'badge-success' : 'badge-error'}">${res.status} ${res.statusText}</span> (${dt} ms)`;
                resultEl.innerText = isJson ? JSON.stringify(body, null, 2) : body;
            } catch (err) {
                statusEl.innerHTML = `<span class="badge badge-error">Failed</span> ${err.message}`;
                resultEl.innerText = err.stack || err.message;
            }
        }

        function previewImage() {
            const input = document.getElementById('imageInput');
            const preview = document.getElementById('imagePreview');
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = e => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(input.files[0]);
            }
        }

        async function uploadAndPredict() {
            const input = document.getElementById('imageInput');
            const statusEl = document.getElementById('predStatus');
            const resultEl = document.getElementById('predResult');
            
            if (!input.files || !input.files[0]) {
                statusEl.innerHTML = '<span class="badge badge-error">Error</span> Please choose an image first.';
                return;
            }

            const formData = new FormData();
            formData.append('file', input.files[0]);

            statusEl.innerHTML = 'Executing prediction on model...';
            const t0 = performance.now();
            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const dt = (performance.now() - t0).toFixed(1);
                const data = await res.json();
                statusEl.innerHTML = `<span class="badge ${res.ok ? 'badge-success' : 'badge-error'}">${res.status} ${res.statusText}</span> Result: <strong>${data.class_name ? data.class_name.toUpperCase() : 'N/A'}</strong> (${(data.confidence * 100).toFixed(2)}%) in ${dt} ms`;
                resultEl.innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                statusEl.innerHTML = `<span class="badge badge-error">Failed</span> ${err.message}`;
                resultEl.innerText = err.stack || err.message;
            }
        }
    </script>
</body>
</html>"""


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service Health & Info"],
    summary="Health Check Endpoint",
)
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
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPG or PNG) to classify"),
    request: Request = None,
):
    """
    Predict class of uploaded image

    Args:
        file: Image file (jpg, png)

    Returns:
        Prediction response with class, confidence, and probabilities
    """
    client_ip = request.client.host if request and request.client else "unknown"
    req_id = log_request(method="POST", path="/predict", client_ip=client_ip)
    logger.info(
        f"➡️ [REQ-{req_id}] Incoming POST /predict from {client_ip} | File: {file.filename}"
    )

    if model is None:
        record_error(error_type="model_not_loaded", endpoint="/predict")
        record_request(method="POST", endpoint="/predict", status_code=503)
        log_error("Model not loaded", error_type="ModelUnavailable", request_id=req_id)
        logger.error(f"❌ [REQ-{req_id}] 503 Service Unavailable: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()
    temp_path = None
    try:
        # Save uploaded file temporarily and close the file handle
        temp_path = f"temp_{req_id}_{file.filename}"
        file_content = await file.read()
        await file.close()  # ← KEY FIX: Close file handle immediately after reading
        
        with open(temp_path, "wb") as f:
            f.write(file_content)

        # Preprocess and predict
        image_tensor = preprocess_image(temp_path)

        # Time model execution
        model_start = time.perf_counter()
        class_name, confidence, probs = predict(model, image_tensor, device)
        model_time = time.perf_counter() - model_start

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

        return PredictionResponse(
            class_name=class_name,
            confidence=float(confidence),
            probabilities=probs,
            message="Prediction successful",
        )
    finally:
        # Ensure temp file is cleaned up in all cases
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up temp file {temp_path}: {cleanup_err}")

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
async def get_logs(limit: int = 50, request: Request = None):
    """Retrieve recent structured inference logs"""
    record_request(method="GET", endpoint="/logs", status_code=200)
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
async def stats(request: Request = None):
    """Retrieve real-time telemetry and statistics"""
    record_request(method="GET", endpoint="/stats", status_code=200)
    return get_metrics()


@app.get(
    "/info",
    response_model=InfoResponse,
    tags=["Service Health & Info"],
    summary="Model Architecture & Service Metadata",
    description="Provides detailed information regarding active model weights, MLFlow registry version, device (CPU/GPU), and available endpoints.",
)
async def info(request: Request = None):
    """Get information about the model and service"""
    record_request(method="GET", endpoint="/info", status_code=200)
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
