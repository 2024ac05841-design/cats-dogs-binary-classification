"""Metrics tracking and Prometheus exporter for the Cats vs Dogs Inference Service"""

import time
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any, Optional
from threading import Lock

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)
METRICS_FILE = Path("logs/metrics.jsonl")
metrics_lock = Lock()

if PROMETHEUS_AVAILABLE:
    # 1. Request counters
    INFERENCE_REQUESTS_TOTAL = Counter(
        "inference_requests_total",
        "Total incoming HTTP requests to inference service",
        ["method", "endpoint", "status_code"],
    )

    # 2. Prediction counters by class & model
    INFERENCE_PREDICTIONS_TOTAL = Counter(
        "inference_predictions_total",
        "Total inference predictions performed",
        ["predicted_class", "model_name"],
    )

    # 3. Inference errors
    INFERENCE_ERRORS_TOTAL = Counter(
        "inference_errors_total",
        "Total inference error occurrences",
        ["error_type", "endpoint"],
    )

    # 4. Latency histograms
    INFERENCE_LATENCY_SECONDS = Histogram(
        "inference_latency_seconds",
        "Total end-to-end request latency in seconds",
        ["endpoint"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )

    INFERENCE_MODEL_TIME_SECONDS = Histogram(
        "inference_model_execution_seconds",
        "Time spent strictly executing model forward pass in seconds",
        ["model_name"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0),
    )

    # 5. Prediction confidence distribution
    INFERENCE_CONFIDENCE_SCORE = Histogram(
        "inference_confidence_score",
        "Distribution of prediction confidence scores",
        ["predicted_class"],
        buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0),
    )

    # 6. Model Metadata & Accuracy Gauges
    MODEL_VAL_ACCURACY = Gauge(
        "model_validation_accuracy_percent",
        "Validation accuracy percentage of the currently active model",
        ["model_name", "version", "stage"],
    )

    MODEL_INFO = Gauge(
        "model_info",
        "Information about the active loaded model",
        ["model_name", "version", "stage", "run_id", "source"],
    )

    INFERENCE_AVG_LATENCY_MS = Gauge(
        "inference_avg_latency_ms",
        "Moving average latency of predictions in milliseconds",
    )

    INFERENCE_LATEST_LATENCY_MS = Gauge(
        "inference_latest_latency_ms",
        "Most recent prediction latency in milliseconds",
    )

    INFERENCE_MODEL_EXECUTION_MS = Gauge(
        "inference_model_execution_ms",
        "Most recent model forward pass execution time in milliseconds",
        ["model_name"],
    )

    INFERENCE_P50_LATENCY_MS = Gauge(
        "inference_p50_latency_ms",
        "Median (p50) prediction latency in milliseconds",
    )

    INFERENCE_P90_LATENCY_MS = Gauge(
        "inference_p90_latency_ms",
        "90th percentile (p90) prediction latency in milliseconds",
    )

    INFERENCE_P95_LATENCY_MS = Gauge(
        "inference_p95_latency_ms",
        "95th percentile (p95) prediction latency in milliseconds",
    )

    INFERENCE_P99_LATENCY_MS = Gauge(
        "inference_p99_latency_ms",
        "99th percentile (p99) prediction latency in milliseconds",
    )

    INFERENCE_CONFIDENCE_TIERS_TOTAL = Counter(
        "inference_confidence_tiers_total",
        "Predictions grouped into confidence tiers",
        ["tier"],
    )

    INFERENCE_AVG_CONFIDENCE_PERCENT = Gauge(
        "inference_avg_confidence_percent",
        "Average prediction confidence percentage",
    )

    INFERENCE_LATEST_CONFIDENCE_PERCENT = Gauge(
        "inference_latest_confidence_percent",
        "Latest prediction confidence percentage",
    )


class MetricsCollector:
    """In-memory collector and stats accumulator"""

    def __init__(self):
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self.cat_count = 0
        self.dog_count = 0
        self.total_latency_ms = 0.0
        self.total_confidence = 0.0
        self.latency_history = []
        self.start_time = datetime.utcnow()
        self.recent_logs = []
        self.max_recent_logs = 100
        self.active_model_info = {
            "model_name": "resnet18",
            "version": "1",
            "stage": "Production",
            "val_accuracy": 99.6,
            "run_id": "",
            "source": "mlflow",
        }

    def record_request(self, method: str = "POST", endpoint: str = "/predict", status_code: int = 200):
        self.request_count += 1
        if PROMETHEUS_AVAILABLE:
            try:
                INFERENCE_REQUESTS_TOTAL.labels(
                    method=method, endpoint=endpoint, status_code=str(status_code)
                ).inc()
            except Exception:
                pass

    def record_prediction(
        self,
        latency_ms: float = 0.0,
        success: bool = True,
        predicted_class: str = "cat",
        confidence: float = 0.95,
        model_name: str = "resnet18",
        model_time_s: Optional[float] = None,
    ):
        if success:
            self.prediction_count += 1
            self.total_latency_ms += latency_ms
            self.total_confidence += confidence
            self.latency_history.append(latency_ms)
            if len(self.latency_history) > 1000:
                self.latency_history.pop(0)

            if predicted_class.lower() == "cat":
                self.cat_count += 1
            elif predicted_class.lower() == "dog":
                self.dog_count += 1

            avg_lat = self.total_latency_ms / max(self.prediction_count, 1)
            avg_conf_pct = (self.total_confidence / max(self.prediction_count, 1)) * 100.0
            latest_conf_pct = confidence * 100.0

            if confidence >= 0.90:
                tier = "90% - 100% (High Confidence)"
            elif confidence >= 0.70:
                tier = "70% - 90% (Medium Confidence)"
            else:
                tier = "< 70% (Low Confidence)"

            # Calculate percentiles
            sorted_latencies = sorted(self.latency_history)
            n = len(sorted_latencies)
            p50 = sorted_latencies[int(n * 0.50)] if n > 0 else latency_ms
            p90 = sorted_latencies[min(int(n * 0.90), n - 1)] if n > 0 else latency_ms
            p95 = sorted_latencies[min(int(n * 0.95), n - 1)] if n > 0 else latency_ms
            p99 = sorted_latencies[min(int(n * 0.99), n - 1)] if n > 0 else latency_ms

            model_time_ms = (model_time_s * 1000.0) if model_time_s is not None else latency_ms

            if PROMETHEUS_AVAILABLE:
                try:
                    INFERENCE_PREDICTIONS_TOTAL.labels(
                        predicted_class=predicted_class, model_name=model_name
                    ).inc()
                    INFERENCE_LATENCY_SECONDS.labels(endpoint="/predict").observe(latency_ms / 1000.0)
                    INFERENCE_CONFIDENCE_SCORE.labels(predicted_class=predicted_class).observe(confidence)
                    INFERENCE_AVG_LATENCY_MS.set(avg_lat)
                    INFERENCE_LATEST_LATENCY_MS.set(latency_ms)
                    INFERENCE_P50_LATENCY_MS.set(p50)
                    INFERENCE_P90_LATENCY_MS.set(p90)
                    INFERENCE_P95_LATENCY_MS.set(p95)
                    INFERENCE_P99_LATENCY_MS.set(p99)
                    INFERENCE_MODEL_EXECUTION_MS.labels(model_name=model_name).set(model_time_ms)
                    INFERENCE_CONFIDENCE_TIERS_TOTAL.labels(tier=tier).inc()
                    INFERENCE_AVG_CONFIDENCE_PERCENT.set(avg_conf_pct)
                    INFERENCE_LATEST_CONFIDENCE_PERCENT.set(latest_conf_pct)

                    if model_time_s is not None:
                        INFERENCE_MODEL_TIME_SECONDS.labels(model_name=model_name).observe(model_time_s)
                except Exception:
                    pass
        else:
            self.error_count += 1
            if PROMETHEUS_AVAILABLE:
                try:
                    INFERENCE_ERRORS_TOTAL.labels(error_type="prediction_failure", endpoint="/predict").inc()
                except Exception:
                    pass

    def record_error(self, error_type: str = "error", endpoint: str = "/predict"):
        self.error_count += 1
        if PROMETHEUS_AVAILABLE:
            try:
                INFERENCE_ERRORS_TOTAL.labels(error_type=error_type, endpoint=endpoint).inc()
            except Exception:
                pass

    def set_model_info(
        self,
        model_name: str,
        version: str = "1",
        stage: str = "Production",
        val_accuracy: Optional[float] = None,
        run_id: str = "",
        source: str = "mlflow",
    ):
        acc = float(val_accuracy) if val_accuracy is not None else 99.6
        self.active_model_info = {
            "model_name": model_name,
            "version": str(version),
            "stage": stage,
            "val_accuracy": acc,
            "run_id": run_id,
            "source": source,
        }

        if PROMETHEUS_AVAILABLE:
            try:
                MODEL_VAL_ACCURACY.labels(
                    model_name=model_name, version=str(version), stage=stage
                ).set(acc)
                MODEL_INFO.labels(
                    model_name=model_name,
                    version=str(version),
                    stage=stage,
                    run_id=run_id,
                    source=source,
                ).set(1.0)
            except Exception:
                pass

    def add_log_entry(self, entry: dict):
        with metrics_lock:
            self.recent_logs.append(entry)
            if len(self.recent_logs) > self.max_recent_logs:
                self.recent_logs.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        avg_latency = (
            self.total_latency_ms / self.prediction_count
            if self.prediction_count > 0
            else 0.0
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.request_count,
            "successful_predictions": self.prediction_count,
            "errors": self.error_count,
            "class_distribution": {
                "cats": self.cat_count,
                "dogs": self.dog_count,
            },
            "average_latency_ms": round(avg_latency, 2),
            "success_rate": round(
                (self.prediction_count / self.request_count * 100)
                if self.request_count > 0
                else 100.0,
                2,
            ),
            "model_info": self.active_model_info,
            "recent_logs_count": len(self.recent_logs),
        }

    def reset(self):
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self.cat_count = 0
        self.dog_count = 0
        self.total_latency_ms = 0.0
        self.start_time = datetime.utcnow()
        self.recent_logs = []


# Global collector
_collector = MetricsCollector()


def record_request(method: str = "POST", endpoint: str = "/predict", status_code: int = 200):
    _collector.record_request(method, endpoint, status_code)


def record_prediction(
    latency_ms: float = 0.0,
    success: bool = True,
    predicted_class: str = "cat",
    confidence: float = 0.95,
    model_name: str = "resnet18",
    model_time_s: Optional[float] = None,
):
    _collector.record_prediction(latency_ms, success, predicted_class, confidence, model_name, model_time_s)


def record_error(error_type: str = "error", endpoint: str = "/predict"):
    _collector.record_error(error_type, endpoint)


def set_model_info(
    model_name: str,
    version: str = "1",
    stage: str = "Production",
    val_accuracy: Optional[float] = None,
    run_id: str = "",
    source: str = "mlflow",
):
    _collector.set_model_info(model_name, version, stage, val_accuracy, run_id, source)


def add_log_entry(entry: dict):
    _collector.add_log_entry(entry)


def get_metrics() -> Dict[str, Any]:
    return _collector.get_stats()


def get_recent_logs():
    return _collector.recent_logs


def get_prometheus_metrics() -> bytes:
    """Generate Prometheus exposition format payload"""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(REGISTRY)
    else:
        stats = _collector.get_stats()
        lines = [
            f"# HELP inference_requests_total Total incoming requests",
            f"# TYPE inference_requests_total counter",
            f"inference_requests_total {stats['total_requests']}",
            f"# HELP inference_predictions_total Total predictions",
            f"# TYPE inference_predictions_total counter",
            f"inference_predictions_total {stats['successful_predictions']}",
            f"# HELP inference_errors_total Total errors",
            f"# TYPE inference_errors_total counter",
            f"inference_errors_total {stats['errors']}",
            f"# HELP inference_avg_latency_ms Average latency in ms",
            f"# TYPE inference_avg_latency_ms gauge",
            f"inference_avg_latency_ms {stats['average_latency_ms']}",
            f"# HELP model_validation_accuracy_percent Model validation accuracy",
            f"# TYPE model_validation_accuracy_percent gauge",
            f"model_validation_accuracy_percent {stats['model_info']['val_accuracy']}",
        ]
        return "\n".join(lines).encode("utf-8")


def log_metrics():
    """Log metrics to file"""
    with metrics_lock:
        stats = get_metrics()
        METRICS_FILE.parent.mkdir(exist_ok=True)
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(stats) + "\n")


def start_prediction_timer():
    """Start a timer for prediction latency"""
    return time.time()


def end_prediction_timer(start_time: float) -> float:
    """End timer and return latency in ms"""
    return (time.time() - start_time) * 1000


# Post-deployment monitoring
class PredictionDataCollector:
    """Collect predictions for post-deployment monitoring"""

    def __init__(self, collection_file: str = "logs/predictions.jsonl"):
        self.collection_file = Path(collection_file)
        self.data_lock = Lock()

    def record_prediction(
        self,
        image_path: str,
        predicted_class: str,
        confidence: float,
        true_label: str = None,
    ):
        """
        Record a prediction for later analysis

        Args:
            image_path: Path to the image
            predicted_class: Predicted class
            confidence: Prediction confidence
            true_label: True label (for post-deployment monitoring)
        """
        with self.data_lock:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "image": image_path,
                "predicted": predicted_class,
                "confidence": confidence,
                "true_label": true_label,
                "correct": (predicted_class == true_label) if true_label else None,
            }

            self.collection_file.parent.mkdir(exist_ok=True)
            with open(self.collection_file, "a") as f:
                f.write(json.dumps(record) + "\n")

    def get_recent_predictions(self, limit: int = 100) -> list:
        """Get recent predictions"""
        if not self.collection_file.exists():
            return []

        with open(self.collection_file, "r") as f:
            lines = f.readlines()

        predictions = [json.loads(line) for line in lines[-limit:]]
        return predictions

    def get_accuracy(self) -> float:
        """Calculate accuracy from collected predictions"""
        predictions = self.get_recent_predictions(limit=None)

        if not predictions:
            return None

        correct = sum(1 for p in predictions if p.get("correct") is True)
        total = len([p for p in predictions if p.get("true_label") is not None])

        return (correct / total * 100) if total > 0 else None


# Global prediction collector
_prediction_collector = PredictionDataCollector()


def record_prediction_data(
    image_path: str, predicted_class: str, confidence: float, true_label: str = None
):
    """Record prediction data"""
    _prediction_collector.record_prediction(
        image_path, predicted_class, confidence, true_label
    )
