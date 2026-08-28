"""Logging configuration for the inference service"""

import logging
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from datetime import datetime

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)


class JSONFormatter(logging.Formatter):
    """Format log messages as JSON"""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        for attr in [
            "user",
            "request_id",
            "latency_ms",
            "client_ip",
            "method",
            "path",
            "status_code",
            "class_name",
            "confidence",
            "model",
        ]:
            if hasattr(record, attr):
                log_obj[attr] = getattr(record, attr)

        return json.dumps(log_obj)


class FlushingRotatingFileHandler(RotatingFileHandler):
    """Custom file handler that flushes after every log"""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(name: str = __name__, level: str = "INFO") -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation (auto-flush)
    file_handler = FlushingRotatingFileHandler(
        "logs/app.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(getattr(logging, level))
    file_formatter = JSONFormatter()
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Ensure logs are written immediately (no buffering)
    file_handler.flush()
    logger.propagate = False  # Prevent duplicate logs

    return logger


# Request/Response logging
request_logger = setup_logging("inference.requests", "INFO")


def log_request(method: str, path: str, client_ip: str = None, request_id: str = None):
    """Log incoming request"""
    req_id = request_id or str(int(time.time() * 1000))
    extra = {
        "request_id": req_id,
        "method": method,
        "path": path,
        "client_ip": client_ip,
    }
    request_logger.info(
        f"Incoming {method} request to {path} from {client_ip}", extra=extra
    )
    return req_id


def log_prediction(
    class_name: str,
    confidence: float,
    latency_ms: float,
    request_id: str = None,
    client_ip: str = None,
    model_name: str = "resnet18",
    probabilities: dict = None,
):
    """Log prediction result and record structured telemetry"""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "client_ip": client_ip,
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "probabilities": probabilities or {},
        "latency_ms": round(latency_ms, 2),
        "model_name": model_name,
        "status": "success",
    }
    request_logger.info(
        f"Prediction: {class_name.upper()} ({confidence:.2%}) in {latency_ms:.1f}ms [Model: {model_name}]",
        extra={
            "request_id": request_id,
            "latency_ms": latency_ms,
            "class_name": class_name,
            "confidence": confidence,
            "model": model_name,
        },
    )
    try:
        from .metrics import add_log_entry

        add_log_entry(entry)
    except Exception:
        pass


def log_error(error_msg: str, error_type: str = None, request_id: str = None):
    """Log error and record structured telemetry"""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "error": error_msg,
        "error_type": error_type,
        "status": "error",
    }
    request_logger.error(
        f"Inference Error [{error_type}]: {error_msg}",
        extra={"request_id": request_id},
    )
    try:
        from .metrics import add_log_entry

        add_log_entry(entry)
    except Exception:
        pass
