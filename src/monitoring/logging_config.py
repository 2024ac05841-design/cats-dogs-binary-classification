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
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        if hasattr(record, 'user'):
            log_obj['user'] = record.user
        if hasattr(record, 'request_id'):
            log_obj['request_id'] = record.request_id
        if hasattr(record, 'latency'):
            log_obj['latency_ms'] = record.latency
        
        return json.dumps(log_obj)


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
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, level))
    file_formatter = JSONFormatter()
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


# Request/Response logging
request_logger = setup_logging("inference.requests", "INFO")


def log_request(method: str, path: str, client_ip: str = None):
    """Log incoming request"""
    request_logger.info(
        f"Request: {method} {path} from {client_ip}",
        extra={'request_id': id({})}
    )


def log_prediction(class_name: str, confidence: float, latency_ms: float):
    """Log prediction result"""
    request_logger.info(
        f"Prediction: {class_name} (confidence: {confidence:.4f})",
        extra={'latency': latency_ms}
    )


def log_error(error_msg: str, error_type: str = None):
    """Log error"""
    request_logger.error(
        f"Error: {error_msg} (type: {error_type})"
    )
