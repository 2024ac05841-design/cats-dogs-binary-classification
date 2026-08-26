"""Metrics tracking for the inference service"""

import time
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any
from threading import Lock

# Metrics file
METRICS_FILE = Path("logs/metrics.jsonl")
metrics_lock = Lock()


class MetricsCollector:
    """Collect and track metrics"""
    
    def __init__(self):
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.start_time = datetime.utcnow()
    
    def record_request(self):
        """Record an incoming request"""
        self.request_count += 1
    
    def record_prediction(self, latency_ms: float, success: bool = True):
        """Record a prediction"""
        if success:
            self.prediction_count += 1
            self.total_latency += latency_ms
        else:
            self.error_count += 1
    
    def record_error(self):
        """Record an error"""
        self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        avg_latency = (
            self.total_latency / self.prediction_count 
            if self.prediction_count > 0 else 0
        )
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'successful_predictions': self.prediction_count,
            'errors': self.error_count,
            'average_latency_ms': avg_latency,
            'success_rate': (
                self.prediction_count / self.request_count * 100
                if self.request_count > 0 else 0
            )
        }
    
    def reset(self):
        """Reset all metrics"""
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.start_time = datetime.utcnow()


# Global collector
_collector = MetricsCollector()


def record_request():
    """Record an incoming request"""
    _collector.record_request()


def record_prediction(latency_ms: float, success: bool = True):
    """Record a prediction result"""
    _collector.record_prediction(latency_ms, success)


def record_error():
    """Record an error"""
    _collector.record_error()


def get_metrics() -> Dict[str, Any]:
    """Get current metrics"""
    return _collector.get_stats()


def log_metrics():
    """Log metrics to file"""
    with metrics_lock:
        stats = get_metrics()
        METRICS_FILE.parent.mkdir(exist_ok=True)
        with open(METRICS_FILE, 'a') as f:
            f.write(json.dumps(stats) + '\n')


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
    
    def record_prediction(self, image_path: str, predicted_class: str, 
                         confidence: float, true_label: str = None):
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
                'timestamp': datetime.utcnow().isoformat(),
                'image': image_path,
                'predicted': predicted_class,
                'confidence': confidence,
                'true_label': true_label,
                'correct': (predicted_class == true_label) if true_label else None
            }
            
            self.collection_file.parent.mkdir(exist_ok=True)
            with open(self.collection_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
    
    def get_recent_predictions(self, limit: int = 100) -> list:
        """Get recent predictions"""
        if not self.collection_file.exists():
            return []
        
        with open(self.collection_file, 'r') as f:
            lines = f.readlines()
        
        predictions = [json.loads(line) for line in lines[-limit:]]
        return predictions
    
    def get_accuracy(self) -> float:
        """Calculate accuracy from collected predictions"""
        predictions = self.get_recent_predictions(limit=None)
        
        if not predictions:
            return None
        
        correct = sum(1 for p in predictions if p.get('correct') is True)
        total = len([p for p in predictions if p.get('true_label') is not None])
        
        return (correct / total * 100) if total > 0 else None


# Global prediction collector
_prediction_collector = PredictionDataCollector()


def record_prediction_data(image_path: str, predicted_class: str, 
                          confidence: float, true_label: str = None):
    """Record prediction data"""
    _prediction_collector.record_prediction(
        image_path, predicted_class, confidence, true_label
    )
