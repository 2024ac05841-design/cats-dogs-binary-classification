"""Structured logging utilities for training and inference monitoring"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_structured_logging(name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup structured logging for pod containers"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)


class TrainingLogger:
    """Structured logging for training metrics"""
    
    def __init__(self, output_dir: str = "models/best_model"):
        self.metrics = []
        self.start_time = datetime.now()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_structured_logging("TrainingLogger")
    
    def log_start(self, model_name: str, epochs: int, batch_size: int, lr: float):
        """Log training start"""
        msg = {
            "event": "TRAINING_START",
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "status": "RUNNING"
        }
        self.logger.info(f"[TRAINING_START] Model: {model_name} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
        self.metrics.append(msg)
    
    def log_epoch(self, model_name: str, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float):
        """Log epoch completion"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        msg = {
            "event": "EPOCH_COMPLETE",
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "elapsed_seconds": elapsed
        }
        self.logger.info(f"[EPOCH {epoch}] {model_name} | Train: L={train_loss:.4f}/A={train_acc:.2f}% | Val: L={val_loss:.4f}/A={val_acc:.2f}%")
        self.metrics.append(msg)
    
    def log_model_complete(self, model_name: str, best_epoch: int, best_val_acc: float,
                          best_val_loss: float, best_train_acc: float, best_train_loss: float):
        """Log model training completion"""
        msg = {
            "event": "MODEL_COMPLETE",
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "best_epoch": best_epoch,
            "best_val_accuracy": float(best_val_acc),
            "best_val_loss": float(best_val_loss),
            "best_train_accuracy": float(best_train_acc),
            "best_train_loss": float(best_train_loss),
            "status": "COMPLETED"
        }
        self.logger.info(f"[MODEL_COMPLETE] {model_name} | Acc: {best_val_acc:.2f}% | Loss: {best_val_loss:.4f}")
        self.metrics.append(msg)
    
    def log_best_selected(self, model_name: str, val_acc: float, val_loss: float):
        """Log best model selection"""
        msg = {
            "event": "BEST_MODEL_SELECTED",
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "status": "SELECTED"
        }
        self.logger.info(f"[BEST_MODEL_SELECTED] Model: {model_name} | Accuracy: {val_acc:.2f}% | Loss: {val_loss:.4f}")
        self.metrics.append(msg)
    
    def save_metrics(self, filename: str = "training_metrics.json") -> str:
        """Save all metrics to JSON"""
        metrics_path = self.output_dir / filename
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"[METRICS_SAVED] Saved to {metrics_path}")
        return str(metrics_path)


class InferenceLogger:
    """Structured logging for inference metrics"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.predictions = []
        self.errors = []
        self.logger = setup_structured_logging("InferenceLogger")
    
    def log_startup(self, model_name: str, val_accuracy: Optional[float], model_source: str):
        """Log inference startup"""
        msg = {
            "event": "STARTUP",
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "val_accuracy": float(val_accuracy) if val_accuracy else None,
            "model_source": model_source,
            "status": "READY"
        }
        self.logger.info(f"[STARTUP] Model: {model_name} | Source: {model_source} | Accuracy: {val_accuracy}")
        self.predictions.append(msg)
    
    def log_prediction(self, class_name: str, confidence: float, inference_time_ms: float):
        """Log prediction"""
        msg = {
            "event": "PREDICTION",
            "timestamp": datetime.now().isoformat(),
            "class_name": class_name,
            "confidence": float(confidence),
            "inference_time_ms": float(inference_time_ms),
            "status": "SUCCESS"
        }
        self.logger.info(f"[PREDICTION] Class: {class_name} | Confidence: {confidence:.4f} | Time: {inference_time_ms:.2f}ms")
        self.predictions.append(msg)
    
    def log_error(self, error_type: str, error_msg: str):
        """Log prediction error"""
        msg = {
            "event": "PREDICTION_ERROR",
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_msg,
            "status": "FAILED"
        }
        self.logger.error(f"[ERROR] {error_type}: {error_msg}")
        self.errors.append(msg)
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        predictions = [p for p in self.predictions if p.get('event') == 'PREDICTION']
        errors = len(self.errors)
        total = len(predictions) + errors
        
        return {
            "uptime_seconds": uptime,
            "total_predictions": len(predictions),
            "total_errors": errors,
            "success_rate": (len(predictions) / total * 100) if total > 0 else 0,
            "avg_inference_time_ms": sum(p['inference_time_ms'] for p in predictions) / len(predictions) if predictions else 0
        }
