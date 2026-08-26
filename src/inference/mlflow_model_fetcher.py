"""MLflow Model Fetcher with Version Tracking

Fetches the best model from MLFlow on startup and caches it locally.
Only downloads if model version changes, avoiding unnecessary transfers.
Falls back to local model if MLFlow is unavailable.
"""

import logging
import os
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLFlowModelFetcher:
    """Fetch and cache best model from MLFlow"""

    def __init__(
        self,
        mlflow_uri: str = "http://mlflow:5000",
        experiment_name: str = "cats-dogs-k8s",
        cache_dir: str = "/app-models",
        version_file: str = "model_version.json",
    ):
        """
        Initialize MLFlow model fetcher

        Args:
            mlflow_uri: MLFlow tracking server URI
            experiment_name: Name of the experiment containing trained models
            cache_dir: Directory to cache the downloaded model
            version_file: JSON file to track model version/checksum
        """
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.cache_dir = Path(cache_dir)
        self.version_file = self.cache_dir / version_file
        self.model_path = self.cache_dir / "model.pkl"
        self.client = None
        self.experiment_id = None

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MLFlow client if available
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(mlflow_uri)
                self.client = MlflowClient(tracking_uri=mlflow_uri)
                logger.info(f"MLFlow client initialized with URI: {mlflow_uri}")
            except Exception as e:
                logger.warning(f"Failed to initialize MLFlow client: {e}")
                self.client = None

    def _get_experiment_id(self) -> Optional[str]:
        """Get experiment ID by name"""
        if not self.client:
            return None

        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Found experiment '{self.experiment_name}' with ID: {self.experiment_id}")
                return self.experiment_id
            else:
                logger.warning(f"Experiment '{self.experiment_name}' not found in MLFlow")
                return None
        except Exception as e:
            logger.warning(f"Error getting experiment ID: {e}")
            return None

    def _find_best_model(self) -> Optional[Tuple[str, str, float]]:
        """
        Find best model run in experiment

        Returns:
            Tuple of (run_id, model_name, val_accuracy) or None if not found
        """
        if not self.client or not self.experiment_id:
            return None

        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["metrics.val_acc DESC"],
                max_results=100,
            )

            if not runs:
                logger.warning("No runs found in experiment")
                return None

            # Find run with best model tag
            for run in runs:
                tags = run.data.tags or {}
                if tags.get("type") == "best_model":
                    run_id = run.info.run_id
                    model_name = tags.get("model_name", "unknown")
                    val_acc = run.data.metrics.get("val_acc", 0.0)
                    logger.info(
                        f"Found best model: {model_name} (run: {run_id}, val_acc: {val_acc:.4f})"
                    )
                    return run_id, model_name, val_acc

            # If no tagged best model, use highest val_acc
            if runs:
                best_run = runs[0]
                run_id = best_run.info.run_id
                model_name = best_run.params.get("model_name", "unknown")
                val_acc = best_run.data.metrics.get("val_acc", 0.0)
                logger.info(
                    f"Using highest val_acc model: {model_name} "
                    f"(run: {run_id}, val_acc: {val_acc:.4f})"
                )
                return run_id, model_name, val_acc

            return None

        except Exception as e:
            logger.warning(f"Error finding best model: {e}")
            return None

    def _get_model_artifact(
        self, run_id: str, artifact_name: str = "best_model.pkl"
    ) -> Optional[str]:
        """
        Download model artifact from MLFlow

        Args:
            run_id: MLFlow run ID
            artifact_name: Name of the model artifact file

        Returns:
            Path to downloaded model or None
        """
        if not self.client:
            return None

        try:
            # Download artifact to cache directory
            local_model_path = self.client.download_artifacts(
                run_id=run_id,
                path=artifact_name,
                dst_path=str(self.cache_dir),
            )

            # MLFlow creates a subdirectory structure, flatten it
            downloaded_file = Path(local_model_path)
            if downloaded_file.is_dir():
                # Find the actual model file
                pkl_files = list(downloaded_file.glob("*.pkl"))
                if pkl_files:
                    model_file = pkl_files[0]
                    shutil.copy(model_file, self.model_path)
                    logger.info(f"Model artifact downloaded to {self.model_path}")
                    return str(self.model_path)
            else:
                # Direct file
                shutil.copy(downloaded_file, self.model_path)
                logger.info(f"Model artifact downloaded to {self.model_path}")
                return str(self.model_path)

            return None

        except Exception as e:
            logger.warning(f"Error downloading model artifact: {e}")
            return None

    def _load_cached_version(self) -> Optional[dict]:
        """Load cached model version information"""
        if not self.version_file.exists():
            return None

        try:
            with open(self.version_file, "r") as f:
                version_info = json.load(f)
            logger.debug(f"Loaded cached version info: {version_info}")
            return version_info
        except Exception as e:
            logger.warning(f"Error loading version file: {e}")
            return None

    def _save_version(self, run_id: str, model_name: str, val_acc: float):
        """Save model version information"""
        version_info = {
            "run_id": run_id,
            "model_name": model_name,
            "val_accuracy": val_acc,
            "fetched_at": datetime.now().isoformat(),
            "mlflow_uri": self.mlflow_uri,
            "experiment_name": self.experiment_name,
        }

        try:
            with open(self.version_file, "w") as f:
                json.dump(version_info, f, indent=2)
            logger.info(f"Saved version info: {version_info}")
        except Exception as e:
            logger.warning(f"Error saving version file: {e}")

    def _version_changed(
        self, new_run_id: str, new_val_acc: float, cached_version: dict
    ) -> bool:
        """Check if model version has changed"""
        if not cached_version:
            logger.info("No cached version, need to fetch new model")
            return True

        cached_run_id = cached_version.get("run_id")
        cached_val_acc = cached_version.get("val_accuracy", 0.0)

        # Version changed if:
        # 1. Different run_id (new training)
        # 2. Same run but different accuracy (data changed)
        if cached_run_id != new_run_id:
            logger.info(f"Model version changed: {cached_run_id} → {new_run_id}")
            return True

        if abs(cached_val_acc - new_val_acc) > 0.0001:  # Small tolerance for float comparison
            logger.info(f"Model accuracy changed: {cached_val_acc:.6f} → {new_val_acc:.6f}")
            return True

        logger.info("Model version unchanged, using cached version")
        return False

    def _has_local_model(self) -> bool:
        """Check if local model file exists"""
        return self.model_path.exists()

    def fetch_best_model(self) -> Tuple[bool, str, dict]:
        """
        Fetch best model from MLFlow with caching

        Returns:
            Tuple of (success: bool, model_path: str, info: dict)
            - success: True if model loaded successfully
            - model_path: Path to the model file (cached or local fallback)
            - info: Dictionary with model information
        """
        info = {
            "source": "unknown",
            "model_name": None,
            "val_accuracy": None,
            "run_id": None,
            "cached": False,
            "message": "",
        }

        # Step 1: Check if MLFlow is available
        if not MLFLOW_AVAILABLE:
            logger.warning("MLFlow not available, using local model")
            info["source"] = "local_fallback"
            info["message"] = "MLFlow not installed"

            if self._has_local_model():
                info["cached"] = True
                return True, str(self.model_path), info

            return False, "", info

        # Step 2: Try to connect to MLFlow
        if not self.client:
            logger.warning("MLFlow client not initialized, using local model")
            info["source"] = "local_fallback"
            info["message"] = "Failed to connect to MLFlow"

            if self._has_local_model():
                info["cached"] = True
                return True, str(self.model_path), info

            return False, "", info

        # Step 3: Get experiment ID
        if not self._get_experiment_id():
            logger.warning("Experiment not found, using local model")
            info["source"] = "local_fallback"
            info["message"] = f"Experiment '{self.experiment_name}' not found"

            if self._has_local_model():
                info["cached"] = True
                return True, str(self.model_path), info

            return False, "", info

        # Step 4: Find best model
        best_model_info = self._find_best_model()
        if not best_model_info:
            logger.warning("No best model found in experiment, using local model")
            info["source"] = "local_fallback"
            info["message"] = "No runs found in experiment"

            if self._has_local_model():
                info["cached"] = True
                return True, str(self.model_path), info

            return False, "", info

        run_id, model_name, val_acc = best_model_info
        info["run_id"] = run_id
        info["model_name"] = model_name
        info["val_accuracy"] = val_acc

        # Step 5: Check if version changed
        cached_version = self._load_cached_version()
        if not self._version_changed(run_id, val_acc, cached_version):
            # Use cached model
            info["source"] = "mlflow_cached"
            info["cached"] = True
            info["message"] = f"Using cached {model_name} model"
            logger.info(f"Using cached model: {model_name}")
            return True, str(self.model_path), info

        # Step 6: Download new model
        logger.info(f"Downloading best model: {model_name} from MLFlow...")
        model_path = self._get_model_artifact(run_id)

        if model_path:
            self._save_version(run_id, model_name, val_acc)
            info["source"] = "mlflow_fresh"
            info["cached"] = False
            info["message"] = f"Downloaded new {model_name} model from MLFlow"
            logger.info(f"Successfully downloaded model: {model_name}")
            return True, model_path, info
        else:
            # Download failed, try local model
            logger.warning("Failed to download model, using local model")
            info["source"] = "local_fallback"
            info["message"] = "Failed to download from MLFlow, using cached version"

            if self._has_local_model():
                info["cached"] = True
                return True, str(self.model_path), info

            return False, "", info


def load_model_with_mlflow(
    mlflow_uri: str = "http://mlflow:5000",
    experiment_name: str = "cats-dogs-k8s",
    cache_dir: str = "/app-models",
) -> Tuple[bool, str, dict]:
    """
    Convenience function to load model with MLFlow

    Returns:
        Tuple of (success, model_path, info)
    """
    fetcher = MLFlowModelFetcher(
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
        cache_dir=cache_dir,
    )
    return fetcher.fetch_best_model()
