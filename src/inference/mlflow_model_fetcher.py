"""MLflow Model Fetcher with Registry and Version Tracking

Fetches the best model from MLFlow (via Model Registry or Experiment search) on startup.
Caches it in persistent volume or local path. Only downloads if the model version has changed.
Falls back safely to local packaged model if MLFlow is temporarily unavailable.
"""

import os
import json
import shutil
import logging
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
    """Fetch and cache best model from MLFlow Model Registry or Experiment Runs"""

    def __init__(
        self,
        mlflow_uri: str = "http://mlflow:5000",
        registered_model_name: str = "cats-dogs-best-model",
        experiment_name: str = "cats-dogs-k8s",
        model_stage: str = "Production",
        cache_dir: str = "/app-models",
        version_file: str = "model_version.json",
    ):
        self.mlflow_uri = mlflow_uri
        self.registered_model_name = registered_model_name
        self.experiment_name = experiment_name
        self.model_stage = model_stage
        self.cache_dir = Path(cache_dir)
        self.version_file = self.cache_dir / version_file
        self.model_path = self.cache_dir / "model.pkl"
        self.client = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(self.mlflow_uri)
                self.client = MlflowClient(tracking_uri=self.mlflow_uri)
                logger.info(f"MLFlow client connected to URI: {self.mlflow_uri}")
            except Exception as e:
                logger.warning(
                    f"Could not connect MLFlow client to {self.mlflow_uri}: {e}"
                )
                self.client = None

    def _load_cached_version(self) -> Optional[dict]:
        """Load cached model version metadata if present"""
        if not self.version_file.exists():
            return None
        try:
            with open(self.version_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read version file: {e}")
            return None

    def _save_version(
        self,
        version: str,
        run_id: str,
        model_name: str,
        val_acc: Optional[float] = None,
        source: str = "mlflow_registry",
    ):
        """Record model version metadata in the cache directory"""
        metadata = {
            "version": str(version),
            "run_id": run_id,
            "model_name": model_name,
            "val_accuracy": val_acc,
            "source": source,
            "downloaded_at": datetime.now().isoformat(),
            "mlflow_uri": self.mlflow_uri,
        }
        try:
            with open(self.version_file, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Updated cache model metadata: {metadata}")
        except Exception as e:
            logger.warning(f"Could not write version file: {e}")

    def _extract_downloaded_model(self, downloaded_path: str) -> bool:
        """Helper to copy downloaded artifact into destination model.pkl"""
        src = Path(downloaded_path)
        if src.is_dir():
            candidates = list(src.rglob("*.pkl"))
            if candidates:
                chosen = next(
                    (c for c in candidates if "best_model" in c.name), candidates[0]
                )
                shutil.copy2(chosen, self.model_path)
                logger.info(f"Extracted artifact {chosen.name} -> {self.model_path}")
                return True
        elif src.is_file():
            shutil.copy2(src, self.model_path)
            logger.info(f"Copied artifact {src.name} -> {self.model_path}")
            return True
        return False

    def _fetch_from_registry(self) -> Optional[Tuple[str, dict]]:
        """Attempt to fetch the latest/production version from Model Registry"""
        if not self.client:
            return None

        try:
            latest_versions = self.client.get_latest_versions(
                name=self.registered_model_name,
                stages=(
                    [self.model_stage] if self.model_stage else ["Production", "None"]
                ),
            )

            if not latest_versions:
                latest_versions = self.client.get_latest_versions(
                    name=self.registered_model_name
                )

            if not latest_versions:
                logger.info(
                    f"No versions found for registered model '{self.registered_model_name}'"
                )
                return None

            target_version_obj = max(latest_versions, key=lambda v: int(v.version))
            v_num = str(target_version_obj.version)
            run_id = target_version_obj.run_id
            tags = target_version_obj.tags or {}
            arch = tags.get("architecture", "resnet18")

            logger.info(
                f"Found Registered Model '{self.registered_model_name}' Version {v_num} "
                f"(Stage: {target_version_obj.current_stage}, Run ID: {run_id}, Arch: {arch})"
            )

            cached = self._load_cached_version()
            if cached and cached.get("version") == v_num and self.model_path.exists():
                logger.info(
                    f"✅ Model version {v_num} already cached in volume. Skipping download."
                )
                info = {
                    "source": "registry_cached",
                    "model_name": cached.get("model_name", arch),
                    "version": v_num,
                    "run_id": run_id,
                    "cached": True,
                    "val_accuracy": cached.get("val_accuracy"),
                    "message": f"Using cached registry model version {v_num}",
                }
                return str(self.model_path), info

            logger.info(
                f"⬇️ Downloading Registered Model '{self.registered_model_name}' v{v_num}..."
            )

            for candidate_art in ["best_model.pkl", f"best_model_{arch}.pkl", "model"]:
                try:
                    downloaded = self.client.download_artifacts(
                        run_id, candidate_art, dst_path=str(self.cache_dir)
                    )
                    if self._extract_downloaded_model(downloaded):
                        break
                except Exception as e:
                    logger.debug(f"Artifact {candidate_art} download trial: {e}")

            if not self.model_path.exists():
                downloaded = self.client.download_artifacts(
                    run_id, "", dst_path=str(self.cache_dir)
                )
                self._extract_downloaded_model(downloaded)

            if self.model_path.exists():
                val_acc = None
                try:
                    run_data = self.client.get_run(run_id).data
                    val_acc = run_data.metrics.get("val_acc") or run_data.metrics.get(
                        "val_accuracy"
                    )
                    arch = run_data.params.get("model_name", arch)
                except Exception:
                    pass

                self._save_version(
                    version=v_num, run_id=run_id, model_name=arch, val_acc=val_acc
                )
                info = {
                    "source": "registry_fresh",
                    "model_name": arch,
                    "version": v_num,
                    "run_id": run_id,
                    "cached": False,
                    "val_accuracy": val_acc,
                    "message": f"Successfully downloaded registry model version {v_num}",
                }
                return str(self.model_path), info

        except Exception as e:
            logger.warning(f"Failed to fetch from model registry: {e}")

        return None

    def _fetch_from_experiment_runs(self) -> Optional[Tuple[str, dict]]:
        """Fallback to searching runs in the experiment"""
        if not self.client:
            return None

        try:
            exp = self.client.get_experiment_by_name(self.experiment_name)
            if not exp:
                return None

            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.val_acc DESC", "metrics.val_accuracy DESC"],
                max_results=50,
            )
            if not runs:
                return None

            best_run = runs[0]
            for r in runs:
                if (r.data.tags or {}).get("type") == "best_model":
                    best_run = r
                    break

            run_id = best_run.info.run_id
            model_name = best_run.params.get("model_name", "resnet18")
            val_acc = best_run.data.metrics.get("val_acc") or best_run.data.metrics.get(
                "val_accuracy", 0.0
            )

            cached = self._load_cached_version()
            if cached and cached.get("run_id") == run_id and self.model_path.exists():
                logger.info(
                    f"✅ Experiment run {run_id} already cached. Skipping download."
                )
                return str(self.model_path), {
                    "source": "experiment_cached",
                    "model_name": cached.get("model_name", model_name),
                    "run_id": run_id,
                    "cached": True,
                    "val_accuracy": val_acc,
                }

            logger.info(
                f"⬇️ Downloading best model from run {run_id} ({model_name})..."
            )
            for art in ["best_model.pkl", f"best_model_{model_name}.pkl", ""]:
                try:
                    dl = self.client.download_artifacts(
                        run_id, art, dst_path=str(self.cache_dir)
                    )
                    if self._extract_downloaded_model(dl):
                        break
                except Exception:
                    pass

            if self.model_path.exists():
                self._save_version(
                    version="run-" + run_id[:8],
                    run_id=run_id,
                    model_name=model_name,
                    val_acc=val_acc,
                )
                return str(self.model_path), {
                    "source": "experiment_fresh",
                    "model_name": model_name,
                    "run_id": run_id,
                    "cached": False,
                    "val_accuracy": val_acc,
                }

        except Exception as e:
            logger.warning(f"Experiment search fallback error: {e}")

        return None

    def fetch_best_model(self) -> Tuple[bool, str, dict]:
        """
        Orchestrate model fetch:
        1. MLFlow Model Registry ('cats-dogs-best-model')
        2. MLFlow Experiment Runs ('cats-dogs-k8s')
        3. Local PV / Cached / Packaged weights fallback
        """
        reg_result = self._fetch_from_registry()
        if reg_result:
            return True, reg_result[0], reg_result[1]

        exp_result = self._fetch_from_experiment_runs()
        if exp_result:
            return True, exp_result[0], exp_result[1]

        if self.model_path.exists():
            cached = self._load_cached_version() or {}
            logger.info("Using existing model file in persistent cache volume.")
            return (
                True,
                str(self.model_path),
                {
                    "source": "volume_cached",
                    "model_name": cached.get("model_name", "resnet18"),
                    "version": cached.get("version", "unknown"),
                    "cached": True,
                },
            )

        for bundled_candidate in ["models/best_model.pkl", "src/models/best_model.pkl"]:
            b_path = Path(bundled_candidate)
            if b_path.exists():
                shutil.copy2(b_path, self.model_path)
                logger.info(
                    f"Copied bundled fallback model {bundled_candidate} to {self.model_path}"
                )
                return (
                    True,
                    str(self.model_path),
                    {
                        "source": "bundled_fallback",
                        "model_name": "resnet18",
                        "cached": True,
                    },
                )

        return (
            False,
            "",
            {"source": "none", "message": "No model found in MLFlow or local storage"},
        )


def load_model_with_mlflow(
    mlflow_uri: str = "http://mlflow:5000",
    registered_model_name: str = "cats-dogs-best-model",
    experiment_name: str = "cats-dogs-k8s",
    model_stage: str = "Production",
    cache_dir: str = "/app-models",
) -> Tuple[bool, str, dict]:
    """Convenience helper function"""
    fetcher = MLFlowModelFetcher(
        mlflow_uri=mlflow_uri,
        registered_model_name=registered_model_name,
        experiment_name=experiment_name,
        model_stage=model_stage,
        cache_dir=cache_dir,
    )
    return fetcher.fetch_best_model()
