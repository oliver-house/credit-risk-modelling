from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import MLFLOW_DB_PATH, MLFLOW_EXPERIMENT, MLRUNS_DIR
from src.utils.helpers import get_logger

logger = get_logger(__name__)

MAX_PARAM_CHARS = 500


def tracking_uri(db_path: Path = MLFLOW_DB_PATH) -> str:
    return "sqlite:///" + Path(db_path).resolve().as_posix()


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, inner in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(name, inner))
        return out
    return {prefix: value}


class RunTracker:

    def __init__(
        self,
        enabled: bool = True,
        run_name: str | None = None,
        experiment: str = MLFLOW_EXPERIMENT,
        db_path: Path = MLFLOW_DB_PATH,
        artifact_dir: Path = MLRUNS_DIR,
        tags: dict[str, Any] | None = None,
    ) -> None:
        self.run_name = run_name
        self.experiment = experiment
        self.db_path = Path(db_path)
        self.artifact_dir = Path(artifact_dir)
        self.tags = tags or {}
        self._mlflow = None
        self._run = None
        self.enabled = enabled and self._connect()

    def _connect(self) -> bool:
        try:
            import mlflow
        except ImportError:
            logger.warning(
                "mlflow-skinny is not installed; this run will not be tracked. "
                "Install it with `pip install -r requirements.txt`."
            )
            return False

        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_uri(self.db_path))
            client = mlflow.tracking.MlflowClient()
            if client.get_experiment_by_name(self.experiment) is None:
                client.create_experiment(
                    self.experiment,
                    artifact_location=self.artifact_dir.resolve().as_uri(),
                )
            mlflow.set_experiment(self.experiment)
        except Exception as exc:
            logger.warning(f"MLflow could not be initialised ({exc}); "
                           f"this run will not be tracked.")
            return False

        self._mlflow = mlflow
        return True


    def __enter__(self) -> RunTracker:
        if not self.enabled:
            return self
        try:
            self._run = self._mlflow.start_run(run_name=self.run_name)
            if self.tags:
                self._mlflow.set_tags({k: str(v) for k, v in self.tags.items()})
            logger.info(f"Tracking run {self._run.info.run_id} in {self.db_path}")
        except Exception as exc:
            self._disable(exc)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.enabled and self._run is not None:
            status = "FINISHED" if exc_type is None else "FAILED"
            try:
                self._mlflow.end_run(status=status)
            except Exception as end_exc:
                logger.warning(f"MLflow could not close the run: {end_exc}")
        return False

    def _disable(self, exc: Exception) -> None:
        logger.warning(f"MLflow call failed ({exc}); tracking disabled for this run.")
        self.enabled = False

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run is not None else None


    def log_params(self, params: dict[str, Any], prefix: str = "") -> None:
        if not self.enabled:
            return
        flat = flatten(prefix, params)
        cleaned = {k: str(v)[:MAX_PARAM_CHARS] for k, v in flat.items() if v is not None}
        if not cleaned:
            return
        try:
            self._mlflow.log_params(cleaned)
        except Exception as exc:
            self._disable(exc)

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "") -> None:
        if not self.enabled:
            return
        flat = flatten(prefix, metrics)
        numeric = {
            k: float(v) for k, v in flat.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if not numeric:
            return
        try:
            self._mlflow.log_metrics(numeric)
        except Exception as exc:
            self._disable(exc)

    def log_artifact(self, path: Path | str) -> None:
        if not self.enabled:
            return
        path = Path(path)
        if not path.exists():
            logger.warning(f"Not logging {path} as an artefact: it does not exist")
            return
        try:
            self._mlflow.log_artifact(str(path))
        except Exception as exc:
            self._disable(exc)

    def log_artifacts(self, paths) -> None:
        for path in paths:
            self.log_artifact(path)
