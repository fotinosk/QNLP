import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import mlflow


@dataclass
class _MockRun:
    """Returned by setup_mlflow_run when MLflow is disabled."""

    class _Info:
        run_name: str = "disabled"

    info: _Info = None

    def __post_init__(self):
        self.info = self._Info()


@contextmanager
def setup_mlflow_run(experiment_name: str, params: dict[str, Any], port: int = 5000, run_name: str | None = None):
    import logging

    logger = logging.getLogger("mlflow_utils")

    ts_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if run_name is None:
        run_name = os.environ.get("MLFLOW_RUN_NAME") or input("Please provide a short description for the run:\n> ")
    run_name = run_name.replace(" ", "_")
    run_name = f"{run_name}_{ts_string}"

    if os.environ.get("MLFLOW_DISABLED", "").lower() in ("1", "true"):
        logger.info(f"MLflow disabled — run '{run_name}', params: {params}")
        mock = _MockRun()
        mock.info.run_name = run_name
        yield mock
        return

    mlflow.config.enable_async_logging()
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"http://localhost:{port}")
    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(15)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        yield run
