from typing import Any
from contextlib import contextmanager
import mlflow
from datetime import datetime


@contextmanager
def setup_mlflow_run(experiment_name: str, params: dict[str, Any], port: int = 5000):
    ts_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    mlflow.config.enable_async_logging()
    mlflow.set_tracking_uri(f"http://localhost:{port}")
    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(15)

    run_name = input("Please provide a short description for the run:\n> ")
    run_name = run_name.replace(" ", "_")
    run_name = f"{run_name}_{ts_string}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        yield run