from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import trange

from qnlp.core.training.protocols import TrainingStep
from qnlp.utils.early_stopping import EarlyStopping, ModelTrainingStatus
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="trainer")


class MetricsAccumulator:
    """
    Accumulates per-batch metric tensors on-device using torchmetrics.MeanMetric.
    Avoids .item() / CPU sync in the hot loop — sync happens once at epoch end
    via compute().
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._metrics: dict[str, MeanMetric] = {}

    def update(self, metrics: dict[str, Tensor]) -> None:
        for k, v in metrics.items():
            if k not in self._metrics:
                self._metrics[k] = MeanMetric().to(self.device)
            self._metrics[k].update(v)

    def compute(self) -> dict[str, float]:
        """Sync to CPU once per epoch."""
        return {k: m.compute().item() for k, m in self._metrics.items()}

    def reset(self) -> None:
        for m in self._metrics.values():
            m.reset()


class Trainer:
    """
    Generic training loop. Knows nothing about task, model architecture, or loss.

    The TrainingStep owns: batch unpacking, model forward, loss, metrics.
    The Trainer owns: epoch loop, optimizer steps, grad clipping, early stopping,
                      checkpointing, MLflow epoch logging.

    Grad clipping: if the model exposes clip_gradients(max_norm), it is called
    (enabling per-sub-model clipping). Otherwise falls back to clipping all
    model parameters together.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: TrainingStep,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        monitor_metric: str,
        checkpoint_path: Path,
        max_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 0.0001,
        max_grad_norm: float = 1.0,
        device: torch.device | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.step = step
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.monitor_metric = monitor_metric
        self.checkpoint_path = Path(checkpoint_path)
        self.max_epochs = max_epochs
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.device = device or torch.device("cpu")

        self._early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, minimize=False)

    def fit(self) -> dict[str, float]:
        """
        Run the full training loop. Returns test metrics from the best checkpoint.
        """
        best_epoch = 0

        for epoch in trange(1, self.max_epochs + 1, desc="Epochs"):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            mlflow.log_metrics({f"train/epoch_{k}": v for k, v in train_metrics.items()}, step=epoch)
            logger.info(f"Epoch {epoch} train: {train_metrics}")

            val_metrics = self._run_epoch(self.val_loader, train=False)
            mlflow.log_metrics({f"val/epoch_{k}": v for k, v in val_metrics.items()}, step=epoch)
            logger.info(f"Epoch {epoch} val: {val_metrics}")

            if self.monitor_metric not in val_metrics:
                raise KeyError(
                    f"monitor_metric '{self.monitor_metric}' not found in step metrics. "
                    f"Available: {list(val_metrics.keys())}"
                )

            status = self._early_stopping(val_metrics[self.monitor_metric])

            if status == ModelTrainingStatus.improved:
                best_epoch = epoch
                self._save_checkpoint(epoch, val_metrics)
                logger.info(f"Epoch {epoch}: new best — checkpoint saved.")
            elif status == ModelTrainingStatus.stop:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

            if self.scheduler is not None:
                self.scheduler.step()

        logger.info(f"Training complete. Best epoch: {best_epoch}. Running test evaluation.")
        self._load_checkpoint()

        test_metrics = self._run_epoch(self.test_loader, train=False)
        mlflow.log_metrics({f"test/epoch_{k}": v for k, v in test_metrics.items()}, step=best_epoch)
        logger.info(f"Test metrics: {test_metrics}")

        return test_metrics

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        accumulator = MetricsAccumulator(self.device)

        with torch.set_grad_enabled(train):
            for batch in loader:
                if train:
                    self.optimizer.zero_grad()

                loss, metrics = self.step(self.model, batch, train)

                if train:
                    loss.backward()
                    self._clip_gradients()
                    self.optimizer.step()

                accumulator.update(metrics)

        return accumulator.compute()

    def _clip_gradients(self) -> None:
        if hasattr(self.model, "clip_gradients"):
            self.model.clip_gradients(self.max_grad_norm)
        else:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def _save_checkpoint(self, epoch: int, val_metrics: dict[str, float]) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            },
            self.checkpoint_path,
        )

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}.")
