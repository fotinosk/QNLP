from datetime import datetime
import torch
from typing import Literal
import mlflow
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor
from torch.nn import Module
from torchmetrics import MeanMetric
from tqdm import trange
from torch.nn.functional import cosine_similarity
import torchmetrics

from qnlp.discoclip2.models.loss import create_loss_functions
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device, create_checkpoint_path

from qnlp.discoclip2.dataset.aro_dataloader import get_aro_dataloader
from qnlp.discoclip2.models.einsum_model import get_einsum_model, EinsumModel
from qnlp.discoclip2.models.image_model import TTNImageModel

EXPERIMENT_NAME = "train_vlm_on_aro"
ts_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
checkpoint_path = create_checkpoint_path(EXPERIMENT_NAME, ts_string)
logger = setup_logger(log_name=EXPERIMENT_NAME, ts_string=ts_string)

DEVICE = get_device()
set_seed()
global_step = 0


class ModelSettings(BaseSettings):
    embedding_dim: int = 512
    bond_dim: int = 10

    batch_size: int = 128
    text_learning_rate: float = 0.001
    text_weight_decay: float = 0.001

    image_learning_rate: float = 0.0001
    image_weight_decay: float = 0.01

    epochs: int = 100
    patience: int = 5

    temperature: float = 0.07
    hard_neg_loss_weight: float = 1.0
    hard_neg_margin: float = 0.2
    hard_neg_distance_function: Literal["euclidean", "cosine"] = "euclidean"
    hard_neg_swap: bool = False

    # Configuration for Environment Variables
    model_config = SettingsConfigDict(
        env_prefix='ML_',           # Prefix for env vars
    )


def create_epoch_metrics(
        num_batches: int
) -> tuple[
    dict[str, Tensor],
    dict[str, MeanMetric]
]:
    epoch_avg_metrics = {
        "loss": torchmetrics.MeanMetric().to(DEVICE),
        "contrastive_loss": torchmetrics.MeanMetric().to(DEVICE),
        "contrastive_acc": torchmetrics.MeanMetric().to(DEVICE),
        "hard_neg_loss": torchmetrics.MeanMetric().to(DEVICE),
        "hard_neg_acc": torchmetrics.MeanMetric().to(DEVICE),
        "hard_neg_draw": torchmetrics.MeanMetric().to(DEVICE),
        "true_caption_embedding_mean_norm": torchmetrics.MeanMetric().to(DEVICE),
        "false_caption_embedding_mean_norm": torchmetrics.MeanMetric().to(DEVICE),
        "image_embedding_mean_norm": torchmetrics.MeanMetric().to(DEVICE),
        "true_cosine_mean": torchmetrics.MeanMetric().to(DEVICE),
        "false_cosine_mean": torchmetrics.MeanMetric().to(DEVICE),
    }
    # avoid running .item() in the training loop
    batch_avg_metrics = {
        "loss": torch.zeros(num_batches, device=DEVICE),
        "contrastive_loss": torch.zeros(num_batches, device=DEVICE),
        "contrastive_acc": torch.zeros(num_batches, device=DEVICE),
        "hard_neg_loss": torch.zeros(num_batches, device=DEVICE),
        "hard_neg_acc": torch.zeros(num_batches, device=DEVICE),
        "hard_neg_draw": torch.zeros(num_batches, device=DEVICE),
        "true_caption_embedding_mean_norm": torch.zeros(num_batches, device=DEVICE),
        "false_caption_embedding_mean_norm": torch.zeros(num_batches, device=DEVICE),
        "image_embedding_mean_norm": torch.zeros(num_batches, device=DEVICE),
        "true_cosine_mean": torch.zeros(num_batches, device=DEVICE),
        "false_cosine_mean": torch.zeros(num_batches, device=DEVICE),
    }
    return batch_avg_metrics, epoch_avg_metrics


def calculate_composite_lost(
        contrastive_criterion: Module,
        hard_neg_criterion: Module,
        true_caption_embeddings: torch.Tensor,
        false_caption_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        hard_neg_loss_weight: float
) -> tuple:
    infonce_loss, infonce_acc = contrastive_criterion(
        image_embeddings, true_caption_embeddings
    )

    hard_neg_loss = hard_neg_criterion(
        image_embeddings,
        true_caption_embeddings,
        false_caption_embeddings
    )

    loss = infonce_loss + hard_neg_loss_weight * hard_neg_loss
    return hard_neg_loss, infonce_acc, infonce_loss, loss


def calculate_and_store_metrics(
        batch_avg_metrics: dict[str, Tensor],
        epoch_avg_metrics: dict[str, MeanMetric],
        false_caption_embeddings: Tensor,
        hard_neg_loss, i: int,
        image_embeddings: Tensor,
        infonce_acc: Tensor,
        infonce_loss: Tensor,
        loss: Tensor,
        true_caption_embeddings: Tensor
) -> None:
    # calculate training metrics
    pos_sim = cosine_similarity(true_caption_embeddings, image_embeddings, dim=-1)
    neg_sim = cosine_similarity(false_caption_embeddings, image_embeddings, dim=-1)

    true_cosine_mean = pos_sim.mean()
    false_cosine_mean = neg_sim.mean()

    true_caption_embeddings_norm = true_caption_embeddings.norm(dim=-1).mean()
    false_caption_embeddings_norm = false_caption_embeddings.norm(dim=-1).mean()
    image_embeddings_norm = image_embeddings.norm(dim=-1).mean()

    hard_neg_acc = (pos_sim > neg_sim).float().mean()
    hard_neg_draw = (pos_sim == neg_sim).float().mean()

    batch_avg_metrics["loss"][i] = loss.detach()
    batch_avg_metrics["contrastive_loss"][i] = infonce_loss.detach()
    batch_avg_metrics["contrastive_acc"][i] = infonce_acc.detach()
    batch_avg_metrics["hard_neg_loss"][i] = hard_neg_loss.detach()
    batch_avg_metrics["hard_neg_acc"][i] = hard_neg_acc.detach()
    batch_avg_metrics["hard_neg_draw"][i] = hard_neg_draw.detach()
    batch_avg_metrics["true_caption_embedding_mean_norm"][i] = true_caption_embeddings_norm.detach()
    batch_avg_metrics["false_caption_embedding_mean_norm"][i] = false_caption_embeddings_norm.detach()
    batch_avg_metrics["image_embedding_mean_norm"][i] = image_embeddings_norm.detach()
    batch_avg_metrics["true_cosine_mean"][i] = true_cosine_mean.detach()
    batch_avg_metrics["false_cosine_mean"][i] = false_cosine_mean.detach()

    for key in epoch_avg_metrics:
        epoch_avg_metrics[key].update(batch_avg_metrics[key][i])


def evaluate_models(
        text_model: torch.nn.Module,
        image_model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        contrastive_criterion: torch.nn.Module,
        hard_neg_criterion: torch.nn.Module,
        hard_neg_loss_weight: float,
        epoch: int,
        usage: Literal['test', 'val']
) -> float:
    text_model.eval()
    image_model.eval()

    num_batches = len(dataloader)
    # Initialize metrics
    batch_avg_metrics, epoch_avg_metrics = create_epoch_metrics(num_batches)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch["images"].to(DEVICE)
            true_captions = batch["true_captions"]
            false_captions = batch["false_captions"]

            image_embeddings = image_model(images)
            true_caption_embeddings = text_model(true_captions)
            false_caption_embeddings = text_model(false_captions)

            hard_neg_loss, infonce_acc, infonce_loss, loss = calculate_composite_lost(
                contrastive_criterion=contrastive_criterion,
                hard_neg_criterion=hard_neg_criterion,
                true_caption_embeddings=true_caption_embeddings,
                false_caption_embeddings=false_caption_embeddings,
                image_embeddings=image_embeddings,
                hard_neg_loss_weight=hard_neg_loss_weight,
            )

            calculate_and_store_metrics(
                batch_avg_metrics=batch_avg_metrics,
                epoch_avg_metrics=epoch_avg_metrics,
                false_caption_embeddings=false_caption_embeddings,
                hard_neg_loss=hard_neg_loss,
                i=i,
                image_embeddings=image_embeddings,
                infonce_acc=infonce_acc,
                infonce_loss=infonce_loss,
                loss=loss,
                true_caption_embeddings=true_caption_embeddings
            )

        cpu_batch_metrics = {
            k: v.cpu().tolist() for k, v in batch_avg_metrics.items()
        }

        start_step = global_step - num_batches
        for offset in range(num_batches):
            mlflow.log_metrics(
                {f"{usage}/batch_{k}": v[offset] for k, v in cpu_batch_metrics.items()},
                step=start_step + offset
            )

        final_epoch_logs = {}
        logger.info(f"\n--- {usage.upper()} Epoch {epoch} Results ---")
        for key, metric_obj in epoch_avg_metrics.items():
            avg_val = metric_obj.compute().item()
            final_epoch_logs[f"{usage}/epoch_{key}"] = avg_val
            logger.info(f"{key}: {avg_val:.4f}")

        mlflow.log_metrics(final_epoch_logs, step=epoch)
        return final_epoch_logs[f"{usage}/epoch_hard_neg_loss"]


def train_epoch(
        text_model: torch.nn.Module,
        image_model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        contrastive_criterion: torch.nn.Module,
        hard_neg_criterion: torch.nn.Module,
        hard_neg_loss_weight: float,
        epoch: int
) -> None:
    global global_step
    text_model.train()
    image_model.train()

    num_batches = len(train_dataloader)
    # Initialize metrics
    batch_avg_metrics, epoch_avg_metrics = create_epoch_metrics(num_batches)

    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        images = batch["images"].to(DEVICE)
        true_captions = batch["true_captions"]
        false_captions = batch["false_captions"]

        image_embeddings = image_model(images)
        true_caption_embeddings = text_model(true_captions)
        false_caption_embeddings = text_model(false_captions)

        hard_neg_loss, infonce_acc, infonce_loss, loss = calculate_composite_lost(
            contrastive_criterion=contrastive_criterion,
            hard_neg_criterion=hard_neg_criterion,
            true_caption_embeddings=true_caption_embeddings,
            false_caption_embeddings=false_caption_embeddings,
            image_embeddings=image_embeddings,
            hard_neg_loss_weight=hard_neg_loss_weight,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(text_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(image_model.parameters(), max_norm=1.0)
        optimizer.step()

        calculate_and_store_metrics(batch_avg_metrics, epoch_avg_metrics, false_caption_embeddings, hard_neg_loss, i,
                                    image_embeddings, infonce_acc, infonce_loss, loss, true_caption_embeddings)

        global_step += 1

    cpu_batch_metrics = {
        k: v.cpu().tolist() for k, v in batch_avg_metrics.items()
    }

    start_step = global_step - num_batches
    for offset in range(num_batches):
        mlflow.log_metrics(
            {f"train/batch_{k}": v[offset] for k, v in cpu_batch_metrics.items()},
            step=start_step + offset
        )

    final_epoch_logs = {}
    logger.info(f"\n--- Epoch {epoch} Results ---")
    for key, metric_obj in epoch_avg_metrics.items():
        avg_val = metric_obj.compute().item()
        final_epoch_logs[f"train/epoch_{key}"] = avg_val
        logger.info(f"{key}: {avg_val:.4f}")

    mlflow.log_metrics(final_epoch_logs, step=epoch)


def run_training():
    hyperparams = ModelSettings()
    with setup_mlflow_run(EXPERIMENT_NAME, hyperparams.model_dump(), 8080):

        # get datasets and dataloaders
        loaders, datasets = get_aro_dataloader(
            batch_size=hyperparams.batch_size,
            return_images=True
        )
        train_loader, val_loader, test_loader = loaders
        train_ds, val_ds, test_ds = datasets

        logger.info({
            "message": "created datasets and dataloaders",
            "train_loader_size": len(train_ds),
            "val_loader_size": len(val_ds),
            "test_loader_size": len(test_ds)
        })

        # get models
        model = get_einsum_model([train_ds, val_ds, test_ds]).to(DEVICE)
        image_model = TTNImageModel(hyperparams.embedding_dim).to(DEVICE)

        mlflow.log_params({
            "text_model_total_params": sum(p.numel() for p in model.parameters()),
            "image_model_total_params": sum(p.numel() for p in image_model.parameters())
        })

        # get loss functions
        contrastive_loss, hard_neg_loss = create_loss_functions(
            temperature=hyperparams.temperature,
            hard_neg_distance_function=hyperparams.hard_neg_distance_function,
            margin=hyperparams.hard_neg_margin,
            swap=hyperparams.hard_neg_swap
        )

        # get optimizer
        optimizer = torch.optim.AdamW([
            {
                "params": model.parameters(),
                "lr": hyperparams.text_learning_rate,
                "weight_decay": hyperparams.text_weight_decay
            },
            {
                "params": image_model.parameters(),
                "lr": hyperparams.image_learning_rate,
                "weight_decay": hyperparams.image_weight_decay
            }
        ])
        logger.info(optimizer)
        logger.info(f"Starting training with {hyperparams.epochs} epochs")

        best_loss = float("inf")

        for epoch in trange(1, hyperparams.epochs + 1, desc="Training Epochs"):
            logger.info(f"Starting epoch {epoch}/{hyperparams.epochs}")

            train_epoch(
                text_model=model,
                image_model=image_model,
                train_dataloader=train_loader,
                optimizer=optimizer,
                contrastive_criterion=contrastive_loss,
                hard_neg_criterion=hard_neg_loss,
                hard_neg_loss_weight=hyperparams.hard_neg_loss_weight,
                epoch=epoch
            )

            loss = evaluate_models(
                text_model=model,
                image_model=image_model,
                dataloader=val_loader,
                contrastive_criterion=contrastive_loss,
                hard_neg_criterion=hard_neg_loss,
                hard_neg_loss_weight=hyperparams.hard_neg_loss_weight,
                epoch=epoch,
                usage="val"
            )

            if loss < best_loss:
                best_loss = loss
                logger.info(f"New best model found - {epoch=}")

                checkpoint = {
                    "text_model_state_dict": model.state_dict(),
                    "image_model_state_dict": image_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                }

                torch.save(checkpoint, checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                logger.info(f"Saved best model checkpoint to {checkpoint_path}")

        logger.info("Finished training model, generating final metrics on test set and best model")
        best_checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        text_model = EinsumModel()
        text_model.load_state_dict(best_checkpoint["text_model_state_dict"])
        text_model = text_model.to(DEVICE)

        image_model = TTNImageModel(hyperparams.embedding_dim)
        image_model.load_state_dict(best_checkpoint["image_model_state_dict"])
        image_model = image_model.to(DEVICE)

        evaluate_models(
            text_model=text_model,
            image_model=image_model,
            dataloader=test_loader,
            contrastive_criterion=contrastive_loss,
            hard_neg_criterion=hard_neg_loss,
            hard_neg_loss_weight=hyperparams.hard_neg_loss_weight,
            epoch=hyperparams.epochs+1,
            usage="test"
        )


if __name__ == "__main__":
    run_training()
