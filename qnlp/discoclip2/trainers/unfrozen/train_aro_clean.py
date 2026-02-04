import torch
from typing import Literal
import mlflow
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import trange
from torch.nn.functional import cosine_similarity
import torchmetrics


from qnlp.discoclip2.models.loss import create_loss_functions
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device

from qnlp.discoclip2.dataset.aro_dataloader import get_aro_dataloader
from qnlp.discoclip2.models.einsum_model import get_einsum_model
from qnlp.discoclip2.models.image_model import TTNImageModel

EXPERIMENT_NAME = "train_vlm_on_aro"

logger = setup_logger(log_name=EXPERIMENT_NAME)

DEVICE = get_device()
set_seed()
global_step = 0


class ModelSettings(BaseSettings):
    embedding_dim: int = 512
    bond_dim: int = 10

    batch_size: int = 128
    text_learning_rate: float = 0.001
    text_weight_decay: float = 0.001

    image_learning_rate: float = 0.005
    image_weight_decay: float = 0.0005

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
    epoch_avg_metrics = {
        "loss": torchmetrics.MeanMetric().to(DEVICE),
        "contrastive_loss": torchmetrics.MeanMetric().to(DEVICE),
        "contrastive_acc": torchmetrics.MeanMetric().to(DEVICE),
        "hard_neg_loss": torchmetrics.MeanMetric().to(DEVICE),
        "hard_neg_acc": torchmetrics.MeanMetric().to(DEVICE),
        "hard_neg_draw": torchmetrics.MeanMetric().to(DEVICE),
        "true_caption_embedding_mean_norm": torchmetrics.MeanMetric().to(DEVICE),
        "false_caption_embedding_mean_norm": torchmetrics.MeanMetric().to(DEVICE),
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
        "true_cosine_mean": torch.zeros(num_batches, device=DEVICE),
        "false_cosine_mean": torch.zeros(num_batches, device=DEVICE),
    }

    for batch in train_dataloader:
        optimizer.zero_grad()

        images = batch["images"].to(DEVICE)
        true_captions = batch["true_captions"]
        false_captions = batch["false_captions"]

        image_embeddings = image_model(images)
        true_caption_embeddings = text_model(true_captions)
        false_caption_embeddings = text_model(false_captions)

        true_caption_embeddings_norm = true_caption_embeddings.norm(dim=-1).mean().item()
        false_caption_embeddings_norm = false_caption_embeddings.norm(dim=-1).mean().item()
        image_embeddings_norm = image_embeddings.norm(dim=-1).mean().item()

        infonce_loss, infonce_acc = contrastive_criterion(
            image_embeddings, true_caption_embeddings
        )
        pos_sim = cosine_similarity(true_caption_embeddings, image_embeddings, dim=-1)
        neg_sim = cosine_similarity(false_caption_embeddings, image_embeddings, dim=-1)

        true_cosine_mean = pos_sim.mean().item()
        false_cosine_mean = neg_sim.mean().item()

        hard_neg_acc = (pos_sim > neg_sim).float().mean().item()
        hard_neg_draw = (pos_sim == neg_sim).float().mean().item()
        hard_neg_loss = hard_neg_criterion(
            image_embeddings,
            true_caption_embeddings,
            false_caption_embeddings
        )

        loss = infonce_loss + hard_neg_loss_weight * hard_neg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(text_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(image_model.parameters(), max_norm=1.0)
        optimizer.step()
        global_step += 1


def run_training():
    hyperparams = ModelSettings()
    with setup_mlflow_run(EXPERIMENT_NAME, hyperparams.model_dump()):

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