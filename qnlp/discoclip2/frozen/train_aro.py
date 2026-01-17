import logging
import os
from diskcache import Cache

import mlflow
import torch
import torch.optim as optim
from lambeq import AtomicType, Rewriter, Symbol
from lambeq.backend.tensor import Dim
from torch import nn
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import trange

from qnlp.discoclip2.dataset.aro_dataset import ARODataset, aro_tn_collate_fn
from qnlp.discoclip2.models.bobcat_text_processor import BobcatTextProcessor
from qnlp.discoclip2.models.loss import InfoNCE
from qnlp.discoclip2.models.lookup_embeddings import LookupEmbedding
from qnlp.discoclip2.models.einsum_model import EinsumModel

from qnlp.discoclip2.parser.cached_bobcat import CachedBobcatParser
from qnlp.discoclip2.parser.asnsatz import CustomMPSAnsatz


torch.serialization.add_safe_globals([Symbol])

READER = "bobcat"
CHECKPOINT_PATH = "checkpoints/"

EMBEDDING_DIM= 768
BOND_DIM= 16

DEVICE = "mps"
SEED   = 42


TRAIN_DATA_PATH = "data/aro/processed/combined/train.json"
VAL_DATA_PATH   = "data/aro/processed/combined/val.json"
TEST_DATA_PATH  = "data/aro/processed/combined/test.json"
IMAGE_LOOKUP_PATH = "models/lookup_embedding_ViT-B_32.pt"

BATCH_SIZE    = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 0.001
EPOCHS        = 10
PATIENCE      = 5

TEMPERATURE = 0.07
HARD_NEG_LOSS_WEIGHT = 1.0
HARD_NEG_MARGIN      = 0.2
HARD_NEG_DISTANCE_FUNCTION = "euclidian" 
HARD_NEG_SWAP        = False          

LOG_PATH        = "runs/logs/"
CHECKPOINT_PATH = "./checkpoints"
MLFLOW_EXPERIMENT = "discoclip_aro_experiment"
MLFLOW_URI = None 


def get_aro_dataset(
    data_path: str, dim: int, bond_dim: int, progress: bool = True
) -> ARODataset:
    
    cache_dir = os.path.join(os.path.dirname(data_path), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cache = Cache(cache_dir, size_limit=10 * 2**30) # 10GB
   
    cache_key = os.path.join(
        os.path.basename(data_path),
        f"{READER}_dim{dim}_bond{bond_dim}"
    )
    
    state_dict = cache.get(cache_key)
    if state_dict is not None:
        print(f"Loading cached dataset from {cache_key}")

        dataset = ARODataset(data_path=data_path, text_transform=None)
        dataset.load_state_dict(state_dict)
        return dataset

    ansatz = CustomMPSAnsatz(
        {
            AtomicType.SENTENCE: Dim(dim),
            AtomicType.NOUN: Dim(dim),
            AtomicType.PREPOSITIONAL_PHRASE: Dim(dim),
        },
        bond_dim=bond_dim,
    )

    rules = [
        "auxiliary",
        "connector",
        "determiner",
        "postadverb",
        "preadverb",
        "prepositional_phrase",
        "coordination",
        "object_rel_pronoun",
        "subject_rel_pronoun",
    ]
    rewriter = Rewriter(rules)
    bobcat_parser = CachedBobcatParser()
    text_transform = BobcatTextProcessor(
        ccg_parser=bobcat_parser,
        ansatz=ansatz,
        rewriter=rewriter,
    )
    
    dataset = ARODataset(
        data_path=data_path, text_transform=text_transform, progress=progress
    )

    print(f"Caching dataset to {cache_key}")
    cache.set(cache_key, dataset.state_dict())
    cache.close()
    return dataset


def setup_logger(log_path):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def train_epoch(
    model,
    image_model,
    dataloader,
    contrastive_criterion,
    hard_neg_criterion,
    optimizer,
    hard_neg_loss_weight=0.0,
    device="cpu",
):
    """
    Train the model for one epoch.
    """
    model.train()
    image_model.train()

    metrics = {
        "loss": 0.0,
        "contrastive_loss": 0.0,
        "contrastive_acc": 0.0,
        "hard_neg_loss": 0.0,
        "hard_neg_acc": 0.0,
        "hard_neg_draw": 0.0,
        "true_caption_embedding_mean_norm": 0.0,
        "false_caption_embedding_mean_norm": 0.0,
        "true_cosine_mean": 0.0,
        "false_cosine_mean": 0.0,
    }

    total_samples = 0

    for batch in dataloader:
        optimizer.zero_grad()

        images = batch["images"]
        true_captions = batch["true_captions"]
        false_captions = batch["false_captions"]
        batch_size = len(images)

        with torch.no_grad():
            image_embeddings = image_model(images).to(device)

        true_caption_embeddings = model(true_captions)
        false_caption_embeddings = model(false_captions)

        metrics["true_caption_embedding_mean_norm"] += (
            true_caption_embeddings.norm(dim=-1).mean().item()
        )
        metrics["false_caption_embedding_mean_norm"] += (
            false_caption_embeddings.norm(dim=-1).mean().item()
        )

        infonce_loss, infonce_acc = contrastive_criterion(
            image_embeddings, true_caption_embeddings
        )

        pos_sim = cosine_similarity(true_caption_embeddings, image_embeddings, dim=-1)
        neg_sim = cosine_similarity(false_caption_embeddings, image_embeddings, dim=-1)
        
        metrics["true_cosine_mean"] += pos_sim.mean().item()
        metrics["false_cosine_mean"] += neg_sim.mean().item()

        hard_neg_acc = (pos_sim > neg_sim).float().mean().item()
        hard_neg_draw = (pos_sim == neg_sim).float().mean().item()
        hard_neg_loss = hard_neg_criterion(image_embeddings, 
                                           true_caption_embeddings, false_caption_embeddings)
        
        loss = infonce_loss + hard_neg_loss_weight * hard_neg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics["loss"] += loss.item() * batch_size
        metrics["contrastive_loss"] += infonce_loss.item() * batch_size
        metrics["contrastive_acc"] += infonce_acc.item() * batch_size
        metrics["hard_neg_loss"] += hard_neg_loss.item() * batch_size
        metrics["hard_neg_acc"] += hard_neg_acc * batch_size
        metrics["hard_neg_draw"] += hard_neg_draw * batch_size
        total_samples += batch_size

    for key in metrics:
        metrics[key] /= total_samples
    return metrics


def evaluate_model(
    model,
    image_model,
    dataloader,
    contrastive_criterion,
    hard_neg_criterion,
    hard_neg_loss_weight=0.0,
    device="cpu",
):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    image_model.eval()

    metrics = {
        "loss": 0.0,
        "contrastive_loss": 0.0,
        "contrastive_acc": 0.0,
        "hard_neg_loss": 0.0,
        "hard_neg_acc": 0.0,
        "hard_neg_draw": 0.0,
        "true_caption_embedding_mean_norm": 0.0,
        "false_caption_embedding_mean_norm": 0.0,
        "true_cosine_mean": 0.0,
        "false_cosine_mean": 0.0,
    }

    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"]
            true_captions = batch["true_captions"]
            false_captions = batch["false_captions"]
            batch_size = len(images)

            image_embeddings = image_model(images).to(device)
            true_caption_embeddings = model(true_captions)
            false_caption_embeddings = model(false_captions)

            metrics["true_caption_embedding_mean_norm"] += (
                true_caption_embeddings.norm(dim=-1).mean().item()
            )
            metrics["false_caption_embedding_mean_norm"] += (
                false_caption_embeddings.norm(dim=-1).mean().item()
            )

            infonce_loss, infonce_acc = contrastive_criterion(
                image_embeddings, true_caption_embeddings
            )

            pos_sim = cosine_similarity(true_caption_embeddings, image_embeddings, dim=-1)
            neg_sim = cosine_similarity(false_caption_embeddings, image_embeddings, dim=-1)

            metrics["true_cosine_mean"] += pos_sim.mean().item()
            metrics["false_cosine_mean"] += neg_sim.mean().item()

            hard_neg_acc = (pos_sim > neg_sim).float().mean().item()
            hard_neg_draw = (pos_sim == neg_sim).float().mean().item()
            hard_neg_loss = hard_neg_criterion(
                image_embeddings, 
                true_caption_embeddings, 
                false_caption_embeddings
            )

            loss = infonce_loss + hard_neg_loss_weight * hard_neg_loss

            metrics["contrastive_loss"] += infonce_loss.item() * batch_size
            metrics["contrastive_acc"] += infonce_acc.item() * batch_size
            metrics["loss"] += loss.item() * batch_size
            metrics["hard_neg_loss"] += hard_neg_loss.item() * batch_size
            metrics["hard_neg_acc"] += hard_neg_acc * batch_size
            metrics["hard_neg_draw"] += hard_neg_draw * batch_size
            total_samples += batch_size

    for key in metrics:
        metrics[key] /= total_samples
    return metrics

def get_einsum_model(datasets: list):
    symbol_sizes = dict()
    for ds in datasets:
        for sym, size in zip(ds.symbols, ds.sizes):
            if sym in symbol_sizes and symbol_sizes[sym] != size:
                raise ValueError(f"Symbol {sym} has different sizes in the datasets: {symbol_sizes[sym]} and {size}")
            symbol_sizes[sym] = size
    
    symbols = list(symbol_sizes.keys())
    sizes = list(symbol_sizes.values())
            
    model = EinsumModel(symbols, sizes)
    return model


def train_model(parent_run=None):
    with mlflow.start_run(parent_run_id=parent_run.info.run_id if parent_run else None,
                          nested=True if parent_run else False) as run:

        logger = setup_logger(
            os.path.join(LOG_PATH, f"train_{run.info.run_id}.log")
        )
        logger.info(
            f"Running experiment: {MLFLOW_EXPERIMENT}, run ID: {run.info.run_id}, run name: {run.info.run_name}"
        )
        
        
        set_seed(SEED)

        # Get datasets
        train_ds = get_aro_dataset(
            data_path=TRAIN_DATA_PATH,
            dim=EMBEDDING_DIM,
            bond_dim=BOND_DIM,
            progress=True,
        )

        val_ds = get_aro_dataset(
            data_path=VAL_DATA_PATH,
            dim=EMBEDDING_DIM,
            bond_dim=BOND_DIM,
            progress=True,
        )

        test_ds = get_aro_dataset(
            data_path=TEST_DATA_PATH,
            dim=EMBEDDING_DIM,
            bond_dim=BOND_DIM,
            progress=True,
        )
        
        mlflow.log_params({"ds_size_train": len(train_ds), 
                           "ds_size_val": len(val_ds), 
                           "ds_size_test": len(test_ds)})


        collate_fn = aro_tn_collate_fn

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = get_einsum_model([train_ds, val_ds, test_ds]).to(DEVICE)
       
        number_params = sum(p.numel() for p in model.parameters())
        mlflow.log_params({"model_num_params": number_params})

        image_model = LookupEmbedding.load_from_checkpoint(IMAGE_LOOKUP_PATH)
        image_model = image_model.to(DEVICE)

        # Define optimizer and loss functions
        contrastive_loss = InfoNCE(temperature=TEMPERATURE)
        if HARD_NEG_DISTANCE_FUNCTION == "cosine":
            distance_function = lambda x, y: 1 - nn.CosineSimilarity(dim=-1)(x, y)
        elif HARD_NEG_DISTANCE_FUNCTION == "euclidean":
            distance_function = nn.PairwiseDistance(p=2)
        else:
            raise ValueError(f"Unknown distance function: {HARD_NEG_DISTANCE_FUNCTION}")
        hard_neg_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=distance_function,
            margin=HARD_NEG_MARGIN,
            swap=HARD_NEG_SWAP,
        )
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        best_val_hard_neg_loss = float("inf")

        for epoch in trange(1, EPOCHS + 1, desc="Training Epochs"):
            logger.info(f"Starting epoch {epoch}/{EPOCHS}")

            # Train
            train_metrics = train_epoch(
                model,
                image_model,
                train_loader,
                contrastive_loss,
                hard_neg_loss,
                optimizer,
                hard_neg_loss_weight=HARD_NEG_LOSS_WEIGHT,
                device=DEVICE,
            )
            mlflow.log_metrics(
                {f"train/{key}": value for key, value in train_metrics.items()},
                step=epoch,
            )

            # Evaluate
            val_metrics = evaluate_model(
                model,
                image_model,
                val_loader,
                contrastive_loss,
                hard_neg_loss,
                hard_neg_loss_weight=HARD_NEG_LOSS_WEIGHT,
                device=DEVICE,
            )

            mlflow.log_metrics(
                {f"val/{key}": value for key, value in val_metrics.items()}, step=epoch
            )
            logger.info(
                f"Epoch {epoch}/{EPOCHS} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['hard_neg_acc']:.4f}"
            )

            # Save best model checkpoint
            if val_metrics["hard_neg_loss"] < best_val_hard_neg_loss:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }
                checkpoint_path = os.path.join(
                    CHECKPOINT_PATH, f"{run.info.run_id}/best_model.pt"
                )
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                logger.info(f"Saved best model checkpoint to {checkpoint_path}")

        # Final evaluation on test set
        best_model_path = os.path.join(
            CHECKPOINT_PATH, f"{run.info.run_id}/best_model.pt"
        )
        best_checkpoint = torch.load(best_model_path, map_location=DEVICE)
        
        best_model = EinsumModel()
        best_model.load_state_dict(best_checkpoint["model_state_dict"])
        best_model = best_model.to(DEVICE)        

        test_metrics = evaluate_model(
            best_model,
            image_model,
            test_loader,
            contrastive_loss,
            hard_neg_loss,
            hard_neg_loss_weight=HARD_NEG_LOSS_WEIGHT,
            device=DEVICE,
        )
        mlflow.log_metrics(
            {f"test/{key}": value for key, value in test_metrics.items()}
        )
        logger.info(
            f"Testing on test set with best model from epoch {best_checkpoint['epoch']}"
        )
        logger.info(
            f"Test Loss: {test_metrics['loss']:.4f}, "
            f"Test Acc: {test_metrics['hard_neg_acc']:.4f}"
        )
        logger.info("Training complete.")
        mlflow.log_artifact(os.path.join(LOG_PATH, f"train_{run.info.run_id}.log"))


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


if __name__ == "__main__":

    # mlflow.set_tracking_uri(MLFLOW_URI)
    # mlflow.set_experiment(MLFLOW_EXPERIMENT)

    train_model()
