import logging
import os
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
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

def parse_args():
    parser = ArgumentParser(
        description="Discoclip ARO Dataset Training Script",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    
    # Add config file argument
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--train-data-path",
        type=str,
        help="Path to the training data JSON file",
    )
    data_group.add_argument(
        "--val-data-path",
        type=str,
        help="Path to the validation data JSON file",
    )
    data_group.add_argument(
        "--test-data-path",
        type=str,
        help="Path to the test data JSON file",
    )
    data_group.add_argument(
        "--image-lookup-path",
        type=str,
        help="Path to the image lookup file",
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--embedding-dim",
        type=int,
        help="Dimension of the embeddings",
    )
    model_group.add_argument(
        "--bond-dim",
        type=int,
        help="Bond dimension for the MPS ansatz",
    )
    
    # Training arguments
    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training",
    )
    training_group.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for the optimizer",
    )
    training_group.add_argument(
        "--weight-decay",
        type=float,
        help="Weight decay for the optimizer",
    )
    training_group.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train the model",
    )
    training_group.add_argument(
        "--patience",
        type=int,
        help="Patience for early stopping",
    )
    
    # Loss arguments
    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument(
        "--temperature",
        type=float,
        help="Temperature for the InfoNCE loss",
    )
    loss_group.add_argument(
        "--hard-neg-loss-weight",
        type=float,
        help="Weight for the hard negative loss component",
    )
    loss_group.add_argument(
        "--hard-neg-margin",
        type=float,
        help="Margin for the hard negative loss",
    )
    loss_group.add_argument(
        "--hard-neg-distance-function",
        type=str,
        choices=["cosine", "euclidean"],
        help="Distance function for the hard negative loss",
    )
    loss_group.add_argument(
        "--hard-neg-swap",
        action="store_true",
        help="Whether to use the distance swap for the hard negative loss",
    )
    
    # Logging arguments
    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument(
        "--log-path",
        type=str,
        help="Path to save training logs",
    )
    logging_group.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to save model checkpoints",
    )
    logging_group.add_argument(
        "--mlflow-uri",
        type=str,
        help="URI for MLflow tracking server",
    )
    logging_group.add_argument(
        "--mlflow-experiment",
        type=str,
        help="Name of the MLflow experiment",
    )
    
    # System arguments
    system_group = parser.add_argument_group("System")
    system_group.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device to run the training on",
    )
    system_group.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # Update args with config values, but don't override command line args
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)

    # If the mlflow uri is not set
    if not args.mlflow_uri:
        args.mlflow_uri = f"sqlite:///{os.path.expanduser('~/mlflow/mlruns.db')}"
   
    print(f"Using MLflow URI: {args.mlflow_uri}") 
    return args



def get_aro_dataset(
    data_path: str, dim: int, bond_dim: int, progress: bool = True
) -> ARODataset:
    
    reader = READER
    cache_dir = os.path.join(os.path.dirname(data_path), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cache = Cache(cache_dir, size_limit=10 * 2**30) # 10GB
   
    cache_key = os.path.join(
        os.path.basename(data_path),
        f"{reader}_dim{dim}_bond{bond_dim}"
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
    hard_neg_loss_weight=0,
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
    hard_neg_criterion=None,
    hard_neg_loss_weight=0,
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
            hard_neg_loss = hard_neg_criterion(image_embeddings, 
                                             true_caption_embeddings, false_caption_embeddings)

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


def train_model(args, parent_run=None):
    with mlflow.start_run(parent_run_id=parent_run.info.run_id if parent_run else None,
                          nested=True if parent_run else False) as run:
        mlflow.log_params(vars(args))

        logger = setup_logger(
            os.path.join(args.log_path, f"train_{run.info.run_id}.log")
        )
        logger.info(
            f"Running experiment: {args.mlflow_experiment}, run ID: {run.info.run_id}, run name: {run.info.run_name}"
        )
        
        logger.info(f"Args: {vars(args)}")
        
        set_seed(args.seed)

        # Get datasets
        train_ds = get_aro_dataset(
            data_path=args.train_data_path,
            dim=args.embedding_dim,
            bond_dim=args.bond_dim,
            progress=True,
        )

        val_ds = get_aro_dataset(
            data_path=args.val_data_path,
            dim=args.embedding_dim,
            bond_dim=args.bond_dim,
            progress=True,
        )

        test_ds = get_aro_dataset(
            data_path=args.test_data_path,
            dim=args.embedding_dim,
            bond_dim=args.bond_dim,
            progress=True,
        )
        
        mlflow.log_params({"ds_size_train": len(train_ds), 
                           "ds_size_val": len(val_ds), 
                           "ds_size_test": len(test_ds)})


        collate_fn = aro_tn_collate_fn

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = get_einsum_model([train_ds, val_ds, test_ds]).to(args.device)
       
        number_params = sum(p.numel() for p in model.parameters())
        mlflow.log_params({"model_num_params": number_params})

        image_model = LookupEmbedding.load_from_checkpoint(args.image_lookup_path)
        image_model = image_model.to(args.device)

        # Define optimizer and loss functions
        contrastive_loss = InfoNCE(temperature=args.temperature)
        if args.hard_neg_distance_function == "cosine":
            distance_function = lambda x, y: 1 - nn.CosineSimilarity(dim=-1)(x, y)
        elif args.hard_neg_distance_function == "euclidean":
            distance_function = nn.PairwiseDistance(p=2)
        else:
            raise ValueError(f"Unknown distance function: {args.hard_neg_distance_function}")
        hard_neg_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=distance_function,
            margin=args.hard_neg_margin,
            swap=args.hard_neg_swap,
        )
        optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        best_val_hard_neg_loss = float("inf")

        for epoch in trange(1, args.epochs + 1, desc="Training Epochs"):
            logger.info(f"Starting epoch {epoch}/{args.epochs}")

            # Train
            train_metrics = train_epoch(
                model,
                image_model,
                train_loader,
                contrastive_loss,
                hard_neg_loss,
                optimizer,
                hard_neg_loss_weight=args.hard_neg_loss_weight,
                device=args.device,
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
                hard_neg_loss_weight=args.hard_neg_loss_weight,
                device=args.device,
            )

            mlflow.log_metrics(
                {f"val/{key}": value for key, value in val_metrics.items()}, step=epoch
            )
            logger.info(
                f"Epoch {epoch}/{args.epochs} - "
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
                    "args": vars(args),
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
        best_checkpoint = torch.load(best_model_path, map_location=args.device)
        
        best_model = EinsumModel()
        best_model.load_state_dict(best_checkpoint["model_state_dict"])
        best_model = best_model.to(args.device)        

        test_metrics = evaluate_model(
            best_model,
            image_model,
            test_loader,
            contrastive_loss,
            hard_neg_loss,
            hard_neg_loss_weight=args.hard_neg_loss_weight,
            device=args.device,
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
        mlflow.log_artifact(os.path.join(args.log_path, f"train_{run.info.run_id}.log"))


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
    args = parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    train_model(args)
