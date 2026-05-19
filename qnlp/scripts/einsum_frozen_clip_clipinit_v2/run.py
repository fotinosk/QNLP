"""einsum_frozen_clip with per-shape magnitude-calibrated CLIP-text init.

Key improvement over v1: for each typed symbol of shape S, the bond
factors are sized so that the reconstructed outer-product tensor has
element-wise std matching what the original EinsumModel would produce
for that shape via its `bound = 1/mean(directed_cod)` formula.

This preserves:
  - the semantic info from the CLIP text encoding on the embedding axis
  - the per-shape numerical regime that prevents 13-step cotengra
    contraction underflow / overflow

Construction for a shape with embedding axis at position `emb_axis`:
  1. emb_vec = CLIP_text_encode(lemma)   # 512-d, L2-normalised (elem std ~0.044)
  2. target_elem_std = original_einsum_bound(directed_cod_of_shape) / sqrt(3)
  3. emb_factor_std ≈ 0.044  (from CLIP)
  4. bond_factor_std = (target_elem_std / emb_factor_std) ** (1 / n_bond_axes)
  5. reconstructed[bond_indices..., emb_idx, bond_indices...] =
       product of bond factors × emb_vec[emb_idx]
"""
from datetime import datetime
import re

import mlflow
import torch
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from qnlp.constants import constants
from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.domain.datasets.dataloader import get_dataloaders
from qnlp.domain.datasets.dataset import collect_symbol_sizes
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.aro_contrastive.step import AROContrastiveStep
from qnlp.scripts.einsum_frozen_clip.image_model import FrozenClipVisionModel
from qnlp.scripts.einsum_frozen_clip_clipinit_v2.config import ExperimentConfig
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "einsum_frozen_clip_clipinit_v2"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path
TRAIN_PARQUET = DATASETS_PATH / "coco_contrastive_train.parquet"
VAL_PARQUET = DATASETS_PATH / "coco_contrastive_val.parquet"
TEST_PARQUET = DATASETS_PATH / "coco_contrastive_test.parquet"

COMPILED_COLUMNS = [
    ("true_diagram", "true_symbols", "true_caption"),
    ("false_diagram", "false_symbols", "false_caption"),
]
SYMBOL_COLS = ["true_symbols", "false_symbols"]

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_NAME_RE = re.compile(r"^(.+)_(\d+)__(.+)$")


def parse_lemma(name: str) -> str:
    m = _NAME_RE.match(name)
    if m is None:
        return name
    return m.group(1)


def original_einsum_bound(directed_cod: int) -> float:
    """Replicate EinsumModel.reset_parameters' `bound = 1/mean(cod)` formula."""
    size = directed_cod
    if size < 6:
        correction_factor = [0, 3, 2.6, 2, 1.6, 1.3][size]
    else:
        correction_factor = 1.0 / (0.16 * size - 0.04)
    mean_val = (size / 3.0 - 1.0 / (15.0 - correction_factor)) ** 0.5
    return 1.0 / mean_val


def compute_clip_lemma_embeddings(lemmas, clip_model_name, device):
    logger.info(f"loading CLIP text encoder ({clip_model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    model = CLIPTextModelWithProjection.from_pretrained(clip_model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    emb = {}
    batch_size = 256
    for start in range(0, len(lemmas), batch_size):
        chunk = lemmas[start:start + batch_size]
        tok = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=8).to(device)
        with torch.no_grad():
            out = model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        for w, v in zip(chunk, out.text_embeds.cpu()):
            emb[w] = v
    del model
    return emb


def init_einsum_with_calibrated_clip(text_model: EinsumModel, lemma_emb: dict, embedding_dim: int = 512):
    """Override each typed tensor with magnitude-calibrated CLIP-text init.

    For each tensor of shape S with single embedding-dim axis:
      - emb factor: clip_emb[lemma]  (std ~ 1/sqrt(D))
      - bond factor std chosen so the reconstructed elem std matches
        original_einsum_bound(directed_cod) / sqrt(3)
    """
    n_init = n_missing = n_skipped = 0
    emb_elem_std = 1.0 / embedding_dim ** 0.5  # ~0.044 for D=512

    with torch.no_grad():
        for sym, param in zip(text_model.symbols, text_model.weights):
            lemma = parse_lemma(sym.name)
            if lemma not in lemma_emb:
                n_missing += 1
                continue
            emb = lemma_emb[lemma]  # (D,)
            size = tuple(param.shape)
            emb_axes = [i for i, d in enumerate(size) if d == embedding_dim]
            if len(emb_axes) != 1:
                if len(size) == 1 and size[0] == embedding_dim:
                    # Pure rank-1, no bond axes, just rescale emb to match orig init.
                    target_norm = (embedding_dim ** 0.5) * (original_einsum_bound(sym.directed_cod) / (3 ** 0.5))
                    emb_scaled = emb / emb.norm().clamp(min=1e-12) * target_norm
                    param.copy_(emb_scaled)
                    n_init += 1
                else:
                    n_skipped += 1
                continue
            emb_axis = emb_axes[0]
            bond_dims_left = list(size[:emb_axis])
            bond_dims_right = list(size[emb_axis + 1:])
            n_bond_axes = len(bond_dims_left) + len(bond_dims_right)

            # Target elem std = original init's elem std for this shape
            bound = original_einsum_bound(sym.directed_cod)
            target_elem_std = bound / (3 ** 0.5)
            # bond_factor_std s.t. emb_elem_std * bond_factor_std^n_bond_axes = target_elem_std
            bond_factor_std = (target_elem_std / emb_elem_std) ** (1.0 / n_bond_axes)

            # Build factors
            bl = torch.randn(*bond_dims_left) * bond_factor_std if bond_dims_left else None
            br = torch.randn(*bond_dims_right) * bond_factor_std if bond_dims_right else None

            # Reconstruct via outer product to target shape
            target = emb.clone()  # (D,)
            # First prepend left-bond dims (will reshape correctly).
            if bl is not None:
                # outer: bl ⊗ emb → shape (b1,..,bk, D)
                target = torch.einsum("b,d->bd", bl.reshape(-1), target)
                target = target.reshape(*bond_dims_left, embedding_dim)
            if br is not None:
                target = torch.einsum("...,r->...r", target, br.reshape(-1))
                target = target.reshape(*bond_dims_left, embedding_dim, *bond_dims_right)

            assert target.shape == size, f"shape mismatch: {target.shape} vs {size} for {sym.name}"
            param.copy_(target)
            n_init += 1

    logger.info(f"CLIP-init v2 done: {n_init} initialised, {n_missing} missing lemmas, {n_skipped} skipped (weird shape).")


def run():
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()

    size = cfg.clip_image_size
    train_transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    loaders, datasets = get_dataloaders(
        train_parquet=TRAIN_PARQUET, val_parquet=VAL_PARQUET, test_parquet=TEST_PARQUET,
        batch_size=cfg.batch_size,
        train_transform=train_transform, val_transform=val_transform,
        compiled_columns=COMPILED_COLUMNS,
    )
    train_loader, val_loader, test_loader = loaders
    train_ds, val_ds, test_ds = datasets
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    symbols, sizes = collect_symbol_sizes([train_ds, val_ds, test_ds], SYMBOL_COLS)
    logger.info(f"Collected {len(symbols)} unique typed symbols.")

    lemmas = sorted({parse_lemma(s.name) for s in symbols})
    logger.info(f"Unique lemmas: {len(lemmas)}")
    lemma_emb = compute_clip_lemma_embeddings(lemmas, cfg.clip_model_name, device)

    text_model = EinsumModel(symbols, sizes)
    init_einsum_with_calibrated_clip(text_model, lemma_emb, embedding_dim=cfg.embedding_dim)
    text_model = text_model.to(device)
    image_model = FrozenClipVisionModel(cfg.clip_model_name, cfg.embedding_dim).to(device)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)

    logger.info(f"Text params: {sum(p.numel() for p in text_model.parameters()):,}")

    loss_fn = ContrastiveLoss(
        temperature=cfg.temperature, triplet_weight=cfg.triplet_weight,
        triplet_margin=cfg.triplet_margin, distance=cfg.distance,
    ).to(device)
    step = AROContrastiveStep(loss_fn=loss_fn, device=device)

    optimizer = torch.optim.AdamW([
        {"params": text_model.parameters(), "lr": cfg.text_lr, "weight_decay": cfg.text_weight_decay},
        {"params": image_model.proj.parameters(), "lr": cfg.image_lr, "weight_decay": cfg.image_weight_decay},
        {"params": list(model.image_head.parameters()) + list(model.text_head.parameters()),
         "lr": cfg.head_lr, "weight_decay": cfg.head_weight_decay},
    ])

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = constants.checkpoints_path / EXPERIMENT_NAME / ts / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with setup_mlflow_run(EXPERIMENT_NAME, cfg.model_dump(), 8080) as run:
        trainer = Trainer(
            model=model, optimizer=optimizer, step=step,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            monitor_metric="hard_neg_acc", checkpoint_path=checkpoint_path,
            max_epochs=cfg.max_epochs, patience=cfg.patience,
            min_delta=cfg.min_delta, max_grad_norm=cfg.max_grad_norm, device=device,
        )
        test_metrics = trainer.fit()
        mlflow.log_artifact(checkpoint_path)
        send_training_finished_notification(
            {"experiment": EXPERIMENT_NAME, "run": run.info.run_name, **test_metrics}
        )


if __name__ == "__main__":
    run()
