"""einsum_frozen_clip + CLIP-text-init for the per-lemma embedding axis.

The diagnostic in llm/coco_overfitting_diagnosis_2026-05-19.md showed
that EinsumModel's overfitting on COCO is NOT primarily from the
vocabulary long tail — even val captions whose every symbol is well-
trained score at chance. The hypothesis tested here: if we start each
typed tensor's embedding axis at a semantically meaningful location
(the CLIP text encoding of its lemma), the model has a useful prior
and the overfitting may relax.

Construction:
  For each unique lemma w in the symbol set, query the frozen CLIP
  text encoder: clip_emb[w] = CLIPTextModelWithProjection(tokenize(w)).
  Then for each typed symbol of shape S, initialise the typed tensor as
    T[i, j, k] = bond_left[i] * clip_emb[w][j_emb] * bond_right[k]
  where the embedding axis (size 512) gets clip_emb[w], and bond
  axes (size 10) get standard Gaussian init of std=bond_init_std.

  The typed tensor is still a FREE nn.Parameter — gradient descent can
  move it anywhere during training. The CLIP init is just a smarter
  starting point.
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
from qnlp.scripts.einsum_frozen_clip_clipinit.config import ExperimentConfig
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "einsum_frozen_clip_clipinit"
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
        return name  # fallback
    return m.group(1)


def compute_clip_lemma_embeddings(lemmas: list[str], clip_model_name: str, device) -> dict:
    """Return {lemma: torch.Tensor (512,)} from frozen CLIP text encoder."""
    logger.info(f"loading frozen CLIP text encoder ({clip_model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    model = CLIPTextModelWithProjection.from_pretrained(clip_model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    emb: dict[str, torch.Tensor] = {}
    batch_size = 256
    logger.info(f"encoding {len(lemmas)} lemmas in batches of {batch_size}...")
    for start in range(0, len(lemmas), batch_size):
        chunk = lemmas[start:start + batch_size]
        tok = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=8).to(device)
        with torch.no_grad():
            out = model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        vec = out.text_embeds  # (B, 512)
        for w, v in zip(chunk, vec.cpu()):
            emb[w] = v
        if (start // batch_size) % 20 == 0:
            logger.info(f"  encoded {start + len(chunk)}/{len(lemmas)}")
    del model
    return emb


def init_einsum_with_clip(text_model: EinsumModel, lemma_emb: dict, bond_init_std: float):
    """Override each typed tensor's init using the CLIP lemma embedding."""
    embedding_dim = 512
    n_missing = 0
    n_init = 0
    with torch.no_grad():
        for sym, param in zip(text_model.symbols, text_model.weights):
            lemma = parse_lemma(sym.name)
            if lemma not in lemma_emb:
                n_missing += 1
                continue
            emb = lemma_emb[lemma]  # (512,)
            size = tuple(param.shape)
            # Identify embedding axis: the one with size embedding_dim
            emb_axes = [i for i, d in enumerate(size) if d == embedding_dim]
            if len(emb_axes) != 1:
                # rare; e.g. (512,) is rank-1 with single axis
                if len(size) == 1 and size[0] == embedding_dim:
                    param.copy_(emb)
                    n_init += 1
                continue
            emb_axis = emb_axes[0]
            # Build outer product: bond_left ⊗ emb ⊗ bond_right
            left_dims = list(size[:emb_axis])
            right_dims = list(size[emb_axis + 1:])
            bl = torch.randn(*left_dims).normal_(0, bond_init_std) if left_dims else None
            br = torch.randn(*right_dims).normal_(0, bond_init_std) if right_dims else None
            # Construct tensor
            new = emb.clone()  # (D,)
            if bl is not None:
                # broadcast: new[b1, ..., bk, d] = bl[b1, ..., bk] * emb[d]
                shape_with_emb = list(left_dims) + [embedding_dim]
                bl_expanded = bl.unsqueeze(-1)  # (left..., 1)
                emb_expanded = emb.view(*([1] * len(left_dims)), embedding_dim)
                new = bl_expanded * emb_expanded
            if br is not None:
                # append right bond dims
                new = new.unsqueeze(-1) * br.view(*([1] * (new.ndim)), *right_dims)
                # actually need more careful: ensure broadcasting matches
                # let's just rebuild explicitly
                target_shape = tuple(left_dims + [embedding_dim] + right_dims)
                if new.shape != target_shape:
                    # rebuild with einsum
                    if bl is None:
                        new = torch.einsum("d,...->d...", emb, br)
                    else:
                        new = torch.einsum("a,d,b->adb", bl.reshape(-1), emb, br.reshape(-1))
                        new = new.reshape(target_shape)
            assert new.shape == size, f"shape mismatch: {new.shape} vs {size} for {sym.name}"
            param.copy_(new)
            n_init += 1
    logger.info(f"CLIP-init done: {n_init} initialised, {n_missing} missing lemmas (random init).")


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

    # Unique lemmas
    lemmas = sorted({parse_lemma(s.name) for s in symbols})
    logger.info(f"Unique lemmas: {len(lemmas)} (of {len(symbols)} typed symbols)")
    lemma_emb = compute_clip_lemma_embeddings(lemmas, cfg.clip_model_name, device)

    text_model = EinsumModel(symbols, sizes)
    logger.info(f"Initialising EinsumModel with CLIP-text lemma embeddings...")
    init_einsum_with_clip(text_model, lemma_emb, cfg.bond_init_std)
    text_model = text_model.to(device)

    image_model = FrozenClipVisionModel(cfg.clip_model_name, cfg.embedding_dim).to(device)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)

    n_text = sum(p.numel() for p in text_model.parameters())
    n_image = sum(p.numel() for p in image_model.parameters() if p.requires_grad)
    logger.info(f"Text params: {n_text:,} | Image trainable: {n_image:,}")

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

    params = {**cfg.model_dump(), "text_model_params": n_text}

    with setup_mlflow_run(EXPERIMENT_NAME, params, 8080) as run:
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
