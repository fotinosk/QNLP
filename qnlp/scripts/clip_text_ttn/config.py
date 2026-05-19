"""Config for the frozen CLIP-text + trainable TTN experiment.

Inverse of einsum_frozen_clip: instead of freezing the image side and training
the EinsumModel text side, here we freeze the text side and train the TTN
image side. Tests whether the TTNImageModel from v1 can learn an alignment
at full COCO scale given a perfect (CLIP-aligned) text signal.

Data is `coco_single_caption_*.parquet` (raw captions) because the CLIP text
encoder operates on text, not CCG diagrams. Synthetic negatives are
generated in-batch via random derangement at training time, matching
clip_baseline.
"""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    embedding_dim: int = 512

    # --- Frozen CLIP text encoder ---
    clip_model_name: str = "openai/clip-vit-base-patch32"
    text_max_len: int = 77  # CLIP's native context length

    # --- Optimization ---
    # bs=256 to match clip_baseline (we know it fits in 16 GB VRAM there).
    # The trainable TTN is small but image processing dominates memory.
    batch_size: int = 256
    text_lr: float = 1e-4
    text_weight_decay: float = 0.01
    image_lr: float = 1e-4
    image_weight_decay: float = 0.01
    head_lr: float = 1e-4
    head_weight_decay: float = 0.01

    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0001
    max_grad_norm: float = 1.0

    # --- Loss (matches clip_baseline / aro_contrastive) ---
    temperature: float = 0.07
    triplet_weight: float = 40000.0
    triplet_margin: float = 0.2
    distance: Literal["cosine", "euclidean"] = "cosine"

    model_config = SettingsConfigDict(env_prefix="CTT_")
