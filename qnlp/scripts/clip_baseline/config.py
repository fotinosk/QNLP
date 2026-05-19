"""Config for the CLIP-from-scratch baseline.

Mirrors aro_contrastive/config.py so the comparison stays apples-to-apples on
loss/optimizer/early-stopping. The two encoder configs (text and image) live
here too because they're picked specifically to roughly match the EinsumModel +
TTNImageModel parameter budget.
"""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    # --- Shared embedding dim (must match AlignmentHead size) ---
    embedding_dim: int = 512

    # --- Text encoder (small BERT, no pretrained weights) ---
    text_hidden: int = 256
    text_layers: int = 4
    text_heads: int = 4
    text_max_len: int = 64

    # --- Image encoder ---
    # torchvision resnet18 with random init; final fc → embedding_dim

    # --- Training ---
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

    # --- Loss (matches aro_contrastive defaults exactly) ---
    temperature: float = 0.07
    triplet_weight: float = 40000.0
    triplet_margin: float = 0.2
    distance: Literal["cosine", "euclidean"] = "cosine"

    model_config = SettingsConfigDict(env_prefix="CLIP_")
