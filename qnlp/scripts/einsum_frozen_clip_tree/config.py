"""Config for the EinsumModel + frozen-CLIP-ViT experiment, TreeReader
variant.

Same as einsum_frozen_clip.config but with a distinct env-prefix so the
experiments don't collide in env-driven hyper-param sweeps.
"""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    embedding_dim: int = 512

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_image_size: int = 224

    batch_size: int = 1024
    text_lr: float = 0.001
    text_weight_decay: float = 0.1
    image_lr: float = 0.0001
    image_weight_decay: float = 0.01
    head_lr: float = 0.001
    head_weight_decay: float = 0.001

    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0001
    max_grad_norm: float = 1.0

    temperature: float = 0.07
    triplet_weight: float = 40000.0
    triplet_margin: float = 0.2
    distance: Literal["cosine", "euclidean"] = "cosine"

    model_config = SettingsConfigDict(env_prefix="EFCT_")
