"""Config for einsum_frozen_clip with CLIP-text-encoded lemma initialisation.

Same as einsum_frozen_clip except every typed symbol's embedding-axis is
initialised from the frozen CLIP text encoder's embedding of that
symbol's lemma (e.g. CLIP("dog") → 512-d). Bond axes get standard
random init. The hope: the model starts in a semantically meaningful
location and only needs to learn small deviations.
"""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    embedding_dim: int = 512
    bond_dim: int = 10

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_image_size: int = 224

    batch_size: int = 1024
    text_lr: float = 0.001
    text_weight_decay: float = 0.001
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

    # Init magnitude for the non-embedding (bond) axes.
    bond_init_std: float = 0.5

    model_config = SettingsConfigDict(env_prefix="EFCC_")
