"""einsum_frozen_clip + inverse-log-frequency gradient scaling per typed symbol.

Each typed symbol's gradient is multiplied by a backward hook so that
rare symbols (seen ≤ 2× in train) get larger updates per step than
common symbols (seen 1000+×). This compensates for the unequal training
exposure that the diagnostic identified.

Scale formula:
    scale[sym] = 1.0 / log(freq[sym] + 2)
Examples:
  freq=0   → 1.44   (1.0/log(2))
  freq=1   → 0.91   (1.0/log(3))
  freq=100 → 0.22   (1.0/log(102))
  freq=10000 → 0.11
"""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    embedding_dim: int = 512
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

    model_config = SettingsConfigDict(env_prefix="EFCG_")
