"""einsum_frozen_clip + CLIP-text-init with PER-SHAPE bond magnitude calibration.

v1 of CLIP-init (qnlp/scripts/einsum_frozen_clip_clipinit/) NaN'd because
it used a uniform bond_init_std=0.5 across all typed-symbol shapes,
while the original EinsumModel uses very different per-shape init
magnitudes (driven by `bound = 1/mean(cod)`).

v2 fixes this: for each typed symbol, the bond factors are scaled so
that the *reconstructed* outer-product tensor has element std matching
what the original EinsumModel would produce for that shape.
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

    model_config = SettingsConfigDict(env_prefix="EFCC2_")
