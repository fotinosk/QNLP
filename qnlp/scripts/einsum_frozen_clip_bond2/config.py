"""Config for einsum_frozen_clip with bond_dim=4 (vs the default 10).

Identical to einsum_frozen_clip except `bond_dim`, which controls the size
the per-typed-symbol tensors are allocated at. The CCG diagrams + symbol
identities on disk don't depend on bond_dim, only the sizes do — so we
remap sizes (10 → bond_dim) at model construction time without
re-running the BERT preprocessing pipeline.

Lower bond_dim → ~6× fewer params per typed symbol (e.g. (4, 512, 4) =
8 192 elems vs (10, 512, 10) = 51 200). Hypothesis: reducing per-symbol
capacity reduces the memorisation route for the long-tail symbols and
may improve generalisation.
"""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    embedding_dim: int = 512
    bond_dim: int = 2              # ← the experimental knob
    original_bond_dim: int = 10    # bond_dim baked into the compiled parquets

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

    model_config = SettingsConfigDict(env_prefix="EFC2_")
