"""Config for the rank-1 lemma-tied EinsumModel + frozen CLIP-ViT experiment.

Mirrors `einsum_frozen_clip` exactly except that the text model is
`LemmaTiedEinsumModel`. The point of keeping every other knob identical
(data, image side, loss, batch size, all weight-decays) is to isolate
the effect of the lemma-tying refactor itself.

If val rises meaningfully above the einsum_frozen_clip ceiling of ~0.53,
the rank-1 lemma-tied parameterisation is doing real work.
"""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    # --- Shared embedding dim (AlignmentHead size; also lemma_emb width) ---
    embedding_dim: int = 512
    bond_dim: int = 10  # must match the bond dim used by the CCG compiler

    # --- CP-decomposition rank ---
    # R=1 collapses every typed variant of a lemma to a single direction —
    # empirically failed to fit train. R=4 gives each typed variant a learned
    # combination of 4 lemma directions. Bump to 8 or 16 if val still stuck.
    rank: int = 4

    # --- Frozen image encoder (same as einsum_frozen_clip) ---
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_image_size: int = 224

    # --- Optimization ---
    batch_size: int = 1024
    text_lr: float = 0.001
    # Much milder default than einsum_frozen_clip's 0.1: with ~13 M tied params
    # vs ~1.04 B free params, each gradient is shared across many more
    # occurrences, so we can run the canonical aro_contrastive WD.
    text_weight_decay: float = 0.001
    image_lr: float = 0.0001
    image_weight_decay: float = 0.01
    head_lr: float = 0.001
    head_weight_decay: float = 0.001

    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0001
    max_grad_norm: float = 1.0

    # --- Loss (matches aro_contrastive exactly) ---
    temperature: float = 0.07
    triplet_weight: float = 40000.0
    triplet_margin: float = 0.2
    distance: Literal["cosine", "euclidean"] = "cosine"

    model_config = SettingsConfigDict(env_prefix="LTE_")
