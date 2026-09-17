from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class SVOExperimentConfig(BaseSettings):
    """
    Matches the legacy ARO training config (qnlp/scripts/aro_contrastive/config.py)
    rather than the newer COCO-style one (qnlp/scripts/coco_multi_caption/config.py).
    COCO is ~70x larger than SVO's training set and has no hard negatives at all —
    its plain in-batch-InfoNCE config is tuned for a very different regime. ARO is
    a small, hard-negative benchmark much closer to SVO-Probes in scale and shape.
    """

    embedding_dim: int = 512
    bond_dim: int = 10

    use_non_linear_contractions: bool = True

    batch_size: int = 128
    text_lr: float = 0.001
    text_weight_decay: float = 0.001
    image_lr: float = 0.00005
    image_weight_decay: float = 0.05
    head_lr: float = 0.001
    head_weight_decay: float = 0.001

    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0001
    max_grad_norm: float = 1.0

    temperature: float = 0.07  # fixed — not passed to the optimizer
    triplet_weight: float = 40000.0
    triplet_margin: float = 0.2
    distance: Literal["cosine", "euclidean"] = "cosine"

    use_mlp_head: bool = False

    model_config = SettingsConfigDict(env_prefix="SVO_ML_")
