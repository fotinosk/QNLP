"""Config for training EinsumModel + TTNImageModel on the ARO train set.

Mirrors the v1 setup that originally produced 0.78 on ARO test:
  - Text: EinsumModel (trainable, free per-symbol tensors at bond_dim=10)
  - Image: TTNImageModel (trainable, same architecture as v1)
  - Data: visual_genome_{relation,attribution}/{train,val,test}.json
          with CCG-compiled sidecar *_processed_512.jsonl
  - Loss: ContrastiveLoss (InfoNCE + triplet) over the dataset's pre-computed
          word-order hard negatives.

Differs from einsum_frozen_clip: trains the image side too, uses ARO data
instead of COCO, smaller batch because total train set is only ~36k pairs.
"""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    embedding_dim: int = 512

    # --- Optimization ---
    batch_size: int = 256
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

    # --- Loss (matches aro_contrastive exactly) ---
    temperature: float = 0.07
    triplet_weight: float = 40000.0
    triplet_margin: float = 0.2
    distance: Literal["cosine", "euclidean"] = "cosine"

    # Which ARO splits to combine. "combined" uses both relation+attribution.
    # "relation" or "attribution" use a single benchmark only.
    aro_subset: Literal["combined", "relation", "attribution"] = "combined"

    model_config = SettingsConfigDict(env_prefix="ART_")
