"""Config for the EinsumModel + frozen-CLIP-ViT experiment.

Isolates EinsumModel's compositional capacity by removing the image-side
training signal entirely: a pretrained CLIP-ViT (frozen) provides image
embeddings, and only the text-side and a small projection head learn.

If EinsumModel can express a faithful alignment given a perfect image
signal, this run will fit. If it still fails, the bottleneck is the
text-side architecture rather than joint-training dynamics.
"""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    # --- Shared embedding dim (AlignmentHead size) ---
    embedding_dim: int = 512

    # --- Frozen image encoder ---
    # HuggingFace CLIP variant. Common options:
    #   "openai/clip-vit-base-patch32"  hidden=768, image_size=224  (default)
    #   "openai/clip-vit-base-patch16"  hidden=768, image_size=224
    #   "openai/clip-vit-large-patch14" hidden=1024, image_size=224
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_image_size: int = 224

    # --- Optimization ---
    batch_size: int = 1024
    text_lr: float = 0.001
    text_weight_decay: float = 0.1  # 100× the inherited aro_contrastive value; matches OpenAI CLIP default
    image_lr: float = 0.0001  # for the trainable Linear on top of frozen CLIP
    image_weight_decay: float = 0.01
    head_lr: float = 0.001
    head_weight_decay: float = 0.001

    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0001
    max_grad_norm: float = 1.0

    # --- Loss (matches aro_contrastive exactly so the comparison is apples-to-apples) ---
    temperature: float = 0.07
    triplet_weight: float = 40000.0
    triplet_margin: float = 0.2
    distance: Literal["cosine", "euclidean"] = "cosine"

    model_config = SettingsConfigDict(env_prefix="EFC_")
