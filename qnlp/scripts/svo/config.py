from pydantic_settings import BaseSettings, SettingsConfigDict


class SVOExperimentConfig(BaseSettings):
    embedding_dim: int = 256
    bond_dim: int = 10

    use_non_linear_contractions: bool = True

    batch_size: int = 128
    text_lr: float = 0.003
    text_weight_decay: float = 0.001
    image_lr: float = 0.0002
    image_weight_decay: float = 0.05

    max_epochs: int = 50
    patience: int = 10
    min_delta: float = 0.0001
    max_grad_norm: float = 1.0

    head_lr: float = 0.001
    head_weight_decay: float = 0.001

    temperature: float = 0.07  # fixed — not passed to the optimizer
    use_mlp_head: bool = False

    # Weight of the per-sample cosine alignment loss relative to InfoNCE.
    # 0.0 = InfoNCE only (COCO's canonical setting, and this file's prior
    # default). A *warmup* schedule for this (weight>0 for a few epochs then
    # dropped) caused embedding collapse in the COCO campaign — a constant
    # weight held throughout training is a different, untested experiment.
    alignment_weight: float = 0.0

    model_config = SettingsConfigDict(env_prefix="SVO_ML_")
