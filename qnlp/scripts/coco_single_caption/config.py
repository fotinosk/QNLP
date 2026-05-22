from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    embedding_dim: int = 256
    bond_dim: int = 20

    batch_size: int = 512
    text_lr: float = 0.001
    text_weight_decay: float = 0.001
    image_lr: float = 0.0002
    image_weight_decay: float = 0.05

    max_epochs: int = 20
    patience: int = 5
    min_delta: float = 0.0001
    max_grad_norm: float = 1.0

    head_lr: float = 0.001
    head_weight_decay: float = 0.001

    temperature: float = 0.07
    alignment_weight: float = 0.5
    alignment_warmup_epochs: int = 5

    model_config = SettingsConfigDict(env_prefix="ML_")
