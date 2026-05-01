from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    embedding_dim: int = 512

    batch_size: int = 512
    text_lr: float = 0.001
    text_weight_decay: float = 0.001
    image_lr: float = 0.0002
    image_weight_decay: float = 0.05

    max_epochs: int = 100
    patience: int = 20
    min_delta: float = 0.0001
    max_grad_norm: float = 1.0

    head_lr: float = 0.001
    head_weight_decay: float = 0.001

    temperature: float = 0.07
    alignment_weight: float = 0.0

    model_config = SettingsConfigDict(env_prefix="ML_")
