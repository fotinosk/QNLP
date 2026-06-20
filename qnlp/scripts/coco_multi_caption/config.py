from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    embedding_dim: int = 256
    bond_dim: int = 10

    use_non_linear_contractions: bool = True
    dataset_name: str | None = None

    batch_size: int = 512
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

    model_config = SettingsConfigDict(env_prefix="ML_")
