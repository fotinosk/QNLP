from pydantic_settings import BaseSettings, SettingsConfigDict


class ExperimentConfig(BaseSettings):
    embedding_dim: int = 256

    # Use non-linear pairwise contractions in the text model. Requires the dataset
    # to have been created with compute_contraction_paths=True.
    use_non_linear_contractions: bool = True

    batch_size: int = 32
    text_lr: float = 0.0005
    text_weight_decay: float = 0.001
    image_lr: float = 0.00002
    image_weight_decay: float = 0.05
    head_lr: float = 0.0005
    head_weight_decay: float = 0.001

    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0001
    max_grad_norm: float = 1.0

    temperature: float = 0.07
    triplet_weight: float = 40000.0
    triplet_margin: float = 0.2
    distance: str = "cosine"

    # Path to a pretrained checkpoint to fine-tune from (e.g. COCO or ARO).
    # Image model + heads load fully; text model loads overlapping symbols.
    pretrained_checkpoint: str | None = None

    model_config = SettingsConfigDict(env_prefix="ML_")
