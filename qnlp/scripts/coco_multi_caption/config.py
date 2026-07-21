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

    # Hard-negative π-sweep (see HARD_NEG_PI_SWEEP_PLAN.md Phase B). pi=0 (default)
    # must leave every code path bit-identical to no-hard-negatives training.
    hard_neg_pi: float = 0.0
    hard_negs_dataset: str | None = None  # base name, resolved as {name}_train.parquet
    hard_neg_softmax_temp: float = 0.5
    hard_neg_h_max: float = 0.95  # redundant with the build-time filter, kept as a knob

    model_config = SettingsConfigDict(env_prefix="ML_")
