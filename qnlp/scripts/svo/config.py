from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class SVOExperimentConfig(BaseSettings):
    """
    Matches the TRUE legacy ARO training setup — qnlp/discoviz/trainers/unfrozen/
    train_aro_clean.py, which trains and evaluates directly on ARO's own data and
    got the documented 78% hard_neg_accuracy — not qnlp/scripts/aro_contrastive/'s
    current config, which drifted from it (see comments below). COCO is ~70x larger
    than SVO's training set and has no hard negatives at all, so its plain
    in-batch-InfoNCE config was never a good reference point either way.
    """

    embedding_dim: int = 512
    bond_dim: int = 10

    # False, not True — train_aro_clean.py (the true legacy script, confirmed
    # via git archaeology) predates NLC entirely. NLC was added much later, in
    # aro_contrastive/ (the ported/refactored pipeline), and was never part of
    # the validated 78% hard_neg_accuracy run.
    use_non_linear_contractions: bool = False

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
    # False — train_aro_clean.py has NO learnable projection head at all: raw
    # image_model(images)/text_model(captions) outputs go straight into the
    # loss. AlignmentHead/head_lr/head_weight_decay were added later, in
    # aro_contrastive/, after the validated 78% run. See
    # qnlp/domain/models/vlm/contrastive_vlm.py::NoOpHead.
    use_alignment_head: bool = False

    model_config = SettingsConfigDict(env_prefix="SVO_ML_")
