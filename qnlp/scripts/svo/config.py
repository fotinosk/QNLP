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

    # Optional path to an aro_contrastive checkpoint (same architecture, same
    # embedding_dim/bond_dim) to warm-start from, instead of training from
    # random init. Motivation: ARO's own training set (36,585 rows) is ~4x
    # SVO's (~8,600), and the same loss/step design reproduces the documented
    # legacy ARO result almost exactly (see SVO_EXPERIMENTS.md's "ARO sanity
    # check") — so a model that has already learned ARO's hard-negative
    # image/text discrimination may need much less of SVO's small training
    # set to adapt, versus learning discrimination from scratch. The
    # image_model transfers in full (identical architecture/shapes). The
    # text_model (EinsumModel) is per-symbol/per-word, so only symbols
    # (words, with matching CCG type/shape) that appear in both ARO's and
    # SVO's vocabularies transfer; the rest stay randomly initialised. None
    # disables this and trains from scratch as before.
    pretrained_checkpoint: str | None = None

    # Routes A/B (TTN_CIFAR_EXPERIMENTS.md's "Implementation spec: Routes N,
    # B, A", then "Route B variations"). "cosine" (default) reproduces every
    # existing run bit-for-bit — image_head/text_head/ImageContrastiveLoss,
    # no score head at all. All others need P1 (TTNImageModel.forward_regions)
    # and route through StructuredContrastiveLoss instead:
    #   trilinear      Route B (B1) — diagnosed as memorising an absolute
    #                  region index; SVO-Probes 0.5168, below the 0.5323
    #                  cosine baseline. Kept for reference/re-runs.
    #   trilinear_gated  V1 — cosine(t, pooled(R)) + gate * trilinear(t, R),
    #                  gate init 0.0.
    #   born           V2 — sum_j |<t, R_j>|^2, permutation-invariant.
    #   aggregation    V3 — max/LSE region aggregation, drops the region-
    #                  indexed factor entirely. See `aggregation_fn`.
    #   role_grounded  Route A — needs subj/verb/obj columns (added to
    #                  svo_{train,val,test}_probes.parquet by prepare_datasets.py).
    score_head: Literal["cosine", "trilinear", "trilinear_gated", "born", "aggregation", "role_grounded"] = "cosine"
    region_level: int = 1  # P1: 4**region_level regions, root-indexed
    score_dim: int = 128
    rank: int = 32  # CP rank (trilinear / trilinear_gated / aggregation)
    aggregation_fn: Literal["max", "lse"] = "max"  # V3 only

    # V4 (capacity reduction / bounded terms) — applies to score_head=trilinear
    # or trilinear_gated. tie_uv shares one CP factor between text/region
    # projections instead of learning them independently; normalize_terms
    # L2-normalises t and each R_j before scoring so every term is bounded.
    tie_uv: bool = False
    normalize_terms: bool = False

    # F1/F2 (TTN_CIFAR_EXPERIMENTS.md's "Route B variations" — Orthogonal:
    # freeze a CIFAR-pretrained image tower). Path to a checkpoint saved by
    # qnlp/discoviz/diagnostic/ttn_supervised_probe.py (a real,
    # image-grounded TTNImageModel backbone, not random init). None trains
    # the image tower from scratch as before. freeze_pretrained_image_tower
    # controls whether it's frozen (F1, isolates capacity) or warm-started
    # and left trainable (F2, isolates initialisation).
    pretrained_image_tower_checkpoint: str | None = None
    freeze_pretrained_image_tower: bool = True

    # DISCOCLIP_REPRODUCTION_PLAN.md's R1: selects which prepare_datasets.py
    # output variant to train/eval on (see its SVO_PREP_OUTPUT_SUFFIX). ""
    # (default) is the standard threshold-10 pipeline every other experiment
    # in this project uses, untouched.
    dataset_suffix: str = ""

    model_config = SettingsConfigDict(env_prefix="SVO_ML_")
