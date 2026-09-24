"""One-off recovery: re-run ARO/SugarCREPE evaluation against an already-
trained checkpoint, without retraining. Used when the unguarded
evaluate_winoground call in run.py crashed on an M3 (text_backbone=clip)
run before evaluate_aro ever got to run -- see PAPER_EXPERIMENTS_PLAN.md's
M3, job 7447090 -- even though training and the checkpoint itself are sound
(the Trainer's own test-set hard_neg_acc was already computed successfully).

Usage:
    python -m qnlp.scripts.aro_contrastive.reeval_checkpoint <checkpoint_path>

Reads the same env-driven config (ML_*, IMAGE_MODEL_*, TEXT_MODEL_*) the
original training run used -- rerun with the identical environment.
"""

import sys

import torch

from qnlp.constants import constants
from qnlp.discoviz.models.clip_text_model import build_text_model, text_model_hyperparams
from qnlp.discoviz.models.image_model import build_image_model
from qnlp.domain.datasets.dataset import VLMDataset, collect_symbol_sizes
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.aro_contrastive.config import ExperimentConfig
from qnlp.scripts.aro_contrastive.run import SYMBOL_COLS, _compiled_columns
from qnlp.scripts.coco_multi_caption.evaluate import evaluate_aro, evaluate_sugarcrepe
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device


def run(checkpoint_path: str) -> None:
    logger = setup_logger(log_name="aro_reeval")
    cfg = ExperimentConfig()
    device = get_device()
    suffix = cfg.dataset_suffix
    DATASETS_PATH = constants.datasets_path
    train_parquet = DATASETS_PATH / f"aro_train{suffix}.parquet"
    val_parquet = DATASETS_PATH / f"aro_val{suffix}.parquet"
    test_parquet = DATASETS_PATH / f"aro_test{suffix}.parquet"

    train_ds = VLMDataset(train_parquet, compiled_columns=_compiled_columns())
    val_ds = VLMDataset(val_parquet, compiled_columns=_compiled_columns())
    test_ds = VLMDataset(test_parquet, compiled_columns=_compiled_columns())

    symbols, sizes = collect_symbol_sizes(
        [train_ds, val_ds, test_ds],
        SYMBOL_COLS,
        remap={constants.embedding_dim: cfg.embedding_dim, constants.bond_dim: cfg.bond_dim},
    )
    logger.info(f"Text backbone: {text_model_hyperparams.text_backbone}")

    text_model = build_text_model(
        cfg.embedding_dim, symbols, sizes, non_linear_contractions=cfg.use_non_linear_contractions, use_weight_norm=True
    ).to(device)
    image_model = build_image_model(cfg.embedding_dim).to(device)
    model = ContrastiveVLM(
        text_model, image_model, embedding_dim=cfg.embedding_dim, use_projection_head=cfg.use_alignment_head
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}: {checkpoint_path}")

    def _guard(name: str, fn):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"{name}: eval skipped ({type(e).__name__}: {e})")
            return None

    aro = _guard("ARO", lambda: evaluate_aro(model, device, cfg.batch_size, parquet=test_parquet))
    sc = _guard("SugarCREPE", lambda: evaluate_sugarcrepe(model, device, cfg.batch_size))

    sep = "=" * 60
    logger.info(sep)
    logger.info("FINAL RESULTS (re-eval)")
    logger.info(sep)
    logger.info("ARO")
    if aro:
        logger.info(f"  {'task':<14}{'N':>7}{'acc':>9}{'true_cos':>10}{'false_cos':>11}")
        for task in [*sorted(k for k in aro if k != "overall"), "overall"]:
            r = aro[task]
            logger.info(
                f"  {task:<14}{r['n']:>7}{r['hard_neg_acc']:>9.4f}{r['true_cos']:>10.4f}{r['false_cos']:>11.4f}"
            )
    else:
        logger.info("  (unavailable)")
    logger.info("SugarCREPE (swap_obj)")
    if sc:
        logger.info(f"  acc: {sc['hard_neg_acc']:.4f}  evaluated: {sc['n_evaluated']}  skipped: {sc['n_skipped']}")
    else:
        logger.info("  (unavailable)")
    logger.info(sep)


if __name__ == "__main__":
    run(sys.argv[1])
