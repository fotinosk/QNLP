"""One-off recovery: re-run SVO-Probes/SVO-Swap evaluation against an
already-trained checkpoint, without retraining. Used when evaluate.py had
a bug that silently dropped SVO-Probes from a run's final report (see
PAPER_EXPERIMENTS_PLAN.md's M3 -- job 7447089) even though training and
the checkpoint itself are sound.

Usage:
    python -m qnlp.scripts.svo.reeval_checkpoint <checkpoint_path>

Reads the same env-driven config (SVO_ML_*, IMAGE_MODEL_*, TEXT_MODEL_*)
the original training run used -- rerun with the identical environment.
"""

import sys

import torch
from torchvision import transforms

from qnlp.constants import constants
from qnlp.discoviz.models.clip_text_model import build_text_model, text_model_hyperparams
from qnlp.discoviz.models.image_model import build_image_model, image_model_hyperparams
from qnlp.domain.datasets.dataloader import get_dataloaders
from qnlp.domain.datasets.dataset import collect_symbol_sizes
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.coco_multi_caption.evaluate import evaluate_svo, print_full_report
from qnlp.scripts.svo.config import SVOExperimentConfig
from qnlp.scripts.svo.run import IMAGE_COLUMNS, SYMBOL_COLS, _compiled_columns
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="svo_reeval")


def run(checkpoint_path: str) -> None:
    cfg = SVOExperimentConfig()
    device = get_device()
    suffix = cfg.dataset_suffix
    DATASETS_PATH = constants.datasets_path
    TRAIN_PARQUET = DATASETS_PATH / f"svo_train_probes{suffix}.parquet"
    VAL_PARQUET = DATASETS_PATH / f"svo_val_probes{suffix}.parquet"
    TEST_PARQUET = DATASETS_PATH / f"svo_test_probes{suffix}.parquet"

    size = image_model_hyperparams.image_size
    val_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    _, (train_ds, val_ds, test_ds) = get_dataloaders(
        train_parquet=TRAIN_PARQUET,
        val_parquet=VAL_PARQUET,
        test_parquet=TEST_PARQUET,
        batch_size=cfg.batch_size,
        train_transform=val_transform,
        val_transform=val_transform,
        image_columns=IMAGE_COLUMNS,
        compiled_columns=_compiled_columns(),
        use_non_linear_contractions=cfg.use_non_linear_contractions,
    )

    symbols, sizes = collect_symbol_sizes(
        [train_ds, val_ds, test_ds],
        SYMBOL_COLS,
        remap={constants.embedding_dim: cfg.embedding_dim, constants.bond_dim: cfg.bond_dim},
    )
    logger.info(f"Text backbone: {text_model_hyperparams.text_backbone}")

    text_model = build_text_model(
        cfg.embedding_dim,
        symbols,
        sizes,
        non_linear_contractions=cfg.use_non_linear_contractions,
        use_weight_norm=cfg.use_weight_norm,
    ).to(device)
    image_model = build_image_model(cfg.embedding_dim).to(device)
    model = ContrastiveVLM(
        text_model,
        image_model,
        embedding_dim=cfg.embedding_dim,
        use_mlp_head=cfg.use_mlp_head,
        use_projection_head=cfg.use_alignment_head,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}: {checkpoint_path}")

    results = evaluate_svo(
        model,
        device,
        cfg.batch_size,
        probes_parquet=TEST_PARQUET,
        swap_parquet=DATASETS_PATH / f"svo_swap_eval{suffix}.parquet",
    )
    print_full_report(
        None, {}, info={"experiment": "svo_probes_reeval", "checkpoint_path": str(checkpoint_path)}, svo=results
    )


if __name__ == "__main__":
    run(sys.argv[1])
