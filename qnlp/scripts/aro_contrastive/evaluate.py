"""ARO hard-negative evaluation, stratified by task (relation vs attribution).

For each (image, true_caption, false_caption) triple, the model is "correct" if
cosine(image, true) > cosine(image, false). We report this hard-negative accuracy
overall and split by ARO sub-task.

Leakage safety:
  * Model inputs come ONLY from the selected split parquet (default: test).
  * `task` labels are not a model input and were never carried into the dataset
    parquets — they are fetched from the atlas manifest, restricted to the
    selected split's sample_ids, reading only the [sample_id, task] columns.
    No train/val rows are ever loaded.
  * Non-linear mode is inferred from the checkpoint, so the dataset's path
    loading and the model architecture always match what was trained.

Usage:
    python -m qnlp.scripts.aro_contrastive.evaluate <checkpoint>
    python -m qnlp.scripts.aro_contrastive.evaluate <checkpoint> --split val
"""

from collections import defaultdict
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from qnlp.constants import constants
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.datasets.dataloader import vlm_collate_fn
from qnlp.domain.datasets.dataset import VLMDataset
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.aro_contrastive.config import ExperimentConfig
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="aro_evaluate")

ARO_MANIFEST = constants.atlases_path / "aro" / "data_manifest.parquet"

SPLIT_PARQUETS = {
    "train": constants.datasets_path / "aro_train.parquet",
    "val": constants.datasets_path / "aro_val.parquet",
    "test": constants.datasets_path / "aro_test.parquet",
}

COMPILED_COLUMNS = [
    ("true_diagram", "true_symbols", "true_caption", "true_path"),
    ("false_diagram", "false_symbols", "false_caption", "false_path"),
]
SYMBOL_COLS = ["true_symbols", "false_symbols"]


def _load_task_map(split_sample_ids: set[str]) -> dict[str, str]:
    """Map sample_id -> task ('relation'/'attribution') for the split's ids only.

    Reads just two columns from the manifest and immediately filters to the
    sample_ids present in the evaluation split, so no other-split data is used.
    """
    m = pl.read_parquet(ARO_MANIFEST, columns=["sample_id", "task"]).filter(
        pl.col("sample_id").is_in(list(split_sample_ids))
    )
    task_map = dict(zip(m["sample_id"].to_list(), m["task"].to_list()))
    missing = split_sample_ids - set(task_map)
    if missing:
        raise ValueError(f"{len(missing)} eval sample_ids have no task in the manifest (e.g. {next(iter(missing))}).")
    return task_map


def _infer_non_linear(state_dict: dict) -> bool:
    return any("nonlinear_gate" in k for k in state_dict)


def evaluate(checkpoint_path: Path, split: str = "test", batch_size: int | None = None) -> dict:
    cfg = ExperimentConfig()
    device = get_device()
    parquet = SPLIT_PARQUETS[split]
    batch_size = batch_size or cfg.batch_size

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    non_linear = _infer_non_linear(state_dict)
    logger.info(
        f"Checkpoint epoch {checkpoint.get('epoch', '?')} | non_linear_contractions={non_linear} | split={split}"
    )

    size = image_model_hyperparams.image_size
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds = VLMDataset(
        parquet,
        compiled_columns=COMPILED_COLUMNS,
        image_transform=transform,
        use_non_linear_contractions=non_linear,
    )
    logger.info(f"Evaluating on {parquet.name} — {len(ds)} pairs")

    task_map = _load_task_map(set(ds.df["sample_id"].to_list()))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vlm_collate_fn,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Build the model and load the trained weights. EinsumModel.load_state_dict
    # rebuilds its symbol table from the checkpoint, so empty construction is fine.
    text_model = EinsumModel(non_linear_contractions=non_linear)
    image_model = TTNImageModel(cfg.embedding_dim)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    known_symbols = set(model.text_model.sym2weight.keys())

    # Per-task accumulators of per-example correctness and cosine diagnostics.
    correct_by_task: dict[str, list[bool]] = defaultdict(list)
    pos_by_task: dict[str, list[float]] = defaultdict(list)
    neg_by_task: dict[str, list[float]] = defaultdict(list)
    n_skipped = 0

    with torch.no_grad():
        for batch in loader:
            true_caps = batch["true_caption"]
            false_caps = batch["false_caption"]
            sample_ids = batch["sample_id"]

            # Skip any pair referencing a symbol the trained model never saw
            # (shouldn't happen — model was trained on all-split symbols — but
            # guards against a crash and is counted transparently).
            valid = [
                i
                for i in range(len(sample_ids))
                if all(s in known_symbols for s in true_caps[i][1])
                and all(s in known_symbols for s in false_caps[i][1])
            ]
            n_skipped += len(sample_ids) - len(valid)
            if not valid:
                continue

            images = torch.stack([batch["local_image_path"][i] for i in valid]).to(device)
            outputs = model(images, [true_caps[i] for i in valid], [false_caps[i] for i in valid])

            pos = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
            neg = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
            correct = (pos > neg).tolist()

            for j, i in enumerate(valid):
                task = task_map[sample_ids[i]]
                correct_by_task[task].append(bool(correct[j]))
                pos_by_task[task].append(float(pos[j]))
                neg_by_task[task].append(float(neg[j]))

    def _acc(xs: list[bool]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    results: dict[str, dict] = {}
    all_correct: list[bool] = []
    all_pos: list[float] = []
    all_neg: list[float] = []
    for task in sorted(correct_by_task):
        c, p, n = correct_by_task[task], pos_by_task[task], neg_by_task[task]
        results[task] = {"n": len(c), "hard_neg_acc": _acc(c), "true_cos": _mean(p), "false_cos": _mean(n)}
        all_correct += c
        all_pos += p
        all_neg += n
    results["overall"] = {
        "n": len(all_correct),
        "hard_neg_acc": _acc(all_correct),
        "true_cos": _mean(all_pos),
        "false_cos": _mean(all_neg),
    }

    logger.info("=== ARO hard-negative accuracy by task ===")
    logger.info(f"  {'task':<14}{'N':>7}{'acc':>9}{'true_cos':>10}{'false_cos':>11}")
    logger.info(f"  {'-' * 50}")
    for task in [*sorted(correct_by_task), "overall"]:
        r = results[task]
        logger.info(f"  {task:<14}{r['n']:>7}{r['hard_neg_acc']:>9.4f}{r['true_cos']:>10.4f}{r['false_cos']:>11.4f}")
    if n_skipped:
        logger.warning(f"Skipped {n_skipped} pairs with unknown symbols.")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    evaluate(
        "runs/checkpoints/aro_contrastive/2026-06-11_10-25-56/best_model.pt",
        split=args.split,
        batch_size=args.batch_size,
    )
