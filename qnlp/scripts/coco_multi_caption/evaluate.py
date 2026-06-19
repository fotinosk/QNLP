"""
Evaluate a trained COCO ContrastiveVLM checkpoint on compositional benchmarks.

Runs four evaluations in a single pass (model loaded once):
  - Winoground  : text / image / group scores (canonical 3-metric eval)
  - ARO         : hard-negative accuracy stratified by task (attribution / relation)
  - SugarCREPE  : hard-negative accuracy on swap_att subset

Architecture (embedding_dim, non-linear mode) is inferred from the checkpoint
so no config flags are needed. Symbols unseen during COCO training are skipped
and counted transparently.

Usage:
    python -m qnlp.scripts.coco_multi_caption.evaluate <checkpoint_path>
    python -m qnlp.scripts.coco_multi_caption.evaluate <checkpoint_path> --batch_size 64
"""

import argparse
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
from qnlp.domain.datasets.winoground_dataset import WinogroundDataset, winoground_eval_collate_fn
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="coco_evaluate")

ARO_MANIFEST = constants.atlases_path / "aro" / "data_manifest.parquet"

ARO_COMPILED_COLUMNS = [
    ("true_diagram", "true_symbols", "true_caption", "true_path"),
    ("false_diagram", "false_symbols", "false_caption", "false_path"),
]
SUGARCREPE_COMPILED_COLUMNS = [
    ("true_diagram", "true_symbols", "true_caption", "true_path"),
    ("false_diagram", "false_symbols", "false_caption", "false_path"),
]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _infer_non_linear(state_dict: dict) -> bool:
    return any("nonlinear_gate" in k for k in state_dict)


def _infer_mlp_head(state_dict: dict) -> bool:
    return "image_head.net.0.weight" in state_dict


def _infer_embedding_dim(state_dict: dict) -> int:
    # MLP head: first linear is dim → dim*2, so weight shape is (dim*2, dim)
    if "image_head.net.0.weight" in state_dict:
        return state_dict["image_head.net.0.weight"].shape[1]
    return state_dict["image_head.proj.weight"].shape[0]


def load_model(checkpoint_path: Path, device: torch.device) -> ContrastiveVLM:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    non_linear = _infer_non_linear(state_dict)
    mlp_head = _infer_mlp_head(state_dict)
    embedding_dim = _infer_embedding_dim(state_dict)
    logger.info(
        f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}) | "
        f"embedding_dim={embedding_dim} | non_linear={non_linear} | mlp_head={mlp_head}"
    )
    text_model = EinsumModel(non_linear_contractions=non_linear)
    image_model = TTNImageModel(embedding_dim)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=embedding_dim, use_mlp_head=mlp_head)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def _make_transform(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _make_loader(ds, batch_size: int, collate_fn) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )


# ---------------------------------------------------------------------------
# Winoground
# ---------------------------------------------------------------------------


def evaluate_winoground(
    model: ContrastiveVLM,
    device: torch.device,
    batch_size: int,
    parquet: Path | None = None,
) -> dict[str, float]:
    parquet = parquet or constants.datasets_path / "winoground_test.parquet"
    non_linear = model.text_model.non_linear_contractions
    size = image_model_hyperparams.image_size

    ds = WinogroundDataset(
        parquet, mode="eval", image_transform=_make_transform(size), use_non_linear_contractions=non_linear
    )
    loader = _make_loader(ds, batch_size, winoground_eval_collate_fn)
    known = set(model.text_model.sym2weight.keys())

    def _all_known(cap) -> bool:
        return all(s in known for s in cap[1])

    text_correct = image_correct = group_correct = n_total = n_skipped = 0

    with torch.no_grad():
        for batch in loader:
            caps0, caps1 = batch["captions_0"], batch["captions_1"]
            valid = [i for i in range(len(caps0)) if _all_known(caps0[i]) and _all_known(caps1[i])]
            n_skipped += len(caps0) - len(valid)
            if not valid:
                continue

            imgs0 = batch["images_0"][valid].to(device)
            imgs1 = batch["images_1"][valid].to(device)
            c0 = [caps0[i] for i in valid]
            c1 = [caps1[i] for i in valid]

            out0 = model(imgs0, c0, c1)
            out1 = model(imgs1, c1, c0)

            img0 = out0["image_embeddings"]
            img1 = out1["image_embeddings"]
            cap0 = out0["true_caption_embeddings"]
            cap1 = out0["false_caption_embeddings"]

            s00 = F.cosine_similarity(img0, cap0, dim=-1)
            s01 = F.cosine_similarity(img0, cap1, dim=-1)
            s10 = F.cosine_similarity(img1, cap0, dim=-1)
            s11 = F.cosine_similarity(img1, cap1, dim=-1)

            finite = torch.isfinite(s00) & torch.isfinite(s01) & torch.isfinite(s10) & torch.isfinite(s11)
            n_skipped += (~finite).sum().item()

            text_correct += ((s00 > s01) & (s11 > s10) & finite).sum().item()
            image_correct += ((s00 > s10) & (s11 > s01) & finite).sum().item()
            group_correct += ((s00 > s01) & (s11 > s10) & (s00 > s10) & (s11 > s01) & finite).sum().item()
            n_total += finite.sum().item()

    metrics = {
        "text_score": text_correct / n_total if n_total else float("nan"),
        "image_score": image_correct / n_total if n_total else float("nan"),
        "group_score": group_correct / n_total if n_total else float("nan"),
        "n_pairs": n_total,
        "n_skipped": n_skipped,
    }
    if n_skipped:
        logger.warning(f"Winoground: skipped {n_skipped} pairs with unknown/NaN symbols.")
    return metrics


# ---------------------------------------------------------------------------
# ARO (attribution + relation)
# ---------------------------------------------------------------------------


def _load_task_map(sample_ids: set[str]) -> dict[str, str]:
    m = pl.read_parquet(ARO_MANIFEST, columns=["sample_id", "task"]).filter(pl.col("sample_id").is_in(list(sample_ids)))
    return dict(zip(m["sample_id"].to_list(), m["task"].to_list()))


def evaluate_aro(
    model: ContrastiveVLM,
    device: torch.device,
    batch_size: int,
    parquet: Path | None = None,
) -> dict[str, dict]:
    parquet = parquet or constants.datasets_path / "aro_test.parquet"
    non_linear = model.text_model.non_linear_contractions
    size = image_model_hyperparams.image_size

    ds = VLMDataset(
        parquet,
        compiled_columns=ARO_COMPILED_COLUMNS,
        image_transform=_make_transform(size),
        use_non_linear_contractions=non_linear,
    )
    task_map = _load_task_map(set(ds.df["sample_id"].to_list()))
    loader = _make_loader(ds, batch_size, vlm_collate_fn)
    known = set(model.text_model.sym2weight.keys())

    correct_by_task: dict[str, list[bool]] = defaultdict(list)
    pos_by_task: dict[str, list[float]] = defaultdict(list)
    neg_by_task: dict[str, list[float]] = defaultdict(list)
    n_skipped = 0

    with torch.no_grad():
        for batch in loader:
            true_caps = batch["true_caption"]
            false_caps = batch["false_caption"]
            sample_ids = batch["sample_id"]

            valid = [
                i
                for i in range(len(sample_ids))
                if all(s in known for s in true_caps[i][1]) and all(s in known for s in false_caps[i][1])
            ]
            n_skipped += len(sample_ids) - len(valid)
            if not valid:
                continue

            images = batch["local_image_path"][valid].to(device)
            outputs = model(images, [true_caps[i] for i in valid], [false_caps[i] for i in valid])

            pos = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
            neg = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
            correct = (pos > neg).tolist()
            finite = (torch.isfinite(pos) & torch.isfinite(neg)).tolist()

            for j, i in enumerate(valid):
                if not finite[j]:
                    n_skipped += 1
                    continue
                task = task_map[sample_ids[i]]
                correct_by_task[task].append(bool(correct[j]))
                pos_by_task[task].append(float(pos[j]))
                neg_by_task[task].append(float(neg[j]))

    def _acc(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    results: dict[str, dict] = {}
    all_c, all_p, all_n = [], [], []
    for task in sorted(correct_by_task):
        c, p, n = correct_by_task[task], pos_by_task[task], neg_by_task[task]
        results[task] = {"n": len(c), "hard_neg_acc": _acc(c), "true_cos": _mean(p), "false_cos": _mean(n)}
        all_c += c
        all_p += p
        all_n += n
    results["overall"] = {
        "n": len(all_c),
        "hard_neg_acc": _acc(all_c),
        "true_cos": _mean(all_p),
        "false_cos": _mean(all_n),
    }

    if n_skipped:
        logger.warning(f"ARO: skipped {n_skipped} pairs with unknown/NaN symbols.")
    return results


# ---------------------------------------------------------------------------
# SugarCREPE
# ---------------------------------------------------------------------------


def evaluate_sugarcrepe(
    model: ContrastiveVLM,
    device: torch.device,
    batch_size: int,
    subset: str = "swap_att",
    parquet: Path | None = None,
) -> dict[str, float]:
    parquet = parquet or constants.datasets_path / f"sugarcrepe_{subset}_test.parquet"
    non_linear = model.text_model.non_linear_contractions
    size = image_model_hyperparams.image_size

    ds = VLMDataset(
        parquet,
        compiled_columns=SUGARCREPE_COMPILED_COLUMNS,
        image_transform=_make_transform(size),
        use_non_linear_contractions=non_linear,
    )
    loader = _make_loader(ds, batch_size, vlm_collate_fn)
    known = set(model.text_model.sym2weight.keys())

    correct: list[bool] = []
    pos_cos: list[float] = []
    neg_cos: list[float] = []
    n_skipped = 0

    with torch.no_grad():
        for batch in loader:
            true_caps = batch["true_caption"]
            false_caps = batch["false_caption"]

            valid = [
                i
                for i in range(len(true_caps))
                if all(s in known for s in true_caps[i][1]) and all(s in known for s in false_caps[i][1])
            ]
            n_skipped += len(true_caps) - len(valid)
            if not valid:
                continue

            images = batch["local_image_path"][valid].to(device)
            outputs = model(images, [true_caps[i] for i in valid], [false_caps[i] for i in valid])

            pos = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
            neg = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
            finite = (torch.isfinite(pos) & torch.isfinite(neg)).tolist()

            for j in range(len(valid)):
                if not finite[j]:
                    n_skipped += 1
                    continue
                correct.append(bool((pos > neg)[j].item()))
                pos_cos.append(float(pos[j]))
                neg_cos.append(float(neg[j]))

    n = len(correct)
    metrics = {
        "hard_neg_acc": sum(correct) / n if n else float("nan"),
        "true_cos": sum(pos_cos) / n if n else float("nan"),
        "false_cos": sum(neg_cos) / n if n else float("nan"),
        "n_evaluated": n,
        "n_skipped": n_skipped,
    }
    if n_skipped:
        logger.warning(f"SugarCREPE {subset}: skipped {n_skipped} pairs with unknown/NaN symbols.")
    return metrics


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(wino: dict, aro: dict, sc: dict) -> None:
    sep = "=" * 60
    logger.info(sep)
    logger.info("EVALUATION SUMMARY")
    logger.info(sep)

    logger.info("Winoground")
    logger.info(f"  text  : {wino['text_score']:.4f}")
    logger.info(f"  image : {wino['image_score']:.4f}")
    logger.info(f"  group : {wino['group_score']:.4f}")
    logger.info(f"  pairs : {wino['n_pairs']}  skipped: {wino['n_skipped']}")

    logger.info("ARO")
    logger.info(f"  {'task':<14}{'N':>7}{'acc':>9}{'true_cos':>10}{'false_cos':>11}")
    logger.info(f"  {'-' * 50}")
    for task in [*sorted(k for k in aro if k != "overall"), "overall"]:
        r = aro[task]
        logger.info(f"  {task:<14}{r['n']:>7}{r['hard_neg_acc']:>9.4f}{r['true_cos']:>10.4f}{r['false_cos']:>11.4f}")

    logger.info("SugarCREPE (swap_att)")
    logger.info(f"  acc      : {sc['hard_neg_acc']:.4f}")
    logger.info(f"  true_cos : {sc['true_cos']:.4f}  false_cos: {sc['false_cos']:.4f}")
    logger.info(f"  evaluated: {sc['n_evaluated']}  skipped: {sc['n_skipped']}")
    logger.info(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def evaluate_all(
    checkpoint_path: Path,
    batch_size: int = 128,
) -> dict[str, dict]:
    device = get_device()
    model = load_model(checkpoint_path, device)

    logger.info("--- Winoground ---")
    wino = evaluate_winoground(model, device, batch_size)

    logger.info("--- ARO ---")
    aro = evaluate_aro(model, device, batch_size)

    logger.info("--- SugarCREPE (swap_att) ---")
    sc = evaluate_sugarcrepe(model, device, batch_size)

    _print_summary(wino, aro, sc)

    return {"winoground": wino, "aro": aro, "sugarcrepe": sc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a COCO checkpoint on compositional benchmarks.")
    parser.add_argument("checkpoint", type=Path, help="Path to best_model.pt checkpoint")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    evaluate_all(args.checkpoint, batch_size=args.batch_size)
