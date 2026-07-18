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
import json
from collections import defaultdict
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from qnlp.constants import constants
from qnlp.core.training.retrieval_eval import retrieval_metrics
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.datasets.dataloader import vlm_collate_fn
from qnlp.domain.datasets.dataset import VLMDataset
from qnlp.domain.datasets.winoground_dataset import WinogroundDataset, winoground_eval_collate_fn
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="coco_evaluate")

# COCO test-set retrieval config (mirrors training-time retrieval eval).
TEST_SIZE = 5000
COCO_COMPILED_COLUMNS = [("diagram", "symbols", "caption", "path")]

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
    # Load to CPU first: the training checkpoint also holds the AdamW optimizer
    # state (~2x the model size). map_location=device would put ALL of it on the
    # GPU and OOM a large linear model. We only need model_state_dict on-device.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    del checkpoint, state_dict  # free the CPU-side optimizer state before moving to GPU
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


# Per-item tag classes for Winoground. Keys are the manifest `id` (0-399); an
# empty list means the item carries no special tag and is bucketed as "normal".
WINO_TAG_PATH = constants.atlases_path / "winoground" / "new_tag_assignments.json"
NORMAL_CLASS = "normal"


def _winoground_pair_results(
    model: ContrastiveVLM,
    device: torch.device,
    batch_size: int,
    parquet: Path | None = None,
) -> tuple[dict[str, tuple[bool, bool, bool]], int]:
    """Run Winoground once and return per-pair (text, image, group) correctness.

    Keyed by pair_id (e.g. 'winoground_123'). Pairs with unknown symbols or
    non-finite scores are omitted and counted in n_skipped.
    """
    parquet = parquet or constants.datasets_path / f"winoground_eval{constants.artifact_suffix}.parquet"
    non_linear = model.text_model.non_linear_contractions
    size = image_model_hyperparams.image_size

    ds = WinogroundDataset(
        parquet, mode="eval", image_transform=_make_transform(size), use_non_linear_contractions=non_linear
    )
    loader = _make_loader(ds, batch_size, winoground_eval_collate_fn)
    known = set(model.text_model.sym2weight.keys())

    def _all_known(cap) -> bool:
        return all(s in known for s in cap[1])

    per_pair: dict[str, tuple[bool, bool, bool]] = {}
    n_skipped = 0

    with torch.no_grad():
        for batch in loader:
            caps0, caps1 = batch["captions_0"], batch["captions_1"]
            pair_ids = batch["pair_ids"]
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
            text = (s00 > s01) & (s11 > s10)
            image = (s00 > s10) & (s11 > s01)
            group = text & image

            for j, i in enumerate(valid):
                if not bool(finite[j]):
                    n_skipped += 1
                    continue
                per_pair[pair_ids[i]] = (bool(text[j]), bool(image[j]), bool(group[j]))

    return per_pair, n_skipped


def _agg_winoground(results: list[tuple[bool, bool, bool]]) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {"text_score": float("nan"), "image_score": float("nan"), "group_score": float("nan"), "n_pairs": 0}
    return {
        "text_score": sum(r[0] for r in results) / n,
        "image_score": sum(r[1] for r in results) / n,
        "group_score": sum(r[2] for r in results) / n,
        "n_pairs": n,
    }


def evaluate_winoground(
    model: ContrastiveVLM,
    device: torch.device,
    batch_size: int,
    parquet: Path | None = None,
) -> dict[str, float]:
    per_pair, n_skipped = _winoground_pair_results(model, device, batch_size, parquet)
    metrics = _agg_winoground(list(per_pair.values()))
    metrics["n_skipped"] = n_skipped
    if n_skipped:
        logger.warning(f"Winoground: skipped {n_skipped} pairs with unknown/NaN symbols.")
    return metrics


def _load_wino_tags(tag_path: Path) -> dict[int, list[str]]:
    """Load {manifest_id -> [tag, ...]} from the tag-assignment JSON."""
    raw = json.loads(Path(tag_path).read_text())
    return {int(k): v for k, v in raw.items()}


def _pair_id_to_index(pair_id: str) -> int:
    """'winoground_123' -> 123."""
    return int(str(pair_id).rsplit("_", 1)[-1])


def evaluate_winoground_by_tag(
    model: ContrastiveVLM,
    device: torch.device,
    batch_size: int,
    parquet: Path | None = None,
    tag_path: Path | None = None,
) -> dict[str, dict]:
    """Winoground scores broken down by tag class, plus overall.

    Tags are multi-label, so a pair contributes to every class it carries; the
    per-class n_pairs therefore sum to more than the overall total. Pairs with no
    tag are bucketed under 'normal'. Returns {class: {text/image/group/n_pairs},
    ..., 'overall': {...}}.
    """
    tag_path = tag_path or WINO_TAG_PATH
    per_pair, n_skipped = _winoground_pair_results(model, device, batch_size, parquet)
    tags_by_id = _load_wino_tags(tag_path)

    by_class: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
    for pair_id, res in per_pair.items():
        classes = tags_by_id.get(_pair_id_to_index(pair_id), [])
        for c in classes or [NORMAL_CLASS]:
            by_class[c].append(res)

    results: dict[str, dict] = {cls: _agg_winoground(vals) for cls, vals in by_class.items()}
    results["overall"] = _agg_winoground(list(per_pair.values()))
    results["overall"]["n_skipped"] = n_skipped
    if n_skipped:
        logger.warning(f"Winoground (tagged): skipped {n_skipped} pairs with unknown/NaN symbols.")
    return results


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
    parquet = parquet or constants.datasets_path / f"aro_eval{constants.artifact_suffix}.parquet"
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
    subset: str = "swap_obj",
    parquet: Path | None = None,
) -> dict[str, float]:
    parquet = parquet or constants.datasets_path / f"sugarcrepe_{subset}_eval{constants.artifact_suffix}.parquet"
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

    logger.info("SugarCREPE (swap_obj)")
    logger.info(f"  acc      : {sc['hard_neg_acc']:.4f}")
    logger.info(f"  true_cos : {sc['true_cos']:.4f}  false_cos: {sc['false_cos']:.4f}")
    logger.info(f"  evaluated: {sc['n_evaluated']}  skipped: {sc['n_skipped']}")
    logger.info(sep)


def _print_winoground_by_tag(by_tag: dict[str, dict]) -> None:
    sep = "=" * 60
    logger.info(sep)
    logger.info("WINOGROUND BY TAG (multi-label; 'normal' = untagged)")
    logger.info(sep)
    logger.info(f"  {'class':<22}{'N':>6}{'text':>9}{'image':>9}{'group':>9}")
    logger.info(f"  {'-' * 55}")
    for cls in [*sorted(k for k in by_tag if k != "overall"), "overall"]:
        r = by_tag[cls]
        logger.info(
            f"  {cls:<22}{r['n_pairs']:>6}{r['text_score']:>9.4f}{r['image_score']:>9.4f}{r['group_score']:>9.4f}"
        )
    logger.info(sep)


# ---------------------------------------------------------------------------
# Full benchmark battery + copy-pasteable report
# ---------------------------------------------------------------------------


def evaluate_all_benchmarks(
    model: ContrastiveVLM,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict | None]:
    """Run every compositional benchmark used at end-of-training. Each eval is
    guarded so a missing dataset (e.g. sugarcrepe++ not yet built) logs a warning
    and returns None instead of killing the whole report."""

    def _guard(name: str, fn):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"{name}: eval skipped ({type(e).__name__}: {e})")
            return None

    return {
        "winoground": _guard("Winoground", lambda: evaluate_winoground(model, device, batch_size)),
        "winoground_by_tag": _guard("Winoground-by-tag", lambda: evaluate_winoground_by_tag(model, device, batch_size)),
        "aro": _guard("ARO", lambda: evaluate_aro(model, device, batch_size)),
        "sugarcrepe_full": _guard(
            "SugarCREPE full",
            lambda: evaluate_sugarcrepe(
                model,
                device,
                batch_size,
                parquet=constants.datasets_path / f"sugarcrepe_full_eval{constants.artifact_suffix}.parquet",
            ),
        ),
        "sugarcrepepp": _guard(
            "SugarCREPE++",
            lambda: evaluate_sugarcrepe(
                model,
                device,
                batch_size,
                parquet=constants.datasets_path / f"sugarcrepepp_eval{constants.artifact_suffix}.parquet",
            ),
        ),
    }


# ---------------------------------------------------------------------------
# Frozen (CLIP image tower) benchmark variants
#
# run_frozen.py / evaluate_frozen.py train only the text model + a linear head
# and encode images with a cached CLIP tower, so they cannot go through the
# ContrastiveVLM-based evaluators above. These mirror them exactly — same
# grouping and skip logic, same returned shapes — but take
# (text_model, text_head, image_cache) instead of a ContrastiveVLM. `image_cache`
# is any callable mapping a list of image paths to a batch of L2-normalised
# CLIP embeddings (run_frozen.CLIPImageCache).
# ---------------------------------------------------------------------------


def _winoground_pair_results_frozen(
    text_model, text_head, image_cache, batch_size, device, nlc, parquet: Path | None = None
) -> tuple[dict[str, tuple[bool, bool, bool]], int]:
    """Per-pair (text, image, group) correctness, keyed by pair_id. Mirrors
    _winoground_pair_results but with a frozen CLIP image tower."""
    parquet = parquet or constants.datasets_path / f"winoground_eval{constants.artifact_suffix}.parquet"
    ds = WinogroundDataset(parquet, mode="eval", use_non_linear_contractions=nlc, return_image_paths=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=winoground_eval_collate_fn, num_workers=0)
    known = set(text_model.sym2weight.keys())

    def _all_known(cap) -> bool:
        return all(s in known for s in cap[1])

    per_pair: dict[str, tuple[bool, bool, bool]] = {}
    n_skipped = 0

    text_model.eval()
    text_head.eval()
    with torch.no_grad():
        for batch in loader:
            caps0, caps1 = batch["captions_0"], batch["captions_1"]
            paths0, paths1 = batch["images_0"], batch["images_1"]  # lists of path strings
            pair_ids = batch["pair_ids"]
            valid = [i for i in range(len(caps0)) if _all_known(caps0[i]) and _all_known(caps1[i])]
            n_skipped += len(caps0) - len(valid)
            if not valid:
                continue

            img0 = image_cache([paths0[i] for i in valid])
            img1 = image_cache([paths1[i] for i in valid])
            cap0 = F.normalize(text_head(text_model([caps0[i] for i in valid])), dim=-1)
            cap1 = F.normalize(text_head(text_model([caps1[i] for i in valid])), dim=-1)

            s00 = F.cosine_similarity(img0, cap0, dim=-1)
            s01 = F.cosine_similarity(img0, cap1, dim=-1)
            s10 = F.cosine_similarity(img1, cap0, dim=-1)
            s11 = F.cosine_similarity(img1, cap1, dim=-1)

            finite = torch.isfinite(s00) & torch.isfinite(s01) & torch.isfinite(s10) & torch.isfinite(s11)
            text = (s00 > s01) & (s11 > s10)
            image = (s00 > s10) & (s11 > s01)
            group = text & image

            for j, i in enumerate(valid):
                if not bool(finite[j]):
                    n_skipped += 1
                    continue
                per_pair[pair_ids[i]] = (bool(text[j]), bool(image[j]), bool(group[j]))

    return per_pair, n_skipped


def evaluate_aro_frozen(
    text_model, text_head, image_cache, batch_size, device, nlc, parquet: Path | None = None
) -> dict[str, dict]:
    """ARO hard-negative accuracy broken down by task (attribution/relation) plus
    overall. Mirrors evaluate_aro with a frozen CLIP image tower."""
    parquet = parquet or constants.datasets_path / f"aro_eval{constants.artifact_suffix}.parquet"
    ds = VLMDataset(
        parquet, compiled_columns=ARO_COMPILED_COLUMNS, use_non_linear_contractions=nlc, return_image_paths=True
    )
    task_map = _load_task_map(set(ds.df["sample_id"].to_list()))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=vlm_collate_fn, num_workers=0)
    known = set(text_model.sym2weight.keys())

    correct_by_task: dict[str, list[bool]] = defaultdict(list)
    pos_by_task: dict[str, list[float]] = defaultdict(list)
    neg_by_task: dict[str, list[float]] = defaultdict(list)
    n_skipped = 0

    text_model.eval()
    text_head.eval()
    with torch.no_grad():
        for batch in loader:
            true_caps = batch["true_caption"]
            false_caps = batch["false_caption"]
            sample_ids = batch["sample_id"]
            paths = batch["local_image_path"]  # path strings (return_image_paths=True)

            valid = [
                i
                for i in range(len(sample_ids))
                if all(s in known for s in true_caps[i][1]) and all(s in known for s in false_caps[i][1])
            ]
            n_skipped += len(sample_ids) - len(valid)
            if not valid:
                continue

            img_emb = image_cache([paths[i] for i in valid])
            true_emb = F.normalize(text_head(text_model([true_caps[i] for i in valid])), dim=-1)
            false_emb = F.normalize(text_head(text_model([false_caps[i] for i in valid])), dim=-1)

            pos = F.cosine_similarity(true_emb, img_emb)
            neg = F.cosine_similarity(false_emb, img_emb)
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

    results: dict[str, dict] = {}
    all_c, all_p, all_n = [], [], []
    for task in sorted(correct_by_task):
        c, p, n = correct_by_task[task], pos_by_task[task], neg_by_task[task]
        results[task] = {"n": len(c), "hard_neg_acc": _acc(c), "true_cos": _acc(p), "false_cos": _acc(n)}
        all_c += c
        all_p += p
        all_n += n
    results["overall"] = {
        "n": len(all_c),
        "hard_neg_acc": _acc(all_c),
        "true_cos": _acc(all_p),
        "false_cos": _acc(all_n),
    }
    if n_skipped:
        logger.warning(f"ARO (frozen): skipped {n_skipped} pairs with unknown/NaN symbols.")
    return results


def evaluate_sugarcrepe_frozen(
    text_model, text_head, image_cache, batch_size, device, nlc, parquet: Path
) -> dict[str, float]:
    """SugarCREPE hard-negative accuracy. Mirrors evaluate_sugarcrepe with a
    frozen CLIP image tower."""
    ds = VLMDataset(
        parquet, compiled_columns=SUGARCREPE_COMPILED_COLUMNS, use_non_linear_contractions=nlc, return_image_paths=True
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=vlm_collate_fn, num_workers=0)
    known = set(text_model.sym2weight.keys())

    correct: list[bool] = []
    pos_cos: list[float] = []
    neg_cos: list[float] = []
    n_skipped = 0

    text_model.eval()
    text_head.eval()
    with torch.no_grad():
        for batch in loader:
            true_caps = batch["true_caption"]
            false_caps = batch["false_caption"]
            paths = batch["local_image_path"]

            valid = [
                i
                for i in range(len(true_caps))
                if all(s in known for s in true_caps[i][1]) and all(s in known for s in false_caps[i][1])
            ]
            n_skipped += len(true_caps) - len(valid)
            if not valid:
                continue

            img_emb = image_cache([paths[i] for i in valid])
            true_emb = F.normalize(text_head(text_model([true_caps[i] for i in valid])), dim=-1)
            false_emb = F.normalize(text_head(text_model([false_caps[i] for i in valid])), dim=-1)

            pos = F.cosine_similarity(true_emb, img_emb)
            neg = F.cosine_similarity(false_emb, img_emb)
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
        logger.warning(f"SugarCREPE (frozen): skipped {n_skipped} pairs with unknown/NaN symbols.")
    return metrics


def evaluate_all_benchmarks_frozen(
    text_model, text_head, image_cache, batch_size, device, nlc
) -> dict[str, dict | None]:
    """Frozen counterpart of evaluate_all_benchmarks. Returns the same-shaped
    dict (winoground / winoground_by_tag / aro / sugarcrepe_full / sugarcrepepp)
    so print_full_report renders an identical report. Each benchmark is guarded
    against missing datasets. Winoground is run once and reused for both the
    overall and by-tag breakdowns."""

    def _guard(name: str, fn):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"{name}: eval skipped ({type(e).__name__}: {e})")
            return None

    wino: dict | None = None
    wino_by_tag: dict | None = None
    per_pair = _guard(
        "Winoground",
        lambda: _winoground_pair_results_frozen(text_model, text_head, image_cache, batch_size, device, nlc),
    )
    if per_pair is not None:
        pairs, n_skipped = per_pair
        wino = _agg_winoground(list(pairs.values()))
        wino["n_skipped"] = n_skipped

        def _by_tag():
            tags_by_id = _load_wino_tags(WINO_TAG_PATH)
            by_class: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
            for pair_id, res in pairs.items():
                for c in tags_by_id.get(_pair_id_to_index(pair_id), []) or [NORMAL_CLASS]:
                    by_class[c].append(res)
            out = {cls: _agg_winoground(vals) for cls, vals in by_class.items()}
            out["overall"] = _agg_winoground(list(pairs.values()))
            out["overall"]["n_skipped"] = n_skipped
            return out

        wino_by_tag = _guard("Winoground-by-tag", _by_tag)

    return {
        "winoground": wino,
        "winoground_by_tag": wino_by_tag,
        "aro": _guard("ARO", lambda: evaluate_aro_frozen(text_model, text_head, image_cache, batch_size, device, nlc)),
        "sugarcrepe_full": _guard(
            "SugarCREPE full",
            lambda: evaluate_sugarcrepe_frozen(
                text_model,
                text_head,
                image_cache,
                batch_size,
                device,
                nlc,
                parquet=constants.datasets_path / f"sugarcrepe_full_eval{constants.artifact_suffix}.parquet",
            ),
        ),
        "sugarcrepepp": _guard(
            "SugarCREPE++",
            lambda: evaluate_sugarcrepe_frozen(
                text_model,
                text_head,
                image_cache,
                batch_size,
                device,
                nlc,
                parquet=constants.datasets_path / f"sugarcrepepp_eval{constants.artifact_suffix}.parquet",
            ),
        ),
    }


def log_banner(title: str, info: dict) -> None:
    """Log a titled key/value banner (used at both start and end of training)."""
    sep = "=" * 72
    logger.info(sep)
    logger.info(title)
    logger.info(sep)
    for k, v in info.items():
        logger.info(f"  {k}: {v}")
    logger.info(sep)


def print_full_report(retrieval: dict | None, benchmarks: dict, info: dict | None = None) -> None:
    """Print one contiguous, copy-pasteable block with the model info and every
    metric: retrieval, Winoground (overall + per-tag), ARO (attribution/relation/
    overall), and SugarCREPE (full / ++)."""
    sep = "=" * 72
    log = logger.info
    log(sep)
    log("FINAL TRAINING REPORT")
    log(sep)

    if info:
        for k, v in info.items():
            log(f"  {k}: {v}")
        log("-" * 72)

    log("Retrieval (test)")
    if retrieval:
        for k in sorted(retrieval):
            log(f"  {k:<18}: {retrieval[k]:.4f}")
    else:
        log("  (not computed)")
    log("-" * 72)

    wino = benchmarks.get("winoground")
    log("Winoground (overall)")
    if wino:
        log(f"  text : {wino['text_score']:.4f}   image: {wino['image_score']:.4f}   group: {wino['group_score']:.4f}")
        log(f"  pairs: {wino['n_pairs']}  skipped: {wino.get('n_skipped', 0)}")
    else:
        log("  (unavailable)")

    by_tag = benchmarks.get("winoground_by_tag")
    if by_tag:
        log("Winoground by tag (multi-label; 'normal' = untagged)")
        log(f"  {'class':<22}{'N':>6}{'text':>9}{'image':>9}{'group':>9}")
        for cls in [*sorted(k for k in by_tag if k != "overall"), "overall"]:
            r = by_tag[cls]
            log(f"  {cls:<22}{r['n_pairs']:>6}{r['text_score']:>9.4f}{r['image_score']:>9.4f}{r['group_score']:>9.4f}")
    log("-" * 72)

    aro = benchmarks.get("aro")
    log("ARO (hard-neg acc by task)")
    if aro:
        log(f"  {'task':<14}{'N':>7}{'acc':>9}{'true_cos':>10}{'false_cos':>11}")
        for task in [*sorted(k for k in aro if k != "overall"), "overall"]:
            r = aro[task]
            log(f"  {task:<14}{r['n']:>7}{r['hard_neg_acc']:>9.4f}{r['true_cos']:>10.4f}{r['false_cos']:>11.4f}")
    else:
        log("  (unavailable)")
    log("-" * 72)

    log("SugarCREPE (hard-neg acc)")
    for name, key in [("full", "sugarcrepe_full"), ("++", "sugarcrepepp")]:
        sc = benchmarks.get(key)
        if sc:
            log(f"  {name:<10} acc={sc['hard_neg_acc']:.4f}  evaluated={sc['n_evaluated']}  skipped={sc['n_skipped']}")
        else:
            log(f"  {name:<10} (unavailable)")
    log(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    model: ContrastiveVLM,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """COCO test-set i2t/t2i retrieval, deduplicated to one caption per image
    (capped at TEST_SIZE). Uses the dataset matching the checkpoint's mode."""
    non_linear = model.text_model.non_linear_contractions
    dataset = "coco_single_caption_nlc" if non_linear else "coco_single_caption"
    parquet = constants.datasets_path / f"{dataset}{constants.artifact_suffix}_test.parquet"
    if not parquet.exists():
        # A non-linear dataset serves linear models too (the path column is just
        # ignored), so fall back to the nlc build when only it was created — e.g.
        # the tree-no-type parser only ever produces the single nlc dataset.
        parquet = constants.datasets_path / f"coco_single_caption_nlc{constants.artifact_suffix}_test.parquet"
    size = image_model_hyperparams.image_size

    ds = VLMDataset(
        parquet,
        compiled_columns=COCO_COMPILED_COLUMNS,
        image_transform=_make_transform(size),
        use_non_linear_contractions=non_linear,
    )
    # One caption per unique image (retrieval assumes a diagonal ground truth).
    seen: set[str] = set()
    indices: list[int] = []
    for i, sid in enumerate(ds.df["sample_id"].to_list()):
        if sid not in seen:
            seen.add(sid)
            indices.append(i)
            if len(indices) >= TEST_SIZE:
                break
    loader = DataLoader(
        Subset(ds, indices), batch_size=batch_size, shuffle=False, collate_fn=vlm_collate_fn, num_workers=4
    )

    model.eval()
    img_embs, txt_embs = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["local_image_path"].to(device)
            outputs = model(images, batch["caption"])
            img_e = outputs["image_embeddings"]
            txt_e = outputs["true_caption_embeddings"]
            finite = torch.isfinite(img_e).all(-1) & torch.isfinite(txt_e).all(-1)
            img_embs.append(img_e[finite].cpu())
            txt_embs.append(txt_e[finite].cpu())

    return retrieval_metrics(torch.cat(img_embs), torch.cat(txt_embs))


def evaluate_all(
    checkpoint_path: Path,
    batch_size: int = 128,
) -> dict[str, dict | None]:
    device = get_device()
    model = load_model(checkpoint_path, device)

    try:
        retrieval = evaluate_retrieval(model, device, batch_size)
    except Exception as e:
        logger.warning(f"Retrieval: eval skipped ({type(e).__name__}: {e})")
        retrieval = None

    benchmarks = evaluate_all_benchmarks(model, device, batch_size)
    info = {
        "checkpoint": str(checkpoint_path),
        "embedding_dim": model.embedding_dim,
        "non_linear": model.text_model.non_linear_contractions,
    }
    print_full_report(retrieval=retrieval, benchmarks=benchmarks, info=info)
    return {"retrieval": retrieval, **benchmarks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a COCO checkpoint on compositional benchmarks.")
    parser.add_argument("checkpoint", type=Path, help="Path to best_model.pt checkpoint")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    evaluate_all(args.checkpoint, batch_size=args.batch_size)
