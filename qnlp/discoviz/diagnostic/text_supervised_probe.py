"""
Supervised capacity probe for the TEXT tower (EinsumModel) — the analogue
of ttn_supervised_probe.py for the image side, run for the same reason:
Routes A and B (TTN_CIFAR_EXPERIMENTS.md's "Implementation spec") both
assume EinsumModel's per-symbol tensors carry meaningful subject/verb/object
structure, and that assumption has never been tested directly. The image
tower turned out to be catastrophically broken specifically because a
supervised probe was run on it instead of trusted by assumption — this is
the same check on the text side, run first per the agreed execution order
(Phase 0 diagnostics -> Route D -> N/P1/A-data -> B/A jobs).

Trains EinsumModel + a linear classifier end-to-end with plain
cross-entropy to predict a caption's verb (or object) class, top-K classes
by frequency, using SVO-Probes' own train/val/test splits and compiled
diagrams (no image tower, no contrastive loss, no CCG changes). Compares
against a majority baseline and a bag-of-words logistic regression
reference built from the same captions' raw words. Also reports kNN class
consistency on the trained caption embeddings (chance = 1/K), the same
diagnostic used in tower_spread_trace.py for the image tower.

Usage:
    python -m qnlp.discoviz.diagnostic.text_supervised_probe --label-col verb --top-k 20
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from qnlp.constants import constants
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.domain.datasets.dataset import VLMDataset, _deserialize_symbols, collect_symbol_sizes
from qnlp.utils.early_stopping import EarlyStopping, ModelTrainingStatus
from qnlp.utils.logging import setup_logger
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="text_supervised_probe")

SVO_DIR = Path("data/svo/raw").resolve()
SVO_CSV = SVO_DIR / "svo_probes_corrected.csv"
IMAGE_DIRS = [SVO_DIR / "images", SVO_DIR / "images_old"]
_WORD_RE = re.compile(r"[a-z']+")


def _resolve_image_path(image_id: str) -> str | None:
    for d in IMAGE_DIRS:
        c = d / f"{image_id}.jpg"
        if c.exists():
            return str(c)
    return None


def _sample_id_to_role_map() -> pl.DataFrame:
    """Reconstruct sample_id -> subj/verb/obj/corrected_sentence, exactly
    mirroring load_svo_to_atlas.py's filtering (same image-availability
    filter, same row order) so sample_id = f"svo_{i}" lines up with the
    compiled parquets' own sample_id. Verified exact (8609/8609, 2908/2908,
    2767/2767 matched) in TTN_CIFAR_EXPERIMENTS.md's Phase 0 follow-up."""
    df = pl.read_csv(SVO_CSV)
    all_ids = set(df["pos_image_id"].cast(pl.String).to_list()) | set(df["neg_image_id"].cast(pl.String).to_list())
    path_map = {i: _resolve_image_path(i) for i in all_ids}
    df = df.with_columns(
        pl.col("pos_image_id").cast(pl.String).replace_strict(path_map, default=None).alias("pos_local_image_path"),
        pl.col("neg_image_id").cast(pl.String).replace_strict(path_map, default=None).alias("neg_local_image_path"),
    )
    df = df.filter(pl.col("pos_local_image_path").is_not_null() & pl.col("neg_local_image_path").is_not_null())
    df = df.with_columns(pl.Series("sample_id", [f"svo_{i}" for i in range(len(df))]))
    return df.select("sample_id", "subj", "verb", "obj", "corrected_sentence")


class CaptionClassifier(nn.Module):
    """EinsumModel + a plain linear classifier. Nothing else."""

    def __init__(self, symbols, sizes, num_classes: int, embedding_dim: int, non_linear: bool = False):
        super().__init__()
        self.text_model = EinsumModel(symbols, sizes, non_linear_contractions=non_linear)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, captions) -> torch.Tensor:
        return self.classifier(self.text_model(captions))

    def embed(self, captions) -> torch.Tensor:
        return self.text_model(captions)


class BagOfWordsClassifier(nn.Module):
    """Raw-word-count linear classifier — the text-side analogue of the
    CIFAR probe's raw-pixel logistic regression floor."""

    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(vocab_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class CaptionDataset(Dataset):
    def __init__(self, diagrams: list, symbols_list: list, labels: list[int]):
        self.diagrams = diagrams
        self.symbols_list = symbols_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (self.diagrams[idx], self.symbols_list[idx]), self.labels[idx]


class BowDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int]):
        self.vocab = vocab
        self.labels = labels
        self.vectors = torch.zeros(len(texts), len(vocab))
        for i, text in enumerate(texts):
            for w in _WORD_RE.findall(text.lower()):
                if w in vocab:
                    self.vectors[i, vocab[w]] += 1.0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.vectors[idx], self.labels[idx]


def _collate_captions(batch):
    captions = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return captions, labels


def _load_split(split: str, label_col: str, top_classes: set, class_to_idx: dict, role_map: pl.DataFrame):
    probes = pl.read_parquet(constants.datasets_path / f"svo_{split}_probes.parquet")
    joined = probes.select("sample_id", "diagram", "symbols").join(role_map, on="sample_id", how="inner")
    joined = joined.filter(pl.col(label_col).is_in(list(top_classes)))

    diagrams = joined["diagram"].to_list()
    symbols_list = [_deserialize_symbols(s) for s in joined["symbols"].to_list()]
    labels = [class_to_idx[v] for v in joined[label_col].to_list()]
    texts = joined["corrected_sentence"].to_list()
    return diagrams, symbols_list, labels, texts


def _run_epoch(model, loader, optimizer, device, train: bool, is_bow: bool = False):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, labels in loader:
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            if is_bow:
                logits = model(x.to(device))
            else:
                logits = model(x)
            finite = torch.isfinite(logits).all(-1)
            if not finite.any():
                continue
            logits, labels_f = logits[finite], labels[finite]
            loss = F.cross_entropy(logits, labels_f)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            bs = labels_f.shape[0]
            total_loss += loss.item() * bs
            correct += (logits.argmax(-1) == labels_f).sum().item()
            total += bs
    return total_loss / max(total, 1), correct / max(total, 1)


def _knn_consistency(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    sim = F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).t()
    sim.fill_diagonal_(-float("inf"))
    topk = sim.topk(min(k, sim.shape[0] - 1), dim=-1).indices
    match = (labels[topk] == labels.unsqueeze(1)).float().mean(dim=-1)
    return match.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-col", choices=["verb", "obj", "subj"], default="verb")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed()
    device = get_device()

    role_map = _sample_id_to_role_map()
    counts = Counter(role_map[args.label_col].to_list())
    top_classes = {c for c, _ in counts.most_common(args.top_k)}
    class_to_idx = {c: i for i, c in enumerate(sorted(top_classes))}
    logger.info(f"Top-{args.top_k} '{args.label_col}' classes: {sorted(top_classes)}")

    splits = {}
    for split in ["train", "val", "test"]:
        diagrams, symbols_list, labels, texts = _load_split(split, args.label_col, top_classes, class_to_idx, role_map)
        splits[split] = {"diagrams": diagrams, "symbols": symbols_list, "labels": labels, "texts": texts}
        logger.info(f"{split}: {len(labels)} rows after top-{args.top_k} filter")

    num_classes = len(class_to_idx)

    # --- EinsumModel + linear classifier ---
    # Reuse collect_symbol_sizes (as every other training script does) rather than
    # re-deriving per-symbol shapes by hand — it already reads the stored (Symbol, size)
    # pairs correctly from the same parquet files.
    vlm_datasets = [
        VLMDataset(
            constants.datasets_path / f"svo_{split}_probes.parquet",
            compiled_columns=[("diagram", "symbols", "caption")],
        )
        for split in ["train", "val", "test"]
    ]
    symbols, sizes = collect_symbol_sizes(
        vlm_datasets, ["symbols"], remap={constants.embedding_dim: args.embedding_dim}
    )
    logger.info(f"Collected {len(symbols)} unique symbols across all splits.")

    model = CaptionClassifier(symbols, sizes, num_classes, args.embedding_dim, non_linear=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(
        CaptionDataset(splits["train"]["diagrams"], splits["train"]["symbols"], splits["train"]["labels"]),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_captions,
    )
    val_loader = DataLoader(
        CaptionDataset(splits["val"]["diagrams"], splits["val"]["symbols"], splits["val"]["labels"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_captions,
    )
    test_loader = DataLoader(
        CaptionDataset(splits["test"]["diagrams"], splits["test"]["symbols"], splits["test"]["labels"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_captions,
    )

    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4, minimize=False)
    best_state = None
    for epoch in range(1, args.epochs + 1):
        _, train_acc = _run_epoch(model, train_loader, optimizer, device, train=True)
        _, val_acc = _run_epoch(model, val_loader, optimizer, device, train=False)
        logger.info(f"[einsum] epoch {epoch:02d} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        status = early_stopping(val_acc)
        if status == ModelTrainingStatus.improved:
            # EinsumModel.state_dict() injects non-tensor symbols_list/sizes_list
            # entries (see ContrastiveVLM.load_state_dict's docstring for why) --
            # filter to real tensors; restoring parameter values within this same
            # live model instance doesn't need those two entries, since symbols/
            # sizes don't change during training.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if torch.is_tensor(v)}
        elif status == ModelTrainingStatus.stop:
            logger.info(f"[einsum] early stopping at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    _, test_acc = _run_epoch(model, test_loader, optimizer, device, train=False)

    majority = Counter(splits["test"]["labels"]).most_common(1)[0][1] / len(splits["test"]["labels"])

    model.eval()
    with torch.no_grad():
        test_embs = torch.cat(
            [model.embed(captions).cpu() for captions, _ in test_loader],
        )
    test_labels_t = torch.tensor(splits["test"]["labels"])
    knn = _knn_consistency(test_embs, test_labels_t)

    logger.info(f"[einsum] FINAL test_acc={test_acc:.4f} majority={majority:.4f} kNN_cons={knn:.4f}")

    # --- Bag-of-words logistic regression floor ---
    vocab_counter: Counter = Counter()
    for text in splits["train"]["texts"]:
        vocab_counter.update(_WORD_RE.findall(text.lower()))
    vocab = {w: i for i, (w, _) in enumerate(vocab_counter.most_common(2000))}
    logger.info(f"BoW vocab size: {len(vocab)}")

    bow_model = BagOfWordsClassifier(len(vocab), num_classes).to(device)
    bow_optimizer = torch.optim.Adam(bow_model.parameters(), lr=1e-3, weight_decay=1e-4)
    bow_train = DataLoader(
        BowDataset(splits["train"]["texts"], splits["train"]["labels"], vocab), batch_size=args.batch_size, shuffle=True
    )
    bow_val = DataLoader(
        BowDataset(splits["val"]["texts"], splits["val"]["labels"], vocab), batch_size=args.batch_size, shuffle=False
    )
    bow_test = DataLoader(
        BowDataset(splits["test"]["texts"], splits["test"]["labels"], vocab), batch_size=args.batch_size, shuffle=False
    )
    bow_early = EarlyStopping(patience=args.patience, min_delta=1e-4, minimize=False)
    bow_best = None
    for epoch in range(1, args.epochs + 1):
        _, train_acc = _run_epoch(bow_model, bow_train, bow_optimizer, device, train=True, is_bow=True)
        _, val_acc = _run_epoch(bow_model, bow_val, bow_optimizer, device, train=False, is_bow=True)
        logger.info(f"[bow] epoch {epoch:02d} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        status = bow_early(val_acc)
        if status == ModelTrainingStatus.improved:
            bow_best = {k: v.detach().cpu().clone() for k, v in bow_model.state_dict().items()}
        elif status == ModelTrainingStatus.stop:
            logger.info(f"[bow] early stopping at epoch {epoch}")
            break
    if bow_best is not None:
        bow_model.load_state_dict(bow_best)
    _, bow_test_acc = _run_epoch(bow_model, bow_test, bow_optimizer, device, train=False, is_bow=True)

    sep = "=" * 70
    logger.info(sep)
    logger.info("TEXT-TOWER CAPACITY PROBE — FINAL RESULTS")
    logger.info(sep)
    logger.info(f"label_col={args.label_col} top_k={args.top_k} num_classes={num_classes}")
    logger.info(f"{'model':<12}{'test_acc':>10}{'majority':>10}")
    logger.info(f"{'einsum':<12}{test_acc:>10.4f}{majority:>10.4f}")
    logger.info(f"{'bow_logreg':<12}{bow_test_acc:>10.4f}{majority:>10.4f}")
    logger.info(f"kNN class consistency (einsum embeddings, k=10): {knn:.4f}  (chance={1/num_classes:.4f})")
    logger.info(sep)


if __name__ == "__main__":
    main()
