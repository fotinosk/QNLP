"""
Supervised capacity probe for TTNImageModel — no text tower, no contrastive
loss, no CCG. Trains TTNImageModel + a linear classifier end-to-end with
plain cross-entropy on a real classification task, and compares it against
a small CNN and a from-scratch ResNet-18 on the IDENTICAL task.

Rationale: nineteen SVO training runs (see SVO_EXPERIMENTS.md) have landed
at chance regardless of loss, hyperparameters, or augmentation, which is
consistent with either (a) the contrastive setup wasting a perfectly good
image tower, or (b) the tower itself lacking the capacity to ground images
at all — a question none of those runs can distinguish, since a contrastive
objective confounds "can the tower represent images" with "does this loss
recover that representation." This probe removes every other suspect
(text tower, CCG, contrastive/triplet loss) and asks the capacity question
directly and cheaply.

Two datasets:
  --dataset cifar10  CIFAR-10, upsampled to 64x64 (TTNImageModel's native
                      input size). An unambiguous, standard classification
                      task independent of anything SVO-specific.
  --dataset svo       SVO-Probes' own images, labeled with their `obj`
                      annotation (the paper's ground-truth object noun for
                      each image), restricted to the top-K most frequent
                      object classes among unique positive images (K=20 by
                      default; pass --label-col verb for the verb variant).
                      Uses the exact same photographs the contrastive runs
                      trained on, just with a supervised objective.

Usage:
    python -m qnlp.discoviz.diagnostic.ttn_supervised_probe --dataset cifar10 --arch all
    python -m qnlp.discoviz.diagnostic.ttn_supervised_probe --dataset svo --arch all
    python -m qnlp.discoviz.diagnostic.ttn_supervised_probe --dataset both --arch all
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import resnet18

from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.utils.logging import setup_logger
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="ttn_supervised_probe")

IMAGE_SIZE = image_model_hyperparams.image_size  # 64
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

SVO_DIR = Path("data/svo/raw").resolve()
SVO_CSV = SVO_DIR / "svo_probes_corrected.csv"
SVO_IMAGE_DIRS = [SVO_DIR / "images", SVO_DIR / "images_old"]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class TTNClassifier(nn.Module):
    """TTNImageModel (the exact tower used in every SVO/ARO/COCO contrastive
    run) + a plain linear classifier on top. Nothing else.

    Reads the PRE-L2-normalisation head output (normalize=False) — see
    TTN_CIFAR_EXPERIMENTS.md Stage 0.2. Contrastive training only ever
    needs the L2-normalised embedding (cosine similarity is scale
    invariant); a softmax classifier is not scale-invariant, and a TN
    classifier's output magnitude can carry class signal that L2-norm
    would discard before the linear head ever sees it."""

    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.backbone = TTNImageModel(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x, normalize=False))


class SmallCNN(nn.Module):
    """4-block conv classifier, roughly comparable capacity to TTNClassifier
    — the "small CNN" baseline."""

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        channels = [in_channels, 32, 64, 128, 256]
        blocks = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def _build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "ttn":
        return TTNClassifier(num_classes)
    if arch == "cnn":
        return SmallCNN(num_classes)
    if arch == "resnet18":
        return resnet18(weights=None, num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch!r}")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def _cifar10_datasets(data_root: str) -> tuple[Dataset, Dataset, Dataset, int]:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            NORMALIZE,
        ]
    )
    train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    val_idx, train_idx = perm[:5000], perm[5000:]
    train = Subset(train_full, train_idx)
    val = Subset(train_full, val_idx)
    return train, val, test, 10


def _resolve_svo_image_path(image_id: str) -> str | None:
    for image_dir in SVO_IMAGE_DIRS:
        candidate = image_dir / f"{image_id}.jpg"
        if candidate.exists():
            return str(candidate)
    return None


class SVOImageDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int]):
        self.paths = paths
        self.labels = labels
        self.transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), NORMALIZE])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = torchvision.io.read_image(self.paths[idx], mode=torchvision.io.ImageReadMode.RGB).float().div(255.0)
        return self.transform(img), self.labels[idx]


def _svo_datasets(label_col: str, top_k: int) -> tuple[Dataset, Dataset, Dataset, int]:
    df = pl.read_csv(SVO_CSV)
    # One row per unique positive image — many rows share a pos image (it can
    # appear as several different rows' negative elsewhere), and we want a
    # standard one-image-one-label classification set, not a caption-weighted one.
    df = df.unique(subset=["pos_image_id"], keep="first").sort("pos_image_id")

    counts = Counter(df[label_col].to_list())
    top_classes = [cls for cls, _ in counts.most_common(top_k)]
    class_to_idx = {cls: i for i, cls in enumerate(top_classes)}
    logger.info(f"Top-{top_k} '{label_col}' classes: {top_classes}")

    df = df.filter(pl.col(label_col).is_in(top_classes))

    paths, labels = [], []
    for image_id, label in zip(df["pos_image_id"].cast(pl.String).to_list(), df[label_col].to_list()):
        path = _resolve_svo_image_path(image_id)
        if path is not None:
            paths.append(path)
            labels.append(class_to_idx[label])

    logger.info(f"SVO '{label_col}' top-{top_k}: {len(paths)} labeled images with resolvable files.")

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(paths), generator=g).tolist()
    n = len(perm)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    ds = SVOImageDataset(paths, labels)
    return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx), len(top_classes)


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------


def _majority_baseline(loader: DataLoader) -> float:
    counts: Counter[int] = Counter()
    total = 0
    for _, labels in loader:
        counts.update(labels.tolist())
        total += len(labels)
    return max(counts.values()) / total if total else float("nan")


def _run_epoch(model: nn.Module, loader: DataLoader, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def train_and_eval(
    arch: str,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    num_classes: int,
    device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
) -> dict:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = _build_model(arch, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc, best_state, epochs_since_best = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, optimizer, device, train=False)
        logger.info(
            f"[{arch}] epoch {epoch:02d} ({time.time() - t0:.1f}s) "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                logger.info(f"[{arch}] early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break

    model.load_state_dict(best_state)
    _, test_acc = _run_epoch(model, test_loader, optimizer, device, train=False)
    majority = _majority_baseline(test_loader)

    logger.info(f"[{arch}] FINAL test_acc={test_acc:.4f} majority_baseline={majority:.4f} params={n_params}")
    return {"arch": arch, "test_acc": test_acc, "majority_baseline": majority, "params": n_params}


def run_overfit_gate(
    arch: str,
    train_ds: Dataset,
    num_classes: int,
    device,
    n: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    """Stage 0.3 (TTN_CIFAR_EXPERIMENTS.md): can this model memorise a tiny
    fixed subset at all? No val/test, no early stopping, no augmentation
    (the datasets built above already apply none). Failing to reach ~100%
    train accuracy on n=500 examples indicates an optimisation/conditioning
    problem, not a capacity problem — a different fix than a val number
    near chance would suggest on its own."""
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(train_ds), generator=g)[:n].tolist()
    subset = Subset(train_ds, idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = _build_model(arch, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, loader, optimizer, device, train=True)
        best_acc = max(best_acc, train_acc)
        if epoch % 10 == 0 or epoch == epochs or train_acc >= 0.999:
            logger.info(f"[{arch}] overfit-{n} epoch {epoch:03d} loss={train_loss:.4f} train_acc={train_acc:.4f}")
        if train_acc >= 0.999:
            logger.info(f"[{arch}] overfit-{n} gate PASSED at epoch {epoch} (train_acc={train_acc:.4f})")
            return {"arch": arch, "overfit_n": n, "final_train_acc": train_acc, "epochs_used": epoch, "passed": True}

    logger.info(f"[{arch}] overfit-{n} gate FAILED after {epochs} epochs (best train_acc={best_acc:.4f})")
    return {"arch": arch, "overfit_n": n, "final_train_acc": best_acc, "epochs_used": epochs, "passed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["cifar10", "svo", "both"], default="both")
    parser.add_argument("--arch", choices=["ttn", "cnn", "resnet18", "all"], default="all")
    parser.add_argument("--label-col", choices=["obj", "verb"], default="obj")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--data-root", type=str, default="data/cifar10")
    parser.add_argument(
        "--overfit-n",
        type=int,
        default=0,
        help="Stage 0.3 gate: if >0, subsample this many train examples per dataset and check whether the "
        "model can memorise them (~100%% train acc), skipping val/test/early-stopping entirely.",
    )
    parser.add_argument("--overfit-epochs", type=int, default=200)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    archs = ["ttn", "cnn", "resnet18"] if args.arch == "all" else [args.arch]
    datasets = ["cifar10", "svo"] if args.dataset == "both" else [args.dataset]

    results = []
    for dataset_name in datasets:
        if dataset_name == "cifar10":
            train_ds, val_ds, test_ds, num_classes = _cifar10_datasets(args.data_root)
        else:
            train_ds, val_ds, test_ds, num_classes = _svo_datasets(args.label_col, args.top_k)
        logger.info(
            f"=== dataset={dataset_name} ({'svo/' + args.label_col if dataset_name == 'svo' else 'cifar10'}) "
            f"classes={num_classes} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} ==="
        )
        for arch in archs:
            if args.overfit_n > 0:
                r = run_overfit_gate(
                    arch, train_ds, num_classes, device, args.overfit_n, args.overfit_epochs, args.batch_size, args.lr
                )
            else:
                r = train_and_eval(
                    arch,
                    train_ds,
                    val_ds,
                    test_ds,
                    num_classes,
                    device,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    args.patience,
                )
            r["dataset"] = dataset_name
            results.append(r)

    sep = "=" * 70
    logger.info(sep)
    if args.overfit_n > 0:
        logger.info("OVERFIT GATE — FINAL RESULTS")
        logger.info(sep)
        logger.info(f"{'dataset':<10}{'arch':<12}{'passed':>8}{'train_acc':>12}{'epochs':>8}")
        for r in results:
            logger.info(
                f"{r['dataset']:<10}{r['arch']:<12}{str(r['passed']):>8}"
                f"{r['final_train_acc']:>12.4f}{r['epochs_used']:>8}"
            )
    else:
        logger.info("SUPERVISED CAPACITY PROBE — FINAL RESULTS")
        logger.info(sep)
        logger.info(f"{'dataset':<10}{'arch':<12}{'test_acc':>10}{'majority':>10}{'params':>14}")
        for r in results:
            logger.info(
                f"{r['dataset']:<10}{r['arch']:<12}{r['test_acc']:>10.4f}{r['majority_baseline']:>10.4f}{r['params']:>14,}"
            )
    logger.info(sep)


if __name__ == "__main__":
    main()
