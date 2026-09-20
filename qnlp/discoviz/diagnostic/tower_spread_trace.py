"""
Where in the TTN image tower does between-image variation die?

Runs real photos through `TTNImageModel` and reports, at every internal
stage, three metrics comparing that stage's representation to the input:

  - mean cos:   mean pairwise cosine between different images' activations.
                Near 1.0 means all images look alike from here on — a
                COLLAPSE detector. This is the only metric the original
                version of this script had, and it turns out to be
                degenerate for the "is class structure preserved" question:
                a faithful embedding of diverse images and pure noise both
                read close to 0 once the input itself has low mean cosine
                (as CIFAR-10 does at 32x32, ~0.005-0.015) — see
                TTN_CIFAR_EXPERIMENTS.md's "B1 confirmed, and the damage
                localised" for the full story of why this was misleading.
  - Gram corr:  Pearson correlation between this stage's pairwise-cosine
                matrix and the INPUT's. 1.0 = this stage preserves which
                images are similar/dissimilar to which; 0 = that geometry
                is destroyed, independent of collapse.
  - kNN cons:   fraction of each image's k nearest neighbours (by cosine,
                at this stage) that share its true class label. Chance is
                1/num_classes. This is the metric that actually answers
                "does class information survive to this point" — requires
                --cifar-root (labels aren't available for the ARO/--parquet
                path, which reports "n/a" for this column).

Run it with no checkpoint to trace a freshly initialised tower, and with
`--checkpoint` to trace a trained one.

Usage:
    python -m qnlp.discoviz.diagnostic.tower_spread_trace --cifar-root data/cifar10
    python -m qnlp.discoviz.diagnostic.tower_spread_trace --checkpoint <path>
"""

import argparse
import math
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torchvision import transforms

from qnlp.constants import constants
from qnlp.discoviz.models.image_model import build_image_model, image_model_hyperparams


def load_images(parquet: Path, column: str, n: int) -> torch.Tensor:
    size = image_model_hyperparams.image_size
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    paths = pl.read_parquet(parquet)[column].to_list()[:n]
    return torch.stack(
        [
            transform(torchvision.io.read_image(p, mode=torchvision.io.ImageReadMode.RGB).float().div(255.0))
            for p in paths
        ]
    )


def load_cifar_images(data_root: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 0.4 (TTN_CIFAR_EXPERIMENTS.md): trace on the exact dataset the
    supervised probe uses, at whatever image_size/patch_size the caller has
    set via IMAGE_MODEL_* env vars (e.g. 32x32 for the CIFAR-10 gates).
    Returns (images, labels) — labels enable the kNN class-consistency metric."""
    size = image_model_hyperparams.image_size
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    images = torch.stack([ds[i][0] for i in idx])
    labels = torch.tensor([ds.targets[i] for i in idx])
    return images, labels


def _pairwise_cos(tensor: torch.Tensor) -> torch.Tensor:
    vecs = F.normalize(tensor.flatten(1), dim=-1)
    return vecs @ vecs.t()


def _gram_corr(sim: torch.Tensor, ref_sim: torch.Tensor) -> float:
    """Pearson correlation between two pairwise-cosine matrices' off-diagonal
    entries. 1.0 = this stage's similarity structure matches the reference
    (usually the input) exactly; 0 = no relationship."""
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
    a, b = sim[mask], ref_sim[mask]
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return ((a * b).sum() / denom).item() if denom > 0 else float("nan")


def _knn_consistency(sim: torch.Tensor, labels: torch.Tensor | None, k: int = 10) -> float:
    """Fraction of each point's k nearest neighbours (by cosine, this stage)
    sharing its true label. None if no labels are available."""
    if labels is None:
        return float("nan")
    sim = sim.clone()
    sim.fill_diagonal_(-float("inf"))
    topk = sim.topk(min(k, sim.shape[0] - 1), dim=-1).indices
    neighbor_labels = labels[topk]
    match = (neighbor_labels == labels.unsqueeze(1)).float().mean(dim=-1)
    return match.mean().item()


def _report(tensor: torch.Tensor, label: str, ref_sim: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
    """Print mean cos / Gram corr / kNN cons for this stage, return its
    similarity matrix (becomes the new `ref_sim` for the next call if the
    caller wants stage-to-stage comparisons instead of stage-to-input)."""
    sim = _pairwise_cos(tensor)
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
    mean_cos = sim[mask].mean().item()
    gram = _gram_corr(sim, ref_sim)
    knn = _knn_consistency(sim, labels)
    knn_str = f"{knn:.4f}" if not math.isnan(knn) else "n/a"
    print(f"  {label:<34} mean cos {mean_cos:>8.4f}   Gram corr {gram:>8.4f}   kNN cons {knn_str:>8}")
    return sim


def trace(
    model: torch.nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor | None = None,
    mean_center: bool = False,
    zero_pos_scale: bool = False,
) -> None:
    """Mirrors TTNImageModel.forward, measuring mean cos / Gram corr / kNN
    consistency at every stage against the INPUT (not the previous stage) —
    Gram corr and kNN cons only fall off from 1.0/input-kNN when a stage
    genuinely discards information, regardless of how the previous stage
    looked, which is what makes them decisive where mean cos was not.

    mean_center (Stage A4) and zero_pos_scale (Stage A5) are cheap
    trace-only ablations — they don't require retraining, just show
    whether either change would help the random-init representation on
    its own."""
    with torch.no_grad():
        ref_sim = _pairwise_cos(x)
        _report(x, "raw pixels (ImageNet-normalised)", ref_sim, labels)
        if mean_center:
            x = x - x.mean(0, keepdim=True)
            _report(x, "raw pixels, dataset-mean-centred [A4 applied]", ref_sim, labels)
        else:
            _report(x - x.mean(0, keepdim=True), "raw pixels, dataset-mean-centred", ref_sim, labels)

        if getattr(model, "use_b1_feature_map", False):
            x01 = (x * model._pixel_std + model._pixel_mean).clamp(0.0, 1.0)
            patches = rearrange(x01, "b c (h p1) (w p2) -> b (h w) c (p1 p2)", p1=model.patch_size, p2=model.patch_size)
            phi = torch.stack([torch.cos(math.pi / 2 * patches), torch.sin(math.pi / 2 * patches)], dim=-1)
            _report(phi.flatten(2), "after phi(x)=[cos,sin] feature map [B1]", ref_sim, labels)
            h = model.feature_proj(phi.flatten(2))
            _report(h, "after feature_proj (linear) [B1]", ref_sim, labels)
        else:
            patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) c (p1 p2)", p1=model.patch_size, p2=model.patch_size)
            c_feat = torch.einsum("bncp, ck -> bnk", patches, model.color_factor)
            p_feat = torch.einsum("bncp, pk -> bnk", patches, model.pixel_factor)
            _report(c_feat, "after colour projection (linear)", ref_sim, labels)
            _report(p_feat, "after pixel projection (linear)", ref_sim, labels)

            h = c_feat * p_feat
            _report(h, "after bilinear product c*p", ref_sim, labels)
        pos_scale = 0.0 if zero_pos_scale else model.pos_scale
        h = h + (model.positional_embedding * pos_scale)
        _report(h, "+ positional embedding" + (" [A5: pos_scale=0]" if zero_pos_scale else ""), ref_sim, labels)

        grid = int(math.sqrt(h.shape[1]))
        for i, layer in enumerate(model.layers):
            h = rearrange(h, "b (h w) c -> b c h w", h=grid)
            h = rearrange(h, "b c (h h2) (w w2) -> b (h w) (h2 w2) c", h2=2, w2=2)
            h = layer(h)
            grid //= 2
            _report(h, f"after quadtree layer {i}", ref_sim, labels)

        _report(model.head(model.final_norm(h.squeeze(1))), "after final_norm + head (output)", ref_sim, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None, help="omit to trace a random-init tower")
    parser.add_argument("--parquet", type=Path, default=constants.datasets_path / "aro_test.parquet")
    parser.add_argument("--column", default="local_image_path")
    parser.add_argument("--cifar-root", type=str, default=None, help="trace CIFAR-10 instead of --parquet")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("-n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mean-center", action="store_true", help="Stage A4 ablation")
    parser.add_argument("--zero-pos-scale", action="store_true", help="Stage A5 ablation")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    labels = None
    if args.cifar_root:
        x, labels = load_cifar_images(args.cifar_root, args.n)
    else:
        x = load_images(args.parquet, args.column, args.n)

    embedding_dim = args.embedding_dim
    state_dict = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        full = checkpoint["model_state_dict"]
        state_dict = {k[len("image_model.") :]: v for k, v in full.items() if k.startswith("image_model.")}
        embedding_dim = state_dict["head.weight"].shape[0]

    model = build_image_model(embedding_dim)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()

    which = f"trained ({args.checkpoint})" if args.checkpoint else "RANDOM INIT"
    has_labels = "yes" if labels is not None else "no"
    print(
        f"{len(x)} real photos, input {tuple(x.shape)} | tower: {which} | embedding_dim={embedding_dim} "
        f"| mean_center={args.mean_center} | zero_pos_scale={args.zero_pos_scale} | labels={has_labels}"
    )
    trace(model, x, labels=labels, mean_center=args.mean_center, zero_pos_scale=args.zero_pos_scale)


if __name__ == "__main__":
    main()
