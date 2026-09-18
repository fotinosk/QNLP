"""
Where in the TTN image tower does between-image variation die?

Runs real photos through `TTNImageModel` and reports, at every internal
stage, the mean pairwise cosine between *different* images' activations.
Near 1.0 at a stage means all images look alike from there on.

Run it with no checkpoint to trace a freshly initialised tower, and with
`--checkpoint` to trace a trained one. The comparison answers a specific
question: is a collapsed image tower (see `image_ablation.py`) an
architectural property of the multilinear quadtree, or something training
drives it into? Random init separating images fine means the capacity is
there and the objective is at fault.

Usage:
    python -m qnlp.discoviz.diagnostic.tower_spread_trace
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
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams


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


def load_cifar_images(data_root: str, n: int) -> torch.Tensor:
    """Stage 0.4 (TTN_CIFAR_EXPERIMENTS.md): trace on the exact dataset the
    supervised probe uses, at whatever image_size/patch_size the caller has
    set via IMAGE_MODEL_* env vars (e.g. 32x32 for the CIFAR-10 gates)."""
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
    return torch.stack([ds[i][0] for i in idx])


def spread(tensor: torch.Tensor, label: str) -> None:
    """Mean pairwise cosine between different images at this stage."""
    vecs = F.normalize(tensor.flatten(1), dim=-1)
    sim = vecs @ vecs.t()
    off_diag = sim[~torch.eye(len(vecs), dtype=torch.bool)]
    print(f"  {label:<34} pairwise cos mean {off_diag.mean():.4f}")


def trace(model: TTNImageModel, x: torch.Tensor) -> None:
    """Mirrors TTNImageModel.forward, measuring spread between every stage."""
    with torch.no_grad():
        spread(x, "raw pixels (ImageNet-normalised)")
        spread(x - x.mean(0, keepdim=True), "raw pixels, dataset-mean-centred")

        patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) c (p1 p2)", p1=model.patch_size, p2=model.patch_size)
        c_feat = torch.einsum("bncp, ck -> bnk", patches, model.color_factor)
        p_feat = torch.einsum("bncp, pk -> bnk", patches, model.pixel_factor)
        spread(c_feat, "after colour projection (linear)")
        spread(p_feat, "after pixel projection (linear)")

        h = c_feat * p_feat
        spread(h, "after bilinear product c*p")
        h = h + (model.positional_embedding * model.pos_scale)
        spread(h, "+ positional embedding")

        grid = int(math.sqrt(h.shape[1]))
        for i, layer in enumerate(model.layers):
            h = rearrange(h, "b (h w) c -> b c h w", h=grid)
            h = rearrange(h, "b c (h h2) (w w2) -> b (h w) (h2 w2) c", h2=2, w2=2)
            h = layer(h)
            grid //= 2
            spread(h, f"after quadtree layer {i}")

        spread(model.head(model.final_norm(h.squeeze(1))), "after final_norm + head (output)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None, help="omit to trace a random-init tower")
    parser.add_argument("--parquet", type=Path, default=constants.datasets_path / "aro_test.parquet")
    parser.add_argument("--column", default="local_image_path")
    parser.add_argument("--cifar-root", type=str, default=None, help="trace CIFAR-10 instead of --parquet")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("-n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x = (
        load_cifar_images(args.cifar_root, args.n)
        if args.cifar_root
        else load_images(args.parquet, args.column, args.n)
    )

    embedding_dim = args.embedding_dim
    state_dict = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        full = checkpoint["model_state_dict"]
        state_dict = {k[len("image_model.") :]: v for k, v in full.items() if k.startswith("image_model.")}
        embedding_dim = state_dict["head.weight"].shape[0]

    model = TTNImageModel(embedding_dim)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()

    which = f"trained ({args.checkpoint})" if args.checkpoint else "RANDOM INIT"
    print(f"{len(x)} real photos, input {tuple(x.shape)} | tower: {which} | embedding_dim={embedding_dim}")
    trace(model, x)


if __name__ == "__main__":
    main()
