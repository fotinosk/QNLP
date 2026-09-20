"""
NODE_ARCHITECTURE_PLAN.md's prerequisite measurement (no cluster job):
excess kurtosis of `merged` (the 4-way Hadamard product inside
CPQuadRankLayer.forward), per layer, on a trained checkpoint.

A product of four unit-RMS vectors is heavy-tailed in expectation: most
components collapse toward zero and a few dominate. This has been
hypothesised four times in the research log and never measured. It
decides whether NODE-2 (degree reduction) is well motivated:
  - heavy-tailed (excess kurtosis >> 0, e.g. >> 3 for a Laplace-like tail)
    -> degree reduction is the right fix, run NODE-2.
  - near-Gaussian (excess kurtosis ~ 0) -> NODE-2's rationale evaporates;
    weight NODE-1 and NODE-3 instead.

Usage:
    python -m qnlp.discoviz.diagnostic.node_kurtosis \
        --checkpoint runs/checkpoints/ttn_supervised_probe/backbone_ttn_nonlin-gelu_best.pt \
        --cifar-root data/cifar10
"""

import argparse
from pathlib import Path

import torch
import torchvision
from torchvision import transforms

from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams


def _excess_kurtosis(x: torch.Tensor) -> float:
    """Fisher (excess) kurtosis: 0 for a Gaussian, positive for heavy tails."""
    x = x.flatten().double()
    x = x - x.mean()
    var = (x**2).mean()
    m4 = (x**4).mean()
    return (m4 / (var**2) - 3.0).item()


def load_cifar_images(data_root: str, n: int, seed: int) -> torch.Tensor:
    size = image_model_hyperparams.image_size
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return torch.stack([ds[i][0] for i in idx])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="ttn_supervised_probe.py backbone checkpoint")
    parser.add_argument("--cifar-root", type=str, required=True)
    parser.add_argument("-n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x = load_cifar_images(args.cifar_root, args.n, args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_sd = checkpoint["backbone_state_dict"]
    embedding_dim = backbone_sd["head.weight"].shape[0]
    model = TTNImageModel(embedding_dim)
    model.load_state_dict(backbone_sd, strict=True)
    model.eval()

    print(
        f"{len(x)} CIFAR-10 images | checkpoint: {args.checkpoint} "
        f"(test_acc={checkpoint.get('test_acc', '?')}) | embedding_dim={embedding_dim} "
        f"| nonlinearity={image_model_hyperparams.nonlinearity}"
    )
    with torch.no_grad():
        model(x, normalize=False)

    print("\nExcess kurtosis of `merged` per layer (0 = Gaussian, >0 = heavy-tailed):")
    for i, layer in enumerate(model.layers):
        merged = layer._last_merged
        k = _excess_kurtosis(merged)
        print(f"  layer {i}: excess kurtosis = {k:8.3f}   (merged shape {tuple(merged.shape)})")


if __name__ == "__main__":
    main()
