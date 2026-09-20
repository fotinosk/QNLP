"""
Does the image tower actually contribute to a hard-negative benchmark score?

Two checks on a trained ContrastiveVLM checkpoint:

1. **Ablation.** Re-score the benchmark with the image corrupted, and compare
   against the real-image score.
     - ARO (fixed image, true/false caption): replace each row's image with
       some other row's image ("shuffled"), and with an all-zero image
       ("zeros"). If accuracy holds up, the decision is being made entirely by
       the text tower and the benchmark never tested visual grounding.
     - SVO (fixed caption, true/false image): the roles are swapped, so the
       corresponding ablation is on the caption — score each image pair
       against some other row's caption. Here accuracy SHOULD collapse to
       chance; if it doesn't, the model is reading something about the images
       that is independent of the caption.

2. **Image embedding spread.** Mean pairwise cosine between the embeddings of
   different images. Near 1.0 means the tower has collapsed to a constant
   vector — it emits the same embedding whatever it is shown.

Usage:
    python -m qnlp.discoviz.diagnostic.image_ablation <checkpoint> --task aro
    python -m qnlp.discoviz.diagnostic.image_ablation <checkpoint> --task svo \
        --parquet data/datasets/svo_test_probes.parquet -n 1024
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from qnlp.constants import constants
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import build_image_model, image_model_hyperparams
from qnlp.domain.datasets.dataloader import vlm_collate_fn
from qnlp.domain.datasets.dataset import VLMDataset
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM

ARO_COMPILED_COLUMNS = [
    ("true_diagram", "true_symbols", "true_caption", "true_path"),
    ("false_diagram", "false_symbols", "false_caption", "false_path"),
]
SVO_COMPILED_COLUMNS = [("diagram", "symbols", "caption", "path")]
SVO_IMAGE_COLUMNS = ["true_local_image_path", "false_local_image_path"]


def load_model(checkpoint_path: Path) -> ContrastiveVLM:
    """Rebuild the model from a checkpoint, inferring every architecture flag
    from the state dict (NLC, embedding_dim, which head — if any — was used)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    non_linear = bool(state_dict.get("non_linear_contractions", False))
    embedding_dim = state_dict["image_model.head.weight"].shape[0]
    use_mlp = "image_head.net.0.weight" in state_dict
    use_proj = "image_head.proj.weight" in state_dict

    model = ContrastiveVLM(
        EinsumModel(non_linear_contractions=non_linear),
        build_image_model(embedding_dim),
        embedding_dim=embedding_dim,
        use_mlp_head=use_mlp,
        use_projection_head=use_proj,
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(
        f"ckpt epoch={checkpoint.get('epoch', '?')} embedding_dim={embedding_dim} "
        f"nlc={non_linear} proj_head={use_proj} mlp_head={use_mlp}"
    )
    return model


def make_loader(parquet: Path, task: str, non_linear: bool, n: int, batch_size: int) -> DataLoader:
    size = image_model_hyperparams.image_size
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    svo = task == "svo"
    ds = VLMDataset(
        parquet,
        image_columns=SVO_IMAGE_COLUMNS if svo else None,
        compiled_columns=SVO_COMPILED_COLUMNS if svo else ARO_COMPILED_COLUMNS,
        image_transform=transform,
        use_non_linear_contractions=non_linear,
    )
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:n].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, collate_fn=vlm_collate_fn)


def _report(results: dict[str, list[bool]], cosines: dict[str, list[float]]) -> None:
    print(f"\n{'variant':<12}{'N':>7}{'hard_neg_acc':>14}{'true_cos':>10}{'false_cos':>11}")
    for name, correct in results.items():
        n = len(correct)
        if not n:
            continue
        print(f"{name:<12}{n:>7}{sum(correct) / n:>14.4f}{cosines[name][0] / n:>10.4f}{cosines[name][1] / n:>11.4f}")


def ablate_aro(model: ContrastiveVLM, loader: DataLoader) -> None:
    """Fixed image, true/false caption — corrupt the IMAGE."""
    known = set(model.text_model.sym2weight.keys())
    results = {k: [] for k in ("real", "shuffled", "zeros")}
    cosines = {k: [0.0, 0.0] for k in results}

    with torch.no_grad():
        for batch in loader:
            true_caps, false_caps = batch["true_caption"], batch["false_caption"]
            valid = [
                i
                for i in range(len(true_caps))
                if all(s in known for s in true_caps[i][1]) and all(s in known for s in false_caps[i][1])
            ]
            if not valid:
                continue
            images = batch["local_image_path"][valid]
            tc = [true_caps[i] for i in valid]
            fc = [false_caps[i] for i in valid]

            variants = {
                "real": images,
                "shuffled": images[torch.roll(torch.arange(len(valid)), 1)],
                "zeros": torch.zeros_like(images),
            }
            for name, imgs in variants.items():
                out = model(imgs, tc, fc)
                pos = F.cosine_similarity(out["true_caption_embeddings"], out["image_embeddings"])
                neg = F.cosine_similarity(out["false_caption_embeddings"], out["image_embeddings"])
                ok = torch.isfinite(pos) & torch.isfinite(neg)
                results[name] += (pos[ok] > neg[ok]).tolist()
                cosines[name][0] += pos[ok].sum().item()
                cosines[name][1] += neg[ok].sum().item()

    _report(results, cosines)


def ablate_svo(model: ContrastiveVLM, loader: DataLoader) -> None:
    """Fixed caption, true/false image — roles are swapped, so corrupt the CAPTION."""
    known = set(model.text_model.sym2weight.keys())
    results = {k: [] for k in ("real", "shuffled_caption")}
    cosines = {k: [0.0, 0.0] for k in results}

    with torch.no_grad():
        for batch in loader:
            caps = batch["caption"]
            valid = [i for i in range(len(caps)) if all(s in known for s in caps[i][1])]
            if len(valid) < 2:
                continue
            true_images = batch["true_local_image_path"][valid]
            false_images = batch["false_local_image_path"][valid]
            real_caps = [caps[i] for i in valid]
            rolled_caps = real_caps[-1:] + real_caps[:-1]

            for name, cap_batch in (("real", real_caps), ("shuffled_caption", rolled_caps)):
                true_out = model(true_images, cap_batch)
                false_out = model(false_images, cap_batch)
                cap_emb = true_out["true_caption_embeddings"]
                pos = F.cosine_similarity(cap_emb, true_out["image_embeddings"])
                neg = F.cosine_similarity(cap_emb, false_out["image_embeddings"])
                ok = torch.isfinite(pos) & torch.isfinite(neg)
                results[name] += (pos[ok] > neg[ok]).tolist()
                cosines[name][0] += pos[ok].sum().item()
                cosines[name][1] += neg[ok].sum().item()

    _report(results, cosines)


def image_spread(model: ContrastiveVLM, loader: DataLoader, image_column: str) -> None:
    """Mean pairwise cosine between different images' embeddings — near 1.0 is
    a collapsed tower emitting one constant vector."""
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            embeddings.append(model.image_head(model.image_model(batch[image_column])))
    emb = F.normalize(torch.cat(embeddings), dim=-1)
    sim = emb @ emb.t()
    off_diag = sim[~torch.eye(len(emb), dtype=torch.bool)]
    print(f"\nimage embeddings: n={len(emb)}")
    print(
        f"  pairwise cosine between DIFFERENT images: mean {off_diag.mean():.4f}  "
        f"min {off_diag.min():.4f}  std {off_diag.std():.4f}"
    )
    print(f"  per-dimension std across images: {emb.std(0).mean():.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", choices=["aro", "svo"], default="aro")
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("-n", type=int, default=512, help="rows to sample (text tower is slow on CPU)")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    default_parquet = constants.datasets_path / (
        "aro_test.parquet" if args.task == "aro" else "svo_test_probes.parquet"
    )
    loader = make_loader(
        args.parquet or default_parquet,
        args.task,
        model.text_model.non_linear_contractions,
        args.n,
        args.batch_size,
    )

    if args.task == "aro":
        ablate_aro(model, loader)
        image_spread(model, loader, "local_image_path")
    else:
        ablate_svo(model, loader)
        image_spread(model, loader, "true_local_image_path")


if __name__ == "__main__":
    main()
