"""SugarCrepe++ hard-negative evaluation for a trained ContrastiveVLM checkpoint.

For each (image, true_caption, false_caption) the model is "correct" if
cosine(image, true) > cosine(image, false).

Usage:
    python -m qnlp.scripts.sugarcrepepp.evaluate --checkpoint /path/to/best_model.pt
    python -m qnlp.scripts.sugarcrepepp.evaluate --checkpoint /path/to/best_model.pt --batch_size 64
"""

import argparse
from pathlib import Path

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
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="sugarcrepepp_evaluate")

COMPILED_COLUMNS = [
    ("true_diagram", "true_symbols", "true_caption", "true_path"),
    ("false_diagram", "false_symbols", "false_caption", "false_path"),
]


def _infer_non_linear(state_dict: dict) -> bool:
    return any("nonlinear_gate" in k for k in state_dict)


def _infer_embedding_dim(state_dict: dict) -> int:
    return state_dict["image_head.proj.weight"].shape[0]


def evaluate(checkpoint_path: Path, batch_size: int = 128) -> dict:
    device = get_device()
    parquet = constants.datasets_path / "sugarcrepepp_eval.parquet"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    non_linear = _infer_non_linear(state_dict)
    embedding_dim = _infer_embedding_dim(state_dict)
    logger.info(
        f"Checkpoint epoch {checkpoint.get('epoch', '?')} | embedding_dim={embedding_dim} | "
        f"non_linear_contractions={non_linear}"
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

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vlm_collate_fn,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

    text_model = EinsumModel(non_linear_contractions=non_linear)
    image_model = TTNImageModel(embedding_dim)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=embedding_dim)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    known_symbols = set(model.text_model.sym2weight.keys())

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
                if all(s in known_symbols for s in true_caps[i][1])
                and all(s in known_symbols for s in false_caps[i][1])
            ]
            n_skipped += len(true_caps) - len(valid)
            if not valid:
                continue

            images = torch.stack([batch["local_image_path"][i] for i in valid]).to(device)
            outputs = model(images, [true_caps[i] for i in valid], [false_caps[i] for i in valid])

            pos = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
            neg = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
            finite = (torch.isfinite(pos) & torch.isfinite(neg)).tolist()
            c = (pos > neg).tolist()
            for j in range(len(valid)):
                if not finite[j]:
                    n_skipped += 1
                    continue
                correct.append(bool(c[j]))
                pos_cos.append(float(pos[j]))
                neg_cos.append(float(neg[j]))

    n = len(correct)
    result = {
        "evaluated": n,
        "skipped": n_skipped,
        "hard_neg_acc": (sum(correct) / n) if n else float("nan"),
        "true_cos": (sum(pos_cos) / n) if n else float("nan"),
        "false_cos": (sum(neg_cos) / n) if n else float("nan"),
    }

    logger.info("=== SugarCrepe++ hard-negative accuracy ===")
    logger.info(f"  evaluated={result['evaluated']}  skipped={result['skipped']}")
    logger.info(
        f"  acc={result['hard_neg_acc']:.4f}  true_cos={result['true_cos']:.4f}  false_cos={result['false_cos']:.4f}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a ContrastiveVLM checkpoint on SugarCrepe++.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    evaluate(Path(args.checkpoint), batch_size=args.batch_size)
