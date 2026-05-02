import torch
import torch.nn as nn
from torch import Tensor

from qnlp.domain.models.other.loss import InfoNCE


class ContrastiveLoss:
    """
    LossFunction implementation for contrastive VLM training.

    Expects model outputs dict with:
        image_embeddings:         [B, D]
        true_caption_embeddings:  [B, D]
        false_caption_embeddings: [B, D]  (optional — enables hard-negative triplet loss)

    Returns on-device metric tensors (no .item() calls).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        triplet_weight: float = 0.5,
        triplet_margin: float = 0.2,
        distance: str = "cosine",
    ):
        self._infonce = InfoNCE(temperature=temperature)

        if distance == "cosine":
            dist_fn = lambda x, y: 1 - nn.CosineSimilarity(dim=-1)(x, y)
        elif distance == "euclidean":
            dist_fn = nn.PairwiseDistance(p=2)
        else:
            raise ValueError(f"Unknown distance: {distance!r}. Use 'cosine' or 'euclidean'.")

        self._triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=dist_fn,
            margin=triplet_margin,
            swap=True,
        )
        self.triplet_weight = triplet_weight

    def __call__(self, outputs: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        image_emb = outputs["image_embeddings"]
        true_emb = outputs["true_caption_embeddings"]
        false_emb = outputs.get("false_caption_embeddings")

        infonce_loss, accuracy = self._infonce(true_emb, image_emb)

        metrics: dict[str, Tensor] = {
            "loss": infonce_loss,
            "infonce_loss": infonce_loss,
            "accuracy": accuracy,
        }

        if false_emb is not None:
            triplet_loss = self._triplet(image_emb, true_emb, false_emb)
            total = infonce_loss + self.triplet_weight * triplet_loss
            metrics["triplet_loss"] = triplet_loss
            metrics["loss"] = total
            return total, metrics

        return infonce_loss, metrics

    def to(self, device: torch.device) -> "ContrastiveLoss":
        self._infonce = self._infonce.to(device)
        self._triplet = self._triplet.to(device)
        return self
