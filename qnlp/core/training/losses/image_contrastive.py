import torch
import torch.nn as nn
from torch import Tensor

from qnlp.domain.models.other.loss import InfoNCE


class ImageContrastiveLoss:
    """
    Mirrors ContrastiveLoss (qnlp/core/training/losses/contrastive.py), the
    legacy ARO training loss, for tasks where the hard negative is on the
    IMAGE side instead of the caption side (SVO-Probes: one caption, two
    candidate images, rather than ARO's one image, two candidate captions).

    Expects model outputs dict with:
        caption_embeddings:     [B, D]
        true_image_embeddings:  [B, D]
        false_image_embeddings: [B, D]  (optional — enables hard-negative triplet loss)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        triplet_weight: float = 0.5,
        triplet_margin: float = 0.2,
        distance: str = "cosine",
        image_as_anchor: bool = False,
    ):
        """
        image_as_anchor: DISCOCLIP_REPRODUCTION_PLAN.md's line-by-line diff
            found discoclip's SVO training calls its (byte-identical)
            InfoNCE as `criterion(image_emb, text_emb)` -- image as the
            anchor/query row of the in-batch similarity matrix -- while
            this loss has always called it as `(caption_emb, true_emb)`,
            text as anchor. Since the cross-caption/image similarity
            matrix isn't symmetric, these are two different loss functions,
            not a cosmetic argument-order difference. Default False
            preserves every existing run; True reproduces discoclip's
            direction exactly.
        """
        self._infonce = InfoNCE(temperature=temperature)
        self.image_as_anchor = image_as_anchor

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
        caption_emb = outputs["caption_embeddings"]
        true_emb = outputs["true_image_embeddings"]
        false_emb = outputs.get("false_image_embeddings")

        if self.image_as_anchor:
            infonce_loss, accuracy = self._infonce(true_emb, caption_emb)
        else:
            infonce_loss, accuracy = self._infonce(caption_emb, true_emb)

        metrics: dict[str, Tensor] = {
            "loss": infonce_loss,
            "infonce_loss": infonce_loss,
            "accuracy": accuracy,
        }

        if false_emb is not None:
            triplet_loss = self._triplet(caption_emb, true_emb, false_emb)
            total = infonce_loss + self.triplet_weight * triplet_loss
            metrics["triplet_loss"] = triplet_loss
            metrics["loss"] = total
            return total, metrics

        return infonce_loss, metrics

    def to(self, device: torch.device) -> "ImageContrastiveLoss":
        self._infonce = self._infonce.to(device)
        self._triplet = self._triplet.to(device)
        return self
