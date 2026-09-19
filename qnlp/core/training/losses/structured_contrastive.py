import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from qnlp.domain.models.vlm.score_heads import ScoreHead


class StructuredContrastiveLoss:
    """
    Mirrors ImageContrastiveLoss (qnlp/core/training/losses/image_contrastive.py)
    but scores via a ScoreHead (Routes A/B, TTN_CIFAR_EXPERIMENTS.md) instead of
    raw cosine similarity — the score head's own score_matrix/score_pairs
    replace the flat-embedding dot products the cosine path uses.

    Expects model outputs dict with:
        caption_embeddings:     text_repr  (pooled vector, or the role-tensor
                                 dict — whatever the score head's needs_roles
                                 flag calls for)
        true_image_embeddings:  image_repr  ([B, n_regions, region_dim])
        false_image_embeddings: image_repr  (optional — hard-negative triplet)
    """

    def __init__(
        self,
        score_head: ScoreHead,
        temperature: float = 0.07,
        triplet_weight: float = 0.5,
        triplet_margin: float = 0.2,
    ):
        self.score_head = score_head
        self.temperature = temperature
        self.triplet_weight = triplet_weight
        self.triplet_margin = triplet_margin
        self._cross_entropy = nn.CrossEntropyLoss()

    def __call__(self, outputs: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        text_repr = outputs["caption_embeddings"]
        true_repr = outputs["true_image_embeddings"]
        false_repr = outputs.get("false_image_embeddings")

        score_matrix = self.score_head.score_matrix(text_repr, true_repr)  # [B, B]
        labels = torch.arange(score_matrix.shape[0], device=score_matrix.device)
        infonce_loss = self._cross_entropy(score_matrix / self.temperature, labels)
        with torch.no_grad():
            accuracy = (score_matrix.argmax(dim=1) == labels).float().mean()

        metrics: dict[str, Tensor] = {
            "loss": infonce_loss,
            "infonce_loss": infonce_loss,
            "accuracy": accuracy,
        }

        if false_repr is not None:
            pos_score = self.score_head.score_pairs(text_repr, true_repr)
            neg_score = self.score_head.score_pairs(text_repr, false_repr)
            # Margin ranking in score space (higher = better match) — the
            # score-space analogue of TripletMarginWithDistanceLoss's intent,
            # since these scores aren't distances and aren't scale-bounded
            # the way cosine similarity is.
            triplet_loss = F.relu(self.triplet_margin - (pos_score - neg_score)).mean()
            total = infonce_loss + self.triplet_weight * triplet_loss
            metrics["triplet_loss"] = triplet_loss
            metrics["loss"] = total
            with torch.no_grad():
                metrics["pos_score_mean"] = pos_score.mean()
                metrics["neg_score_mean"] = neg_score.mean()
                metrics["hard_neg_acc"] = (pos_score > neg_score).float().mean()
            return total, metrics

        return infonce_loss, metrics

    def to(self, device: torch.device) -> "StructuredContrastiveLoss":
        self.score_head = self.score_head.to(device)
        return self
