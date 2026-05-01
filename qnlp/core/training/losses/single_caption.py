import torch
from torch import Tensor

from qnlp.core.training.losses.symmetric_infonce import SymmetricInfoNCE


class SingleCaptionLoss:
    """
    LossFunction for single-caption contrastive training (COCO-style).

    Expects model outputs dict with:
        image_embeddings:   [B, D]
        caption_embeddings: [B, D]

    Uses symmetric InfoNCE — both image→text and text→image directions —
    to maximise gradient signal from in-batch negatives alone.
    """

    def __init__(self, temperature: float = 0.07):
        self._loss = SymmetricInfoNCE(temperature=temperature)

    def __call__(self, outputs: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        image_emb = outputs["image_embeddings"]
        caption_emb = outputs["caption_embeddings"]

        loss, accuracy = self._loss(image_emb, caption_emb)

        metrics: dict[str, Tensor] = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return loss, metrics

    def to(self, device: torch.device) -> "SingleCaptionLoss":
        self._loss = self._loss.to(device)
        return self
