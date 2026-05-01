import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SymmetricInfoNCE(nn.Module):
    """
    Symmetric InfoNCE loss (CLIP-style).

    Computes contrastive loss in both directions — image→text and text→image —
    and returns their mean. This doubles the gradient signal per batch compared
    to single-direction InfoNCE and is the standard for single-caption datasets
    where no explicit negatives are available.

    Expects:
        image_embeddings:  [B, D]
        text_embeddings:   [B, D]

    Both are L2-normalised internally; do not pre-normalise.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(self, image_emb: Tensor, text_emb: Tensor) -> tuple[Tensor, Tensor]:
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        batch_size = image_emb.shape[0]
        labels = torch.arange(batch_size, device=image_emb.device)

        logits = torch.matmul(image_emb, text_emb.t()) / self.temperature

        i2t_loss = F.cross_entropy(logits, labels)
        t2i_loss = F.cross_entropy(logits.t(), labels)
        loss = (i2t_loss + t2i_loss) / 2

        with torch.no_grad():
            i2t_acc = (logits.argmax(dim=1) == labels).float().mean()
            t2i_acc = (logits.t().argmax(dim=1) == labels).float().mean()
            accuracy = (i2t_acc + t2i_acc) / 2

        return loss, accuracy
