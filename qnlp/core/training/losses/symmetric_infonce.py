import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SymmetricInfoNCE(nn.Module):
    """
    Symmetric InfoNCE loss (CLIP-style) with learnable temperature.

    The logit scale (1/temperature) is a learnable scalar parameter following
    CLIP convention. It is clamped to prevent instability (equivalent to
    keeping temperature in [0.01, 1.0]).

    Expects:
        image_embeddings:  [B, D]
        text_embeddings:   [B, D]

    Both are L2-normalised internally.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        # logit_scale = log(1/temperature); exp(logit_scale) multiplies the logits
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(self, image_emb: Tensor, text_emb: Tensor) -> tuple[Tensor, Tensor]:
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        # Clamp scale to [1/0.3, 100] → temperature stays in [0.01, 0.3]
        logit_scale = self.logit_scale.exp().clamp(min=1 / 0.3, max=100.0)

        batch_size = image_emb.shape[0]
        labels = torch.arange(batch_size, device=image_emb.device)

        logits = logit_scale * torch.matmul(image_emb, text_emb.t())

        i2t_loss = F.cross_entropy(logits, labels)
        t2i_loss = F.cross_entropy(logits.t(), labels)
        loss = (i2t_loss + t2i_loss) / 2

        with torch.no_grad():
            i2t_acc = (logits.argmax(dim=1) == labels).float().mean()
            t2i_acc = (logits.t().argmax(dim=1) == labels).float().mean()
            accuracy = (i2t_acc + t2i_acc) / 2

        return loss, accuracy
