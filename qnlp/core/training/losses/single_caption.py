from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from qnlp.core.training.losses.symmetric_infonce import SymmetricInfoNCE


class SingleCaptionLoss:
    """
    Loss for single-caption contrastive training (COCO-style).

    Combines:
      - SymmetricInfoNCE with learnable temperature for in-batch discrimination
      - Per-sample cosine alignment loss on matched pairs for direct alignment signal

    All batch-size-agnostic metrics are tracked:
      cosine_similarity   mean similarity of matched pairs
      alignment_gap       mean_pos - mean_neg (discrimination signal)
      hard_neg_accuracy   fraction where matched pair beats its hardest negative
      sim_ratio           mean_pos / |mean_neg| (discrimination ratio)
      modality_gap        cosine similarity between mean image and mean text embedding
      temperature         current effective temperature (1 / exp(logit_scale))
    """

    def __init__(
        self,
        temperature: float = 0.07,
        alignment_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        self._loss = SymmetricInfoNCE(temperature=temperature)
        self.alignment_weight = alignment_weight
        self._eps = eps

    def __call__(self, outputs: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        image_emb = outputs["image_embeddings"]
        caption_emb = outputs["caption_embeddings"]

        img_n = F.normalize(image_emb, dim=-1)
        txt_n = F.normalize(caption_emb, dim=-1)

        # InfoNCE with learnable temperature
        infonce_loss, accuracy = self._loss(image_emb, caption_emb)

        # Per-sample cosine alignment on matched pairs — range [0, 2]
        # Provides a direct per-sample gradient independent of batch hardness
        alignment_loss = (1.0 - (img_n * txt_n).sum(dim=-1)).mean()

        loss = infonce_loss + self.alignment_weight * alignment_loss

        with torch.no_grad():
            S = torch.matmul(img_n, txt_n.t())
            B = S.shape[0]
            eye = torch.eye(B, dtype=torch.bool, device=S.device)

            pos_sim = S.diagonal()
            mean_pos = pos_sim.mean()
            mean_neg = S[~eye].mean()

            alignment_gap = mean_pos - mean_neg

            hardest_neg = S.masked_fill(eye, -torch.inf).max(dim=1).values
            hard_neg_accuracy = (pos_sim > hardest_neg).float().mean()

            sim_ratio = mean_pos / (mean_neg.abs() + self._eps)

            modality_gap = F.cosine_similarity(
                img_n.mean(dim=0, keepdim=True),
                txt_n.mean(dim=0, keepdim=True),
            ).squeeze()

            temperature = 1.0 / self._loss.logit_scale.exp().clamp(max=100.0)

        metrics: dict[str, Tensor] = {
            "loss": loss,
            "infonce_loss": infonce_loss,
            "alignment_loss": alignment_loss,
            "accuracy": accuracy,
            "cosine_similarity": mean_pos,
            "alignment_gap": alignment_gap,
            "hard_neg_accuracy": hard_neg_accuracy,
            "sim_ratio": sim_ratio,
            "modality_gap": modality_gap,
            "temperature": temperature,
        }
        return loss, metrics

    def parameters(self) -> Iterator[nn.Parameter]:
        return self._loss.parameters()

    def to(self, device) -> "SingleCaptionLoss":
        self._loss = self._loss.to(device)
        return self
