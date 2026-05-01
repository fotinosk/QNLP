import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from qnlp.core.training.losses.contrastive import ContrastiveLoss


class AROContrastiveStep:
    """
    TrainingStep for contrastive VLM training on ARO-style batches.

    Expects batch keys:
        local_image_path:      Tensor [B, C, H, W]
        true_caption:          list of (diagram_str, [Symbol, ...]) — length B
        false_caption:         list of (diagram_str, [Symbol, ...]) — length B

    Returns (loss, metrics) where all metric tensors are on-device.
    Augments ContrastiveLoss metrics with cosine-similarity diagnostics.
    """

    def __init__(self, loss_fn: ContrastiveLoss, device: torch.device):
        self.loss_fn = loss_fn
        self.device = device

    def __call__(
        self,
        model: nn.Module,
        batch: dict,
        train: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        images = batch["local_image_path"].to(self.device)
        true_captions = batch["true_caption"]
        false_captions = batch["false_caption"]

        outputs = model(images, true_captions, false_captions)
        loss, metrics = self.loss_fn(outputs)

        with torch.no_grad():
            pos_sim = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
            neg_sim = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
            metrics["true_cosine_mean"] = pos_sim.mean()
            metrics["false_cosine_mean"] = neg_sim.mean()
            metrics["hard_neg_acc"] = (pos_sim > neg_sim).float().mean()

        return loss, metrics
