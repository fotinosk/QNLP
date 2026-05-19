"""Training step for clip_baseline. Same shape as AROContrastiveStep:
returns (loss, metrics) where every metric tensor lives on-device.

In-batch negatives via random derangement: each true_caption's "false" partner
is another caption from the same batch (pseudo-derangement so no self-pairs).
This is the standard CLIP InfoNCE contrastive setup; equivalent in expectation
to the pre-computed derangement used by aro_contrastive over coco_contrastive.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from qnlp.core.training.losses.contrastive import ContrastiveLoss


class ClipBaselineStep:
    def __init__(self, loss_fn: ContrastiveLoss, device: torch.device):
        self.loss_fn = loss_fn
        self.device = device

    def __call__(
        self,
        model: nn.Module,
        batch: dict,
        train: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        images = batch["image"].to(self.device, non_blocking=True)
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        B = images.size(0)

        # Build "false" caption batch via random derangement (no self-pair).
        # Doing this in the step (rather than in the dataloader) keeps the
        # negatives shuffled differently every epoch, matching aro_contrastive's
        # behaviour of resampling synthetic negatives at dataset-creation time.
        if B > 1:
            perm = torch.randperm(B, device=self.device)
            collisions = perm == torch.arange(B, device=self.device)
            if collisions.any():
                # Rotate self-pairs by 1 to break them
                perm[collisions] = (perm[collisions] + 1) % B
        else:
            perm = torch.zeros(B, dtype=torch.long, device=self.device)

        true_text = {"input_ids": input_ids, "attention_mask": attention_mask}
        false_text = {"input_ids": input_ids[perm], "attention_mask": attention_mask[perm]}

        outputs = model(images, true_text, false_text)
        loss, metrics = self.loss_fn(outputs)

        with torch.no_grad():
            pos_sim = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
            neg_sim = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
            metrics["true_cosine_mean"] = pos_sim.mean()
            metrics["false_cosine_mean"] = neg_sim.mean()
            metrics["hard_neg_acc"] = (pos_sim > neg_sim).float().mean()

        return loss, metrics
