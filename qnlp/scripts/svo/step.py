import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from qnlp.core.training.batch_utils import drop_nonfinite_rows
from qnlp.core.training.losses.image_contrastive import ImageContrastiveLoss


class SVOHardNegStep:
    """
    TrainingStep for contrastive VLM training on SVO-Probes-style batches —
    mirrors AROContrastiveStep (qnlp/scripts/aro_contrastive/step.py) with
    image/caption roles swapped: one caption, two candidate images.

    Expects batch keys:
        true_local_image_path:  Tensor [B, C, H, W]
        false_local_image_path: Tensor [B, C, H, W]
        caption:                 list of (diagram_str, [Symbol, ...]) — length B

    Two forward passes are needed (one per image) since ContrastiveVLM's
    forward signature only embeds one image against up to two captions.
    """

    def __init__(self, loss_fn: ImageContrastiveLoss, device: torch.device):
        self.loss_fn = loss_fn
        self.device = device

    def __call__(
        self,
        model: nn.Module,
        batch: dict,
        train: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        true_images = batch["true_local_image_path"].to(self.device)
        false_images = batch["false_local_image_path"].to(self.device)
        captions = batch["caption"]

        true_out = model(true_images, captions)
        false_out = model(false_images, captions)

        outputs = {
            "caption_embeddings": true_out["true_caption_embeddings"],
            "true_image_embeddings": true_out["image_embeddings"],
            "false_image_embeddings": false_out["image_embeddings"],
        }

        outputs, n_dropped = drop_nonfinite_rows(outputs, list(outputs))

        if outputs["caption_embeddings"].shape[0] == 0:
            return torch.zeros((), device=self.device, requires_grad=True), {
                "n_skipped": true_images.new_tensor(float(n_dropped))
            }

        loss, metrics = self.loss_fn(outputs)
        if n_dropped:
            metrics["n_skipped"] = true_images.new_tensor(float(n_dropped))

        with torch.no_grad():
            pos_sim = F.cosine_similarity(outputs["caption_embeddings"], outputs["true_image_embeddings"])
            neg_sim = F.cosine_similarity(outputs["caption_embeddings"], outputs["false_image_embeddings"])
            metrics["true_cosine_mean"] = pos_sim.mean()
            metrics["false_cosine_mean"] = neg_sim.mean()
            metrics["hard_neg_acc"] = (pos_sim > neg_sim).float().mean()

            gate = getattr(model.text_model, "nonlinear_gate", None)
            if gate is not None:
                metrics["nonlinear_gate"] = gate.detach()

        return loss, metrics
