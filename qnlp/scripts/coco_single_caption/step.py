import torch.nn as nn
from torch import Tensor

from qnlp.core.training.losses.single_caption import SingleCaptionLoss


class COCOSingleCaptionStep:
    """
    TrainingStep for single-caption contrastive training on COCO-style batches.

    Uses in-batch negatives via SymmetricInfoNCE — with batch_size=512 each
    sample has 511 natural negatives from the other (image, caption) pairs in
    the batch.  No explicit negatives or projector heads required.

    Expects batch keys:
        local_image_path:  Tensor [B, C, H, W]
        caption:           list of (diagram_str, [Symbol, ...]) — length B
    """

    def __init__(
        self,
        loss_fn: SingleCaptionLoss,
        device,
        warmup_epochs: int = 0,
        warmup_alignment_weight: float = 0.0,
    ):
        self.loss_fn = loss_fn
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.warmup_alignment_weight = warmup_alignment_weight

    def on_epoch_start(self, epoch: int) -> None:
        """Switch alignment weight off after the warmup phase."""
        self.loss_fn.alignment_weight = self.warmup_alignment_weight if epoch <= self.warmup_epochs else 0.0

    def __call__(
        self,
        model: nn.Module,
        batch: dict,
        train: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        images = batch["local_image_path"].to(self.device)
        captions = batch["caption"]

        outputs = model(images, captions)

        return self.loss_fn(
            {
                "image_embeddings": outputs["image_embeddings"],
                "caption_embeddings": outputs["true_caption_embeddings"],
            }
        )
