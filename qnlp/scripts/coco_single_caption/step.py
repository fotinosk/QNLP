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

    def __init__(self, loss_fn: SingleCaptionLoss, device):
        self.loss_fn = loss_fn
        self.device = device

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
