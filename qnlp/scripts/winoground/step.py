import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.core.training.losses.winoground_pair import WinogroundPairLoss


class WinogroundStep:
    """
    TrainingStep for contrastive VLM training on Winoground batches.

    Expects batch keys produced by winoground_train_collate_fn:
        images:         Tensor [B, C, H, W]
        true_captions:  list of (diagram_str, [Symbol, ...]) — length B
        false_captions: list of (diagram_str, [Symbol, ...]) — length B
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
        images = batch["images"].to(self.device)
        true_captions = batch["true_captions"]
        false_captions = batch["false_captions"]

        outputs = model(images, true_captions, false_captions)
        loss, metrics = self.loss_fn(outputs)

        with torch.no_grad():
            pos_sim = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
            neg_sim = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
            metrics["true_cosine_mean"] = pos_sim.mean()
            metrics["false_cosine_mean"] = neg_sim.mean()
            metrics["hard_neg_acc"] = (pos_sim > neg_sim).float().mean()

        return loss, metrics


class WinogroundPairStep:
    """
    TrainingStep exploiting the full 2x2 Winoground structure.

    Each item contributes 4 triplets — the 2 cross-modal wrong pairs
    (img_0, cap_1) and (img_1, cap_0) used bidirectionally as negatives:

        triplet(img_0, pos=cap_0, neg=cap_1)
        triplet(img_1, pos=cap_1, neg=cap_0)
        triplet(cap_0, pos=img_0, neg=img_1)
        triplet(cap_1, pos=img_1, neg=img_0)

    Expects batch keys produced by winoground_eval_collate_fn:
        images_0:   Tensor [B, C, H, W]
        images_1:   Tensor [B, C, H, W]
        captions_0: list of (diagram_str, [Symbol, ...]) — length B
        captions_1: list of (diagram_str, [Symbol, ...]) — length B
    """

    def __init__(self, loss_fn: WinogroundPairLoss, device: torch.device):
        self.loss_fn = loss_fn
        self.device = device

    def __call__(
        self,
        model: nn.Module,
        batch: dict,
        train: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        images_0 = batch["images_0"].to(self.device)
        images_1 = batch["images_1"].to(self.device)
        captions_0 = batch["captions_0"]
        captions_1 = batch["captions_1"]

        out0 = model(images_0, captions_0, captions_1)
        out1 = model(images_1, captions_1, captions_0)

        img0_emb = out0["image_embeddings"]
        cap0_emb = out0["true_caption_embeddings"]
        cap1_emb = out0["false_caption_embeddings"]
        img1_emb = out1["image_embeddings"]

        return self.loss_fn(img0_emb, img1_emb, cap0_emb, cap1_emb)
