import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WinogroundPairLoss:
    """
    Bidirectional triplet loss over the full 2x2 Winoground structure.

    Given embeddings (img_0, img_1, cap_0, cap_1) where the correct pairings
    are (img_0 <-> cap_0) and (img_1 <-> cap_1), four triplets are formed:

        triplet(img_0,  pos=cap_0, neg=cap_1)   image-anchored
        triplet(img_1,  pos=cap_1, neg=cap_0)   image-anchored
        triplet(cap_0,  pos=img_0, neg=img_1)   caption-anchored
        triplet(cap_1,  pos=img_1, neg=img_0)   caption-anchored

    The 4 wrong pairs used as negatives:
        (img_0, cap_1), (img_1, cap_0)          cross-modal
        (cap_0, img_1), (cap_1, img_0)          cross-modal reversed

    Metrics track per-triplet losses and the Winoground text/image/group
    accuracies directly, so training metrics mirror evaluation scores.
    """

    def __init__(self, margin: float = 0.2, distance: str = "cosine"):
        if distance == "cosine":
            dist_fn = lambda x, y: 1 - nn.CosineSimilarity(dim=-1)(x, y)
        elif distance == "euclidean":
            dist_fn = nn.PairwiseDistance(p=2)
        else:
            raise ValueError(f"Unknown distance: {distance!r}")

        self._triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=dist_fn,
            margin=margin,
            swap=True,
        )

    def __call__(
        self,
        img0_emb: Tensor,
        img1_emb: Tensor,
        cap0_emb: Tensor,
        cap1_emb: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        t_img0 = self._triplet(img0_emb, cap0_emb, cap1_emb)
        t_img1 = self._triplet(img1_emb, cap1_emb, cap0_emb)
        t_cap0 = self._triplet(cap0_emb, img0_emb, img1_emb)
        t_cap1 = self._triplet(cap1_emb, img1_emb, img0_emb)

        total = t_img0 + t_img1 + t_cap0 + t_cap1

        with torch.no_grad():
            s00 = F.cosine_similarity(img0_emb, cap0_emb, dim=-1)
            s01 = F.cosine_similarity(img0_emb, cap1_emb, dim=-1)
            s10 = F.cosine_similarity(img1_emb, cap0_emb, dim=-1)
            s11 = F.cosine_similarity(img1_emb, cap1_emb, dim=-1)

            text_score = (s00 > s01) & (s11 > s10)
            image_score = (s00 > s10) & (s11 > s01)

        metrics = {
            "loss": total,
            "t_img0": t_img0,
            "t_img1": t_img1,
            "t_cap0": t_cap0,
            "t_cap1": t_cap1,
            "text_acc": text_score.float().mean(),
            "image_acc": image_score.float().mean(),
            "group_acc": (text_score & image_score).float().mean(),
        }
        return total, metrics

    def to(self, device: torch.device) -> "WinogroundPairLoss":
        self._triplet = self._triplet.to(device)
        return self
