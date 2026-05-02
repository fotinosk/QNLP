import torch
import torch.nn.functional as F
from torch import Tensor


class VICRegLoss:
    """
    VICReg: Variance-Invariance-Covariance Regularization.
    Bardes et al. 2022 (https://arxiv.org/abs/2105.04906)

    Three terms:
        invariance:  per-sample MSE between image and text embeddings.
                     Batch-size invariant — the core alignment signal.
        variance:    penalises dimensions whose std across the batch drops
                     below a threshold. Prevents dimensional collapse.
        covariance:  penalises off-diagonal entries of the covariance matrix.
                     Decorrelates embedding dimensions.

    Expects model outputs dict with:
        image_embeddings:   [B, D]
        caption_embeddings: [B, D]

    Embeddings should NOT be L2-normalised before this loss —
    VICReg operates on raw projected embeddings.
    """

    def __init__(
        self,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        variance_margin: float = 1.0,
        eps: float = 1e-4,
    ):
        self.inv_w = invariance_weight
        self.var_w = variance_weight
        self.cov_w = covariance_weight
        self.margin = variance_margin
        self.eps = eps

    def __call__(self, outputs: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        z_img = outputs["image_embeddings"]
        z_txt = outputs["caption_embeddings"]

        inv = self._invariance(z_img, z_txt)
        var = (self._variance(z_img) + self._variance(z_txt)) / 2
        cov = (self._covariance(z_img) + self._covariance(z_txt)) / 2

        loss = self.inv_w * inv + self.var_w * var + self.cov_w * cov

        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                F.normalize(z_img, dim=-1),
                F.normalize(z_txt, dim=-1),
            ).mean()

        metrics: dict[str, Tensor] = {
            "loss": loss,
            "invariance": inv,
            "variance": var,
            "covariance": cov,
            "cosine_similarity": cos_sim,
        }
        return loss, metrics

    def _invariance(self, z1: Tensor, z2: Tensor) -> Tensor:
        return F.mse_loss(z1, z2)

    def _variance(self, z: Tensor) -> Tensor:
        std = z.std(dim=0)
        return F.relu(self.margin - std).mean()

    def _covariance(self, z: Tensor) -> Tensor:
        B, D = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (B - 1)
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / D

    def to(self, device) -> "VICRegLoss":
        return self
