import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from qnlp.core.training.batch_utils import drop_nonfinite_rows
from qnlp.core.training.losses.image_contrastive import ImageContrastiveLoss
from qnlp.core.training.losses.structured_contrastive import StructuredContrastiveLoss


def _finite_mask(repr_: Tensor | dict) -> Tensor:
    """isfinite check over a caption/image representation that may be a
    plain tensor (cosine/trilinear) or a role-tensor dict (role-grounded,
    Route A) — see EinsumModel.forward_roles."""
    if isinstance(repr_, dict):
        parts = [
            repr_["subj"],
            repr_["obj"],
            repr_["verb_left"].flatten(1),
            repr_["verb_mid"].flatten(1),
            repr_["verb_right"].flatten(1),
        ]
        return torch.cat(parts, dim=-1).isfinite().all(dim=-1)
    return repr_.isfinite().flatten(1).all(dim=-1)


def _index_rows(repr_: Tensor | dict, mask: Tensor) -> Tensor | dict:
    if isinstance(repr_, dict):
        return {k: v[mask] for k, v in repr_.items()}
    return repr_[mask]


class SVOHardNegStep:
    """
    TrainingStep for contrastive VLM training on SVO-Probes-style batches —
    mirrors AROContrastiveStep (qnlp/scripts/aro_contrastive/step.py) with
    image/caption roles swapped: one caption, two candidate images.

    Expects batch keys:
        true_local_image_path:  Tensor [B, C, H, W]
        false_local_image_path: Tensor [B, C, H, W]
        caption:                 list of (diagram_str, [Symbol, ...]) — length B
        subj/verb/obj:           list of str — length B, only read when
                                  model.score_head.needs_roles (Route A)

    Two forward passes are needed (one per image) since ContrastiveVLM's
    forward signature only embeds one image against up to two captions.
    """

    def __init__(self, loss_fn: ImageContrastiveLoss | StructuredContrastiveLoss, device: torch.device):
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

        score_head = getattr(model, "score_head", None)
        roles = None
        if score_head is not None and score_head.needs_roles:
            roles = list(
                zip(
                    [w.lower() for w in batch["subj"]],
                    [w.lower() for w in batch["verb"]],
                    [w.lower() for w in batch["obj"]],
                )
            )

        true_out = model(true_images, captions, roles=roles) if roles is not None else model(true_images, captions)
        false_out = model(false_images, captions, roles=roles) if roles is not None else model(false_images, captions)

        outputs = {
            "caption_embeddings": true_out["true_caption_embeddings"],
            "true_image_embeddings": true_out["image_embeddings"],
            "false_image_embeddings": false_out["image_embeddings"],
        }

        if score_head is not None:
            mask = (
                _finite_mask(outputs["caption_embeddings"])
                & _finite_mask(outputs["true_image_embeddings"])
                & _finite_mask(outputs["false_image_embeddings"])
            )
            n_dropped = int((~mask).sum().item())
            if n_dropped:
                outputs = {k: _index_rows(v, mask) for k, v in outputs.items()}
        else:
            outputs, n_dropped = drop_nonfinite_rows(outputs, list(outputs))

        n_remaining = (
            outputs["caption_embeddings"]["subj"].shape[0]
            if isinstance(outputs["caption_embeddings"], dict)
            else outputs["caption_embeddings"].shape[0]
        )
        if n_remaining == 0:
            return torch.zeros((), device=self.device, requires_grad=True), {
                "n_skipped": true_images.new_tensor(float(n_dropped))
            }

        loss, metrics = self.loss_fn(outputs)
        if n_dropped:
            metrics["n_skipped"] = true_images.new_tensor(float(n_dropped))

        if score_head is not None:
            # Structured heads (Routes A/B): score_pairs/hard_neg_acc etc. are
            # already logged by StructuredContrastiveLoss above — the cosine-
            # specific diagnostics below don't apply to non-flat representations.
            return loss, metrics

        with torch.no_grad():
            pos_sim = F.cosine_similarity(outputs["caption_embeddings"], outputs["true_image_embeddings"])
            neg_sim = F.cosine_similarity(outputs["caption_embeddings"], outputs["false_image_embeddings"])
            metrics["true_cosine_mean"] = pos_sim.mean()
            metrics["false_cosine_mean"] = neg_sim.mean()
            metrics["hard_neg_acc"] = (pos_sim > neg_sim).float().mean()

            # Discrimination-within-batch check: near-zero std means the model
            # assigns almost the same true/false similarity to every example in
            # this batch — i.e. it isn't discriminating AT ALL for this batch,
            # a different (and more diagnostic) failure than "discriminating
            # the wrong way". Complements the mean-only metrics above, which
            # can't distinguish "no discrimination" from "systematic wrong-way
            # discrimination".
            if pos_sim.numel() > 1:
                metrics["true_cosine_std"] = pos_sim.std()
                metrics["false_cosine_std"] = neg_sim.std()

            # Anisotropic-collapse check: mean off-diagonal pairwise cosine
            # similarity among this batch's (already L2-normalised) true-image
            # and caption embeddings. If this climbs toward 1.0 on val while
            # staying lower on train, embeddings are collapsing into a narrow
            # cone that can still satisfy the one specific triplet trained on
            # per example, without preserving the general discriminative
            # structure needed to generalise to held-out pairs.
            B = outputs["true_image_embeddings"].shape[0]
            if B > 1:
                img_n = outputs["true_image_embeddings"]
                cap_n = outputs["caption_embeddings"]
                off_diag = ~torch.eye(B, dtype=torch.bool, device=img_n.device)
                metrics["image_pairwise_cos_mean"] = (img_n @ img_n.t())[off_diag].mean()
                metrics["caption_pairwise_cos_mean"] = (cap_n @ cap_n.t())[off_diag].mean()

            gate = getattr(model.text_model, "nonlinear_gate", None)
            if gate is not None:
                metrics["nonlinear_gate"] = gate.detach()

        return loss, metrics
