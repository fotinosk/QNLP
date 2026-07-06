import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel


class AlignmentHead(nn.Module):
    """
    Single linear layer mapping backbone outputs into a shared L2-normalised
    space. Initialised as identity so training starts from the same geometry
    as having no head. Used at both train and eval time — checkpointed as
    part of ContrastiveVLM.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class MLPProjectionHead(nn.Module):
    """
    Two-layer MLP projection head: Linear → GELU → Linear → L2Norm.

    Gives the model capacity to learn a non-linear bridge between backbone
    output spaces and the shared contrastive embedding. The hidden dimension
    is 2× the embedding dim, matching the CLIP projection head design.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ContrastiveVLM(nn.Module):
    """
    Model wrapper for contrastive VLM training.

    Holds text and image backbones plus per-modality alignment heads.
    The heads are lightweight Linear → L2Norm layers that learn a shared
    embedding space without BatchNorm or throw-away projectors.
    All four modules are checkpointed together.
    """

    def __init__(
        self,
        text_model: EinsumModel,
        image_model: TTNImageModel,
        embedding_dim: int,
        use_mlp_head: bool = False,
    ):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.embedding_dim = embedding_dim
        head_cls = MLPProjectionHead if use_mlp_head else AlignmentHead
        self.image_head = head_cls(embedding_dim)
        self.text_head = head_cls(embedding_dim)

    def forward(self, images, true_captions, false_captions=None) -> dict:
        image_emb = self.image_head(self.image_model(images))
        true_emb = self._safe_text_embed(true_captions)
        outputs = {
            "image_embeddings": image_emb,
            "true_caption_embeddings": true_emb,
        }
        if false_captions is not None:
            outputs["false_caption_embeddings"] = self._safe_text_embed(false_captions)
        return outputs

    def _safe_text_embed(self, captions) -> "torch.Tensor":
        """Apply text_model → text_head, preventing NaN gradient contamination.

        Infeasible NLC contractions return a NaN sentinel from EinsumModel. If
        those NaN rows are passed directly through text_head (an nn.Linear),
        backprop computes grad_weight = grad_output.T @ input, and
        0 * nan = nan in IEEE arithmetic — poisoning text_head.weight even
        though those rows are masked out by drop_nonfinite_rows.

        Fix: replace NaN rows with 0 before text_head (so 0 * 0 = 0 in backprop),
        apply text_head, then restore NaN so drop_nonfinite_rows still filters them.
        """
        import torch

        text_out = self.text_model(captions)
        finite = torch.isfinite(text_out).all(dim=-1, keepdim=True)
        text_out_safe = torch.where(finite, text_out, torch.zeros_like(text_out))
        emb = self.text_head(text_out_safe)
        return torch.where(finite, emb, torch.full_like(emb, float("nan")))

    def load_state_dict(self, state_dict, strict: bool = True):
        # EinsumModel injects "symbols_list"/"sizes_list" as bare (unprefixed) keys.
        # PyTorch's recursive load calls _load_from_state_dict, not load_state_dict,
        # so EinsumModel.load_state_dict (which rebuilds its ParameterList) never
        # fires automatically. We manually split and delegate to each submodule.
        text_sd, image_sd, img_head_sd, txt_head_sd = {}, {}, {}, {}
        for k, v in state_dict.items():
            if k in ("symbols_list", "sizes_list"):
                text_sd[k] = v
            elif k.startswith("text_model."):
                text_sd[k[len("text_model.") :]] = v
            elif k.startswith("image_model."):
                image_sd[k[len("image_model.") :]] = v
            elif k.startswith("image_head."):
                img_head_sd[k[len("image_head.") :]] = v
            elif k.startswith("text_head."):
                txt_head_sd[k[len("text_head.") :]] = v
        self.text_model.load_state_dict(text_sd, strict=strict)
        self.image_model.load_state_dict(image_sd, strict=strict)
        self.image_head.load_state_dict(img_head_sd, strict=strict)
        self.text_head.load_state_dict(txt_head_sd, strict=strict)

    def clip_gradients(self, max_norm: float) -> None:
        clip_grad_norm_(self.text_model.parameters(), max_norm)
        clip_grad_norm_(self.image_model.parameters(), max_norm)
        clip_grad_norm_(self.image_head.parameters(), max_norm)
        clip_grad_norm_(self.text_head.parameters(), max_norm)
