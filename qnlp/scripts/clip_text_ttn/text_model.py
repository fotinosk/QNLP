"""Frozen CLIP text encoder for the clip_text_ttn experiment.

Wraps `transformers.CLIPTextModelWithProjection` with all parameters frozen.
The model produces a 512-d "text_embeds" tensor that already lives in CLIP's
shared multimodal projection space. A small trainable Linear adapts it to
the experiment's `embedding_dim` (also 512 by default) so the downstream
AlignmentHead has something to bend.

This is the *text-side* ablation: by giving the trainable TTNImageModel a
clean, never-changing text signal, we test whether the TTN can learn an
alignment at full COCO scale. If yes, the historical v1 setup's
generalisation was bottlenecked by the EinsumModel side, not the TTN side.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModelWithProjection


class FrozenClipTextModel(nn.Module):
    """Pretrained CLIP text transformer + projection, all frozen.

    Only the final Linear adapter is trainable.
    forward() takes a dict of (input_ids, attention_mask) and returns
    L2-normalised [B, embedding_dim] embeddings.
    """

    def __init__(self, clip_model_name: str, embedding_dim: int):
        super().__init__()
        self.backbone = CLIPTextModelWithProjection.from_pretrained(clip_model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.text_embed_dim = self.backbone.config.projection_dim  # 512 for ViT-B/32
        self.embedding_dim = embedding_dim
        self.proj = nn.Linear(self.text_embed_dim, embedding_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # Stay frozen regardless of outer mode
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # kwarg signature so that ClipBaselineVLM-style `**true_text` splatting works.
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = out.text_embeds  # [B, projection_dim]
        x = self.proj(text_embeds)
        return F.normalize(x, p=2, dim=-1)
