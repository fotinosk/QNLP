"""Frozen CLIP-ViT image encoder for the einsum_frozen_clip experiment.

Wraps a pretrained `transformers.CLIPVisionModel` whose parameters are frozen
(requires_grad=False, always eval mode) and exposes a small trainable Linear
that projects the pooled CLS embedding to the experiment's shared
`embedding_dim`. Output is L2-normalised, matching the convention of
TTNImageModel so the downstream AlignmentHead behaves identically.

This is the *image-side ablation* for the EinsumModel question: by giving the
text side a clean, never-changing image signal, we test whether EinsumModel
itself can express a faithful alignment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel


class FrozenClipVisionModel(nn.Module):
    """Pretrained CLIP visual transformer with all backbone params frozen.

    Only the final Linear (pooled hidden → embedding_dim) is trainable.
    forward() returns L2-normalised [B, embedding_dim] embeddings.
    """

    def __init__(self, clip_model_name: str, embedding_dim: int):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(clip_model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, embedding_dim)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden

    def train(self, mode: bool = True):
        # Keep the frozen CLIP backbone in eval mode regardless of the
        # outer model's train/eval state. Only the trainable proj layer
        # is affected by `mode`.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Backbone is frozen — skip grad tracking to save memory/compute.
        with torch.no_grad():
            out = self.backbone(pixel_values=pixel_values)
        pooled = out.pooler_output  # [B, hidden]
        x = self.proj(pooled)
        return F.normalize(x, p=2, dim=-1)
