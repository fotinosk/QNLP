"""
Frozen CLIP image tower — a sanity check, not a candidate architecture.

Motivation: SVO training has landed at chance with every image tower tried
so far (TTN, in every configuration; DTTN, from scratch, despite hitting
0.8589 on CIFAR-10) — see TTN_CIFAR_EXPERIMENTS.md. That's strong evidence
the bottleneck is elsewhere (most likely the text tower), but every one of
those image towers was still learning visual features on SVO's own small
(~8,600 row) dataset, which leaves a residual doubt: maybe SVO's images
just don't carry enough usable signal for ANY from-scratch tower this
small a dataset can train. A frozen, externally-pretrained CLIP encoder
settles that: if the caption/image task is learnable at all with a
known-excellent image representation held fixed, SVO should clear chance;
if it still can't, that's decisive evidence the bottleneck is the text
side (or the loss/data), not "the image tower needs more/better data."

Not a real image tower for this project — CLIP is not quantum-inspired,
not trained here, and this backbone is not meant to be adopted. It exists
solely to answer the question above.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

# CLIP's own normalisation (distinct from the ImageNet normalisation every
# other dataset transform in this project applies before the image ever
# reaches the model).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class CLIPImageModel(nn.Module):
    """Drop-in image tower interface (TTNImageModel/DTTNImageModel-
    compatible forward signature), backed by a frozen pretrained CLIP
    vision encoder. Every existing caller's ImageNet-normalised input is
    accepted unchanged: internally inverted back to ~[0,1], resized to
    CLIP's native resolution, and re-normalised with CLIP's own stats --
    no transform pipeline needs to change to use this.

    Always frozen: CLIP's parameters have requires_grad=False and the
    encoder runs in eval() + no_grad() regardless of the outer training
    loop's mode, since the entire point is to hold the image side fixed."""

    def __init__(self, embedding_dim: int, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.clip.requires_grad_(False)
        self.clip.eval()
        clip_dim = self.clip.config.projection_dim
        self.embedding_dim = embedding_dim
        self.proj = nn.Identity() if embedding_dim == clip_dim else nn.Linear(clip_dim, embedding_dim)
        self.register_buffer("_imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("_clip_mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_clip_std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1))
        self._clip_resolution = self.clip.config.vision_config.image_size

    def _to_clip_pixels(self, x: torch.Tensor) -> torch.Tensor:
        x01 = (x * self._imagenet_std + self._imagenet_mean).clamp(0.0, 1.0)
        x_resized = F.interpolate(
            x01, size=(self._clip_resolution, self._clip_resolution), mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        return (x_resized - self._clip_mean) / self._clip_std

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        pixel_values = self._to_clip_pixels(x)
        with torch.no_grad():
            feats = self.clip.get_image_features(pixel_values=pixel_values)
        out = self.proj(feats)
        return F.normalize(out, p=2, dim=-1) if normalize else out

    def forward_regions(self, x: torch.Tensor, level: int = 1) -> torch.Tensor:
        raise NotImplementedError(
            "CLIPImageModel is a cosine-only sanity check (see this module's docstring) "
            "-- it does not implement forward_regions for Route A/B score heads."
        )
