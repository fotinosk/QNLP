"""
Frozen CLIP text tower — PAPER_EXPERIMENTS_PLAN.md's M3, which isolates the
image tower (TTN) against a strong text encoder, mirroring how M1 isolates
the text tower (DisCoCat/TTN) against a strong image encoder (frozen CLIP).

Mirrors qnlp/discoviz/models/clip_image_model.py's pattern exactly: frozen
pretrained CLIP, held fixed regardless of the outer training loop's mode,
with a linear projection only if embedding_dim differs from CLIP's own
projection_dim. Unlike EinsumModel, this has no fixed vocabulary — any
string tokenises, so no `sym2weight`/symbol-collection step is needed or
possible; callers that assume `model.text_model.sym2weight` (e.g.
evaluate_sugarcrepe's OOV filter) must special-case this class.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import CLIPModel, CLIPTokenizer


class CLIPTextModel(nn.Module):
    """Drop-in text tower interface (EinsumModel-compatible: takes a list of
    per-row inputs, returns a stacked [B, embedding_dim] tensor), backed by
    a frozen pretrained CLIP text encoder. Inputs are raw caption strings,
    not (diagram, symbols) tuples -- see VLMDataset's 2-tuple compiled_columns
    form for how the dataset layer produces plain strings for this model.

    Always frozen, same rationale as CLIPImageModel: the entire point of M3
    is to hold the text side fixed at a known-strong representation."""

    non_linear_contractions = False  # SVOHardNegStep/evaluate.py compatibility shim

    def __init__(self, embedding_dim: int, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.clip.requires_grad_(False)
        self.clip.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        clip_dim = self.clip.config.projection_dim
        self.embedding_dim = embedding_dim
        self.proj = nn.Identity() if embedding_dim == clip_dim else nn.Linear(clip_dim, embedding_dim)

    def forward(self, texts: list[str], normalize: bool = True) -> torch.Tensor:
        device = next(self.clip.parameters()).device
        tokens = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = self.clip.get_text_features(**tokens)
        out = self.proj(feats)
        return F.normalize(out, p=2, dim=-1) if normalize else out


class TextModelSettings(BaseSettings):
    """Mirrors ImageModelSettings (qnlp/discoviz/models/image_model.py): a
    module-level singleton read directly wherever the text tower is
    selected, independent of any experiment's own SVO_ML_/ML_-prefixed
    config, matching the IMAGE_MODEL_ prefix's existing convention."""

    text_backbone: Literal["discocat", "clip"] = "discocat"
    clip_model_name: str = "openai/clip-vit-base-patch32"

    model_config = SettingsConfigDict(env_prefix="TEXT_MODEL_")


text_model_hyperparams = TextModelSettings()


def build_text_model(
    embedding_dim: int,
    symbols: list | None = None,
    sizes: list | None = None,
    non_linear_contractions: bool = False,
    use_weight_norm: bool = True,
) -> nn.Module:
    """PAPER_EXPERIMENTS_PLAN.md's M3 selector. `symbols`/`sizes` are ignored
    (and may be empty) when text_backbone == "clip" -- CLIP has no fixed
    vocabulary, so the caller's usual collect_symbol_sizes() step is
    harmless dead work for M3, not an error."""
    if text_model_hyperparams.text_backbone == "clip":
        return CLIPTextModel(embedding_dim, model_name=text_model_hyperparams.clip_model_name)
    from qnlp.discoviz.models.einsum_model import EinsumModel

    return EinsumModel(
        symbols or [],
        sizes or [],
        non_linear_contractions=non_linear_contractions,
        use_weight_norm=use_weight_norm,
    )


def caption_compiled_columns(
    diagram_col: str, symbols_col: str, text_col: str, output_key: str, path_col: str = "path"
) -> tuple:
    """Returns the right VLMDataset compiled_columns entry for whichever
    text backbone is selected: the usual 4-tuple (diagram, symbols,
    output_key, path_col) for DisCoCat/EinsumModel, or a 2-tuple
    (text_col, output_key) -- plain passthrough-with-rename, no diagram/
    symbol deserialisation -- for CLIP text."""
    if text_model_hyperparams.text_backbone == "clip":
        return (text_col, output_key)
    return (diagram_col, symbols_col, output_key, path_col)
