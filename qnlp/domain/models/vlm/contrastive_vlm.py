import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel


class ContrastiveVLM(nn.Module):
    """
    Model wrapper for contrastive VLM training.

    Holds text and image sub-models as named children so a single
    Trainer.checkpoint covers both. Exposes clip_gradients so the
    Trainer can clip each sub-model independently.
    """

    def __init__(self, text_model: EinsumModel, image_model: TTNImageModel):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model

    def forward(self, images, true_captions, false_captions=None) -> dict:
        image_emb = self.image_model(images)
        true_emb = self.text_model(true_captions)
        outputs = {
            "image_embeddings": image_emb,
            "true_caption_embeddings": true_emb,
        }
        if false_captions is not None:
            outputs["false_caption_embeddings"] = self.text_model(false_captions)
        return outputs

    def load_state_dict(self, state_dict, strict: bool = True):
        # EinsumModel injects "symbols_list"/"sizes_list" as bare (unprefixed) keys.
        # PyTorch's recursive load calls _load_from_state_dict, not load_state_dict,
        # so EinsumModel.load_state_dict (which rebuilds its ParameterList) never
        # fires automatically. We manually split and delegate to each submodule.
        text_sd, image_sd = {}, {}
        for k, v in state_dict.items():
            if k in ("symbols_list", "sizes_list"):
                text_sd[k] = v
            elif k.startswith("text_model."):
                text_sd[k[len("text_model.") :]] = v
            elif k.startswith("image_model."):
                image_sd[k[len("image_model.") :]] = v
        self.text_model.load_state_dict(text_sd, strict=strict)
        self.image_model.load_state_dict(image_sd, strict=strict)

    def clip_gradients(self, max_norm: float) -> None:
        clip_grad_norm_(self.text_model.parameters(), max_norm)
        clip_grad_norm_(self.image_model.parameters(), max_norm)
