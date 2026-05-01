from typing import Protocol

import torch
from torch import Tensor


class LossFunction(Protocol):
    def __call__(self, outputs: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute loss from model outputs.

        Args:
            outputs: Dict of named tensors from the model forward pass
                     (e.g. image_embeddings, true_caption_embeddings).

        Returns:
            (loss, metrics) where loss is used for .backward() and metrics is a
            dict of on-device tensors (including "loss") passed to the accumulator.
            No .item() calls here — stay on device.
        """
        ...


class TrainingStep(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        batch: dict,
        train: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Run one batch through the model and compute loss + metrics.

        Args:
            model: The model wrapper (contains sub-models as nn.Module attributes).
            batch: Collated batch dict from VLMDataset / vlm_collate_fn.
            train: If True, model is in train mode and gradients are active.
                   If False, model is in eval mode under torch.no_grad().

        Returns:
            (loss, metrics) where all metric tensors are on-device.
            The metrics dict must include "loss" for the accumulator.
            No .item() calls — the Trainer handles epoch-end sync.
        """
        ...
