from typing import Any, Dict, List

import torch
import torch.nn as nn
from lambeq import Symbol

from qnlp.core.non_linear_contraction.atom import IntermediateTooLargeError
from qnlp.core.non_linear_contraction.determine_optimal_contraction_path import get_left_to_right_path
from qnlp.discoviz.models.einsum_model import EinsumModel


class EinsumModelLR(EinsumModel):
    """EinsumModel with left-to-right fold contraction order.

    Replaces opt_einsum path planning with a sequential fold-left path computed
    on-the-fly from operand count alone. Each NLC gate application then corresponds
    to one incremental left-to-right composition step, so intermediate tensors always
    represent contiguous left prefixes of the sentence.

    Always runs with non_linear_contractions=True — there is no point using LR ordering
    without NLC (pure linear contraction is order-invariant).
    """

    def __init__(self, symbols: List[Symbol] = [], sizes: List[tuple[int, ...]] = []):
        super().__init__(symbols, sizes, non_linear_contractions=True)

    def _forward_single(self, input: tuple) -> torch.Tensor:
        einsum_expr, symbols = input[0], input[1]
        # stored path (input[2]) is intentionally ignored — LR path is derived from operand count
        tensors = [self.sym2weight[sym] for sym in symbols]
        path = get_left_to_right_path(len(tensors))

        try:
            x = self.contractions_function(einsum_expr, tensors, path, gate=self.nonlinear_gate)
        except IntermediateTooLargeError:
            output_idx = einsum_expr.split("->")[1]
            size_map = {
                c: dim
                for repr_, t in zip(einsum_expr.split("->")[0].split(","), tensors)
                for c, dim in zip(repr_, t.shape)
            }
            out_dim = size_map[output_idx[0]] if output_idx else 1
            return torch.full((out_dim,), float("nan"), device=tensors[0].device, dtype=tensors[0].dtype)

        if x.ndim != 1:
            shapes = {str(sym): tuple(self.sym2weight[sym].shape) for sym in symbols}
            raise RuntimeError(
                f"Expected 1D output, got shape {tuple(x.shape)}\n  diagram: {einsum_expr}\n  symbol shapes: {shapes}"
            )
        return nn.functional.normalize(x, dim=-1)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        base = super().state_dict(*args, **kwargs)
        base["path_strategy"] = "left_to_right"
        return base

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        state_dict.pop("path_strategy", None)
        return super().load_state_dict(state_dict, strict=strict)
