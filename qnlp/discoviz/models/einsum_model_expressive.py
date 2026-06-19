"""
EinsumModel with two independent expressivity improvements:

  1. Per-symbol NLC gates
     Each symbol carries a learnable scalar gate initialised to 0 (linear floor).
     When two tensors are contracted the effective gate is the arithmetic mean of
     their gates; intermediates inherit that mean, propagating recursively. This
     lets the model learn which words benefit from non-linear composition
     (e.g. verbs > determiners) without accumulating a single global value through
     every step of a long chain.

  2. Word bypass
     A learnable embedding of size embedding_dim is stored per symbol. At each
     forward pass the bypass embeddings for all symbols in the sentence are
     mean-pooled and added to the contraction result before L2 normalisation. This
     gives every word a direct path to the output, bypassing the depth bottleneck
     of the contraction chain.

Both additions are zero-initialised so the model is equivalent to the base
EinsumModel at the start of training.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
from cotengra import einsum
from lambeq import Symbol

from qnlp.core.non_linear_contraction.atom import IntermediateTooLargeError, non_linear_contraction
from qnlp.discoviz.models.einsum_model import EinsumModel


class EinsumModelExpressive(EinsumModel):
    def __init__(
        self,
        symbols: List[Symbol],
        sizes: List[tuple[int, ...]],
        embedding_dim: int,
    ):
        # Always NLC — per-symbol gates are meaningless without it.
        super().__init__(symbols, sizes, non_linear_contractions=True)
        self.embedding_dim = embedding_dim
        n = len(symbols)

        # Per-symbol scalar NLC gates.  Init=0 preserves the linear floor.
        self.symbol_gates = nn.Parameter(torch.zeros(n))

        # Word bypass: per-symbol bypass embedding + projection.
        # bypass_embeddings init=0; bypass_proj init=identity so that
        # bypass_embeddings receive non-zero gradients from step 1 even though
        # bypass contribution is zero at initialisation.
        self.bypass_embeddings = nn.Parameter(torch.zeros(n, embedding_dim))
        self.bypass_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.eye_(self.bypass_proj.weight)

        self._rebuild_sym2gate_idx()

    def _rebuild_sym2gate_idx(self) -> None:
        self.sym2gate_idx: Dict[Symbol, int] = {sym: i for i, sym in enumerate(self.symbols)}

    def _forward_single(self, input: tuple) -> torch.Tensor:
        einsum_expr, symbols = input[0], input[1]
        stored_path = input[2] if len(input) > 2 else None

        tensors = [self.sym2weight[sym] for sym in symbols]

        # Path resolution — same fallback logic as base class.
        if stored_path is not None:
            path = stored_path
        else:
            path = self._get_path(einsum_expr, tensors)

        if path is None:
            return self._nan_output(einsum_expr, tensors)

        parts = einsum_expr.split("->")
        protected_indices = parts[1] if len(parts) > 1 else ""
        reprs = parts[0].split(",")

        gate_indices = [self.sym2gate_idx[sym] for sym in symbols]
        operands: list[tuple[str, torch.Tensor]] = list(zip(reprs, tensors))
        # Parallel list of per-operand gates; intermediates inherit mean of constituents.
        operand_gates: list[torch.Tensor] = [self.symbol_gates[idx] for idx in gate_indices]

        try:
            for i, j in path:
                left_repr, left_tensor = operands.pop(j)
                right_repr, right_tensor = operands.pop(i)
                left_gate = operand_gates.pop(j)
                right_gate = operand_gates.pop(i)

                effective_gate = (left_gate + right_gate) / 2

                new_tensor, new_repr = non_linear_contraction(
                    left_tensor=left_tensor,
                    left_einsum_repr=left_repr,
                    right_tensor=right_tensor,
                    right_einsum_repr=right_repr,
                    non_linear_fn=nn.functional.gelu,
                    protected_indices=protected_indices,
                    gate=effective_gate,
                )
                operands.append((new_repr, new_tensor))
                operand_gates.append(effective_gate)

        except IntermediateTooLargeError:
            return self._nan_output(einsum_expr, tensors)

        curr_repr, x = operands[0]
        if curr_repr != protected_indices:
            x = einsum(f"{curr_repr}->{protected_indices}", x)

        if x.ndim != 1:
            shapes = {str(sym): tuple(self.sym2weight[sym].shape) for sym in symbols}
            raise RuntimeError(
                f"Expected 1D output, got shape {tuple(x.shape)}\n"
                f"  diagram: {einsum_expr}\n  symbol shapes: {shapes}"
            )

        # Word bypass: mean-pool per-symbol bypass embeddings and add to x.
        bypass_idxs = torch.tensor(gate_indices, dtype=torch.long, device=x.device)
        bypass_vec = self.bypass_embeddings[bypass_idxs].mean(dim=0)
        x = x + self.bypass_proj(bypass_vec)

        return nn.functional.normalize(x, dim=-1)

    def _nan_output(self, einsum_expr: str, tensors: list[torch.Tensor]) -> torch.Tensor:
        output_idx = einsum_expr.split("->")[1]
        size_map = {
            c: dim for repr_, t in zip(einsum_expr.split("->")[0].split(","), tensors) for c, dim in zip(repr_, t.shape)
        }
        out_dim = size_map[output_idx[0]] if output_idx else 1
        return torch.full((out_dim,), float("nan"), device=tensors[0].device, dtype=tensors[0].dtype)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        base = super().state_dict(*args, **kwargs)
        base["embedding_dim"] = self.embedding_dim
        return base

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        if "embedding_dim" in state_dict:
            self.embedding_dim = state_dict.pop("embedding_dim")
        result = super().load_state_dict(state_dict, strict=strict)
        self._rebuild_sym2gate_idx()
        return result
