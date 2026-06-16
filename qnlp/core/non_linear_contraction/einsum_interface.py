from collections.abc import Callable

import torch
from cotengra import einsum  # supports non-alpha index characters, unlike torch.einsum

from qnlp.core.non_linear_contraction.atom import non_linear_contraction
from qnlp.core.non_linear_contraction.determine_optimal_contraction_path import get_contraction_path


def contract_einsum_non_linearly(
    einsum_str: str,
    symbols: list[torch.Tensor],
    path: list[tuple[int, int]] | None = None,
    *,
    non_linear_fn: Callable,
    gate: torch.Tensor | float | None = None,
) -> torch.Tensor:
    parts = einsum_str.split("->")
    einsum_input = parts[0]
    protected_indices = parts[1] if len(parts) > 1 else ""

    reprs = einsum_input.split(",")
    assert len(reprs) == len(symbols), "Einsum and symbols do not match"

    if path is None:
        path = get_contraction_path(einsum_str, tuple(s.shape for s in symbols))
    operands = list(zip(reprs, symbols))

    for i, j in path:
        left_repr, left_tensor = operands.pop(j)
        right_repr, right_tensor = operands.pop(i)

        # On a too-large intermediate this raises IntermediateTooLargeError, which
        # propagates up to EinsumModel._forward_single so the diagram is skipped
        # (rather than silently falling back to a linear contraction).
        new_tensor, new_repr = non_linear_contraction(
            left_tensor=left_tensor,
            left_einsum_repr=left_repr,
            right_tensor=right_tensor,
            right_einsum_repr=right_repr,
            non_linear_fn=non_linear_fn,
            protected_indices=protected_indices,
            gate=gate,
        )
        operands.append((new_repr, new_tensor))

    curr_repr, curr_symbol = operands[0]
    if curr_repr != protected_indices:
        curr_symbol = einsum(f"{curr_repr}->{protected_indices}", curr_symbol)
    return curr_symbol
