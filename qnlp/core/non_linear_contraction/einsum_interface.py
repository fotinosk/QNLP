from collections.abc import Callable

import torch
from cotengra import einsum  # supports non-alpha index characters, unlike torch.einsum

from qnlp.core.non_linear_contraction.atom import IntermediateTooLargeError, non_linear_contraction
from qnlp.core.non_linear_contraction.determine_optimal_contraction_path import get_contraction_path
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="non_linear_contraction")


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

        try:
            new_tensor, new_repr = non_linear_contraction(
                left_tensor=left_tensor,
                left_einsum_repr=left_repr,
                right_tensor=right_tensor,
                right_einsum_repr=right_repr,
                non_linear_fn=non_linear_fn,
                protected_indices=protected_indices,
                gate=gate,
            )
        except IntermediateTooLargeError as e:
            logger.warning(
                f"Large intermediate in diagram {einsum_str!r} — "
                f"falling back to linear einsum for full diagram. {e}"
            )
            result = einsum(einsum_str, *symbols)
            g = gate if gate is not None else 1.0
            return result + g * non_linear_fn(result)

        operands.append((new_repr, new_tensor))

    curr_repr, curr_symbol = operands[0]
    if curr_repr != protected_indices:
        curr_symbol = einsum(f"{curr_repr}->{protected_indices}", curr_symbol)
    return curr_symbol
