import math
from collections.abc import Callable
from functools import lru_cache

import torch
from cotengra import einsum

from qnlp.core.non_linear_contraction.determine_optimal_contraction_path import MAX_INTERMEDIATE_ELEMENTS
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="non_linear_contraction")


class IntermediateTooLargeError(RuntimeError):
    pass


@lru_cache(maxsize=1000)
def determine_contracted_str_repr(
    left_einsum_repr: str,
    right_einsum_repr: str,
    protected_indices: str = "",
) -> str:
    shared = set(left_einsum_repr) & set(right_einsum_repr)
    to_contract = shared - set(protected_indices)
    seen: set[str] = set()
    result = []
    for c in left_einsum_repr + right_einsum_repr:
        if c not in to_contract and c not in seen:
            result.append(c)
            seen.add(c)
    return "".join(result)


def non_linear_contraction(
    left_tensor: torch.Tensor,
    left_einsum_repr: str,
    right_tensor: torch.Tensor,
    right_einsum_repr: str,
    non_linear_fn: Callable,
    include_residual: bool = True,
    protected_indices: str = "",
    gate: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, str]:
    output_einsum_repr = determine_contracted_str_repr(
        left_einsum_repr=left_einsum_repr,
        right_einsum_repr=right_einsum_repr,
        protected_indices=protected_indices,
    )

    size_map = dict(zip(left_einsum_repr, left_tensor.shape))
    size_map.update(zip(right_einsum_repr, right_tensor.shape))
    expected_numel = math.prod(size_map[c] for c in output_einsum_repr) if output_einsum_repr else 1

    if expected_numel > MAX_INTERMEDIATE_ELEMENTS:
        raise IntermediateTooLargeError(
            f"left={left_einsum_repr}{tuple(left_tensor.shape)} "
            f"right={right_einsum_repr}{tuple(right_tensor.shape)} "
            f"-> {output_einsum_repr} ({expected_numel:,} elements)"
        )

    contraction = einsum(f"{left_einsum_repr},{right_einsum_repr}->{output_einsum_repr}", left_tensor, right_tensor)

    nonlinear = non_linear_fn(contraction)
    if include_residual:
        # Gated residual: at gate=0 this is exactly the linear contraction (linear floor).
        g = gate if gate is not None else 1.0
        output = contraction + g * nonlinear
    else:
        output = nonlinear
    return output, output_einsum_repr
