from collections.abc import Callable
from functools import lru_cache

import torch


@lru_cache(maxsize=1000)
def determine_contracted_str_repr(
    left_einsum_repr: str,
    right_einsum_repr: str,
) -> str:
    # Pre-allocate character array
    total_len = len(left_einsum_repr) + len(right_einsum_repr)
    chars = [""] * total_len
    idx = 0

    # Build presence masks
    right_mask = 0
    left_mask = 0

    # Single pass through right to build mask
    for i in range(len(right_einsum_repr)):
        right_mask |= 1 << (ord(right_einsum_repr[i]) - 97)

    # Pass through left
    for i in range(len(left_einsum_repr)):
        ch_ord = ord(left_einsum_repr[i])
        if (right_mask >> (ch_ord - 97)) & 1 == 0:
            chars[idx] = chr(ch_ord)
            idx += 1
        left_mask |= 1 << (ch_ord - 97)

    # Pass through right
    for i in range(len(right_einsum_repr)):
        ch_ord = ord(right_einsum_repr[i])
        if (left_mask >> (ch_ord - 97)) & 1 == 0:
            chars[idx] = chr(ch_ord)
            idx += 1

    # Single allocation for output
    return "".join(chars[:idx])


def non_linear_contraction(
    left_tensor: torch.Tensor,
    left_einsum_repr: str,
    right_tensor: torch.Tensor,
    right_einsum_repr: str,
    non_linear_fn: Callable,
    include_residual: bool = True,
) -> tuple[torch.Tensor, str]:
    output_einsum_repr = determine_contracted_str_repr(
        left_einsum_repr=left_einsum_repr, right_einsum_repr=right_einsum_repr
    )
    contraction = torch.einsum(
        f"{left_einsum_repr},{right_einsum_repr}->{output_einsum_repr}", left_tensor, right_tensor
    )

    output = non_linear_fn(contraction)
    if include_residual:
        # TODO: figure this out
        pass
    return output, output_einsum_repr
