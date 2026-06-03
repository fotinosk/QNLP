from collections.abc import Callable

import torch

from qnlp.core.non_linear_contraction.atom import non_linear_contraction


def contract_einsum_non_linearly(einsum_str: str, symbols: list[torch.Tensor], non_linear_fn: Callable) -> torch.Tensor:
    einsum_input = einsum_str.split("->")[0]
    einsum_components = einsum_input.split(",")[::-1]
    symbols = symbols[::-1]

    assert len(einsum_components) == len(symbols), "Einsum and symbols do not match"

    curr_symbol = symbols.pop()
    curr_einsum_repr = einsum_components.pop()

    while einsum_components:
        next_symbol = symbols.pop()
        next_einsum_repr = einsum_components.pop()

        curr_symbol, curr_einsum_repr = non_linear_contraction(
            left_tensor=curr_symbol,
            left_einsum_repr=curr_einsum_repr,
            right_tensor=next_symbol,
            right_einsum_repr=next_einsum_repr,
            non_linear_fn=non_linear_fn,
        )
    return curr_symbol
