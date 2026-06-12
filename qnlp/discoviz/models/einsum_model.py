from functools import partial
from typing import Any, Dict, List

import torch
import torch.nn as nn
from cotengra import einsum
from lambeq import Symbol

from qnlp.core.non_linear_contraction.einsum_interface import contract_einsum_non_linearly

torch.serialization.add_safe_globals([Symbol])


def get_einsum_model(datasets: list):
    symbol_sizes = dict()
    for ds in datasets:
        for sym, size in zip(ds.symbols, ds.sizes):
            if sym in symbol_sizes and symbol_sizes[sym] != size:
                raise ValueError(f"Symbol {sym} has different sizes in the datasets: {symbol_sizes[sym]} and {size}")
            symbol_sizes[sym] = size

    symbols = list(symbol_sizes.keys())
    sizes = list(symbol_sizes.values())

    model = EinsumModel(symbols, sizes)
    return model


class EinsumModel(nn.Module):
    def __init__(
        self, symbols: List[Symbol] = [], sizes: List[tuple[int, ...]] = [], non_linear_contractions: bool = False
    ):
        """
        symbols: a list of strings (can be any words, with punctuation, etc.)
        """
        if len(symbols) != len(sizes):
            raise ValueError("Symbols and sizes must have the same length.")

        if len(set(symbols)) != len(symbols):
            raise ValueError("Symbols must be unique.")

        super().__init__()
        self.symbols = list(symbols)
        self.sizes = list(sizes)
        self.non_linear_contractions = non_linear_contractions
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(size)) for size in sizes])

        if non_linear_contractions:
            # Global scalar residual gate. Init 0 => the contraction is exactly linear
            # at the start of training (linear floor); the model learns how much
            # non-linearity to add.
            self.nonlinear_gate = nn.Parameter(torch.zeros(()))

        self._setup_contractions_function()
        self.reset_parameters()
        self.sym2weight = self.compute_sym2weight()

    def _setup_contractions_function(self):
        if self.non_linear_contractions:
            self.contractions_function = partial(contract_einsum_non_linearly, non_linear_fn=nn.functional.gelu)
        else:
            self.contractions_function = lambda expr, tensors, path=None, gate=None: einsum(expr, *tensors)

    def compute_sym2weight(self) -> Dict[Symbol, nn.Parameter]:
        return {sym: weight for sym, weight in zip(self.symbols, self.weights)}

    def reset_parameters(self, symbols: List[Symbol] = None):
        def mean(size: int) -> float:
            if size < 6:
                correction_factor = [0, 3, 2.6, 2, 1.6, 1.3][size]
            else:
                correction_factor = 1 / (0.16 * size - 0.04)
            return (size / 3 - 1 / (15 - correction_factor)) ** 0.5

        for sym, weight in zip(self.symbols, self.weights):
            if symbols is not None and sym not in symbols:
                continue
            bound = 1 / mean(sym.directed_cod)
            nn.init.uniform_(weight, -bound, bound)

    def set_weights(self, symbols: List[Symbol], tensors: List[torch.Tensor], freeze: bool = False):
        if len(symbols) != len(tensors):
            raise ValueError("Symbols and tensors must have the same length.")

        if not all(s in self.symbols for s in symbols):
            raise ValueError(f"Some symbols {set(symbols) - set(self.symbols)} are not in the model's symbols list.")

        sym2idx = {sym: idx for idx, sym in enumerate(self.symbols)}
        for sym, tensor in zip(symbols, tensors):
            idx = sym2idx[sym]
            if self.weights[idx].shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for symbol '{sym}': expected {self.weights[idx].shape}, got {tensor.shape}"
                )
            with torch.no_grad():
                self.weights[idx].data.copy_(tensor.data)

    def add_symbols(self, symbols: List[Symbol], sizes: List[tuple[int, ...]]):
        if len(symbols) != len(sizes):
            raise ValueError("Symbols and sizes must have the same length.")

        if any(sym in self.symbols for sym in symbols):
            raise ValueError(f"Some symbols {set(symbols) & set(self.symbols)} already exist in the model.")

        for sym, size in zip(symbols, sizes):
            if sym not in self.symbols:
                new_weight = nn.Parameter(torch.empty(size))
                self.symbols.append(sym)
                self.weights.append(new_weight)
                self.sizes.append(size)

        self.reset_parameters(symbols=symbols)
        self.sym2weight = self.compute_sym2weight()

    def remove_symbols(self, symbols: List[Symbol]):
        sym2idx = {sym: idx for idx, sym in enumerate(self.symbols)}
        indices_to_remove = [sym2idx[sym] for sym in symbols if sym in sym2idx]

        indices_to_remove.sort(reverse=True)
        for idx in indices_to_remove:
            del self.symbols[idx]
            del self.weights[idx]
            del self.sizes[idx]

        self.sym2weight = self.compute_sym2weight()

    def _forward_single(self, input: tuple) -> torch.Tensor:
        # Input is (einsum_expr, symbols) or, for non-linear contractions,
        # (einsum_expr, symbols, path) where path is pre-computed at dataset creation.
        einsum_expr, symbols = input[0], input[1]
        path = input[2] if len(input) > 2 else None

        tensors = [self.sym2weight[sym] for sym in symbols]
        gate = self.nonlinear_gate if self.non_linear_contractions else None
        x = self.contractions_function(einsum_expr, tensors, path, gate=gate)
        if x.ndim != 1:
            shapes = {str(sym): tuple(self.sym2weight[sym].shape) for sym in symbols}
            raise RuntimeError(
                f"Expected 1D output, got shape {tuple(x.shape)}\n  diagram: {einsum_expr}\n  symbol shapes: {shapes}"
            )
        return nn.functional.normalize(x, dim=-1)

    def forward(self, inputs: List[tuple[str, List[Symbol]]]) -> torch.Tensor:
        return torch.stack([self._forward_single(input) for input in inputs])

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        base = super().state_dict(*args, **kwargs)
        base["symbols_list"] = self.symbols
        base["sizes_list"] = self.sizes
        base["non_linear_contractions"] = self.non_linear_contractions
        return base

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        if "symbols_list" in state_dict:
            loaded_symbols = state_dict.pop("symbols_list")
            self.symbols = list(loaded_symbols)
        if "sizes_list" in state_dict:
            loaded_sizes = state_dict.pop("sizes_list")
            self.sizes = list(loaded_sizes)
        if "non_linear_contractions" in state_dict:
            self.non_linear_contractions = state_dict.pop("non_linear_contractions")
            self._setup_contractions_function()

        self.weights = nn.ParameterList([nn.Parameter(torch.empty(size)) for size in self.sizes])

        self.sym2weight = self.compute_sym2weight()
        return super().load_state_dict(state_dict, strict=strict)
