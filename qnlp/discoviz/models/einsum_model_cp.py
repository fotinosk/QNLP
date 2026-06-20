"""
EinsumModel with CP (Canonical Polyadic) decomposition of word tensors.

Instead of storing one full parameter tensor per symbol, each symbol's tensor
is represented as a sum of R outer products of factor vectors — one factor
vector per tensor dimension.  For a verb-core tensor of shape [20, 256, 20]:

    T[a, b, c] = sum_r  f0[r,a] * f1[r,b] * f2[r,c]

stored as three factor matrices of shape [R,20], [R,256], [R,20].

Parameter count per verb tensor: R*(20+256+20)=1,184 at R=4 vs 102,400 full.

Pure noun vectors (rank-1 tensors) are stored directly — no decomposition.
All contractions are linear (no NLC gate).
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
from cotengra import einsum
from lambeq import Symbol

torch.serialization.add_safe_globals([Symbol])

_EINSUM_IDX = "abcdefghij"


class EinsumModelCP(nn.Module):
    def __init__(
        self,
        symbols: List[Symbol],
        sizes: List[tuple[int, ...]],
        rank: int = 4,
    ):
        if len(symbols) != len(sizes):
            raise ValueError("symbols and sizes must have the same length.")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique.")

        super().__init__()
        self.symbols = list(symbols)
        self.sizes = list(sizes)
        self.rank = rank
        self.sym2idx: Dict[Symbol, int] = {sym: i for i, sym in enumerate(symbols)}

        self.cp_factors = nn.ModuleList([self._init_factors(size) for size in sizes])

    def _init_factors(self, size: tuple[int, ...]) -> nn.ParameterList:
        ndim = len(size)
        if ndim <= 1:
            # Pure vector — no decomposition.
            p = nn.Parameter(torch.empty(*size) if size else torch.empty(()))
            nn.init.normal_(p, std=0.5)
            return nn.ParameterList([p])

        factors: list[nn.Parameter] = []
        for d in size:
            f = nn.Parameter(torch.empty(self.rank, d))
            # Target materialised tensor std ≈ 0.1.
            # std(T) = R^0.5 * std_f^ndim  →  std_f = (0.1 / R^0.5)^(1/ndim)
            std = (0.1 / self.rank**0.5) ** (1.0 / ndim)
            nn.init.normal_(f, std=std)
            factors.append(f)
        return nn.ParameterList(factors)

    def _materialise(self, sym: Symbol) -> torch.Tensor:
        i = self.sym2idx[sym]
        size = self.sizes[i]
        factors = self.cp_factors[i]
        ndim = len(size)

        if ndim <= 1:
            return factors[0]

        # Build 'ra,rb,rc,...->abc...' dynamically.
        out_idx = _EINSUM_IDX[:ndim]
        lhs = ",".join(f"r{out_idx[k]}" for k in range(ndim))
        expr = f"{lhs}->{out_idx}"
        return torch.einsum(expr, *list(factors))

    def _forward_single(self, input: tuple) -> torch.Tensor:
        einsum_expr, symbols = input[0], input[1]
        tensors = [self._materialise(sym) for sym in symbols]
        x = einsum(einsum_expr, *tensors)
        if x.ndim != 1:
            raise RuntimeError(f"Expected 1D output, got shape {tuple(x.shape)}\n  expr: {einsum_expr}")
        return nn.functional.normalize(x, dim=-1)

    def forward(self, inputs: List[tuple]) -> torch.Tensor:
        return torch.stack([self._forward_single(inp) for inp in inputs])

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        base = super().state_dict(*args, **kwargs)
        base["symbols_list"] = self.symbols
        base["sizes_list"] = self.sizes
        base["rank"] = self.rank
        return base

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        if "symbols_list" in state_dict:
            self.symbols = list(state_dict.pop("symbols_list"))
        if "sizes_list" in state_dict:
            self.sizes = list(state_dict.pop("sizes_list"))
        if "rank" in state_dict:
            self.rank = state_dict.pop("rank")
        self.sym2idx = {sym: i for i, sym in enumerate(self.symbols)}
        return super().load_state_dict(state_dict, strict=strict)
