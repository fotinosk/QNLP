"""
EinsumModel with tied noun embeddings across grammatical forms.

Each unique lemma (e.g. "lick") gets one shared embedding vector of size
embedding_dim.  All symbols derived from that lemma — noun, verb components,
adjective forms — share this embedding as the semantic core.  Only the
bond-dimension factors are learned independently per symbol.

For a verb-core tensor of shape [bond_dim, embedding_dim, bond_dim]:

    T[a, s, c] = noun[s] * M[a, c]
    M[a, c]    = sum_r  u_r[a] * v_r[c]    (rank-R bond matrix)

This means every occurrence of "lick" in any role trains the same 256D (or
512D) noun embedding, giving it ~5-6x more gradient signal than a symbol seen
once per role.  The bond matrix M captures role-specific selectional behaviour.

Lemma extraction: symbol names follow the pattern {lemma}_{index}__{type},
e.g. "lick_1__B.r@s@B" → lemma "lick".  Symbols that do not match fall back
to using the full name as their lemma (no sharing, but no failure).
"""

import re
from typing import Any, Dict, List

import torch
import torch.nn as nn
from lambeq import Symbol
from opt_einsum import contract as einsum

torch.serialization.add_safe_globals([Symbol])

_LEMMA_RE = re.compile(r"^(.+?)_\d+__")


def _extract_lemma(sym: Symbol) -> str:
    m = _LEMMA_RE.match(str(sym))
    return m.group(1) if m else str(sym)


class EinsumModelTiedNoun(nn.Module):
    def __init__(
        self,
        symbols: List[Symbol],
        sizes: List[tuple[int, ...]],
        embedding_dim: int,
        bond_dim: int,
        bond_rank: int = 4,
    ):
        if len(symbols) != len(sizes):
            raise ValueError("symbols and sizes must have the same length.")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique.")

        super().__init__()
        self.symbols = list(symbols)
        self.sizes = list(sizes)
        self.embedding_dim = embedding_dim
        self.bond_dim = bond_dim
        self.bond_rank = bond_rank
        self.sym2idx: Dict[Symbol, int] = {sym: i for i, sym in enumerate(symbols)}

        # Extract lemma for each symbol (parallel list).
        self.sym_lemmas: List[str] = [_extract_lemma(sym) for sym in symbols]

        # Deduplicated lemma list (insertion-ordered).
        seen: dict[str, int] = {}
        for lemma in self.sym_lemmas:
            if lemma not in seen:
                seen[lemma] = len(seen)
        self.unique_lemmas: List[str] = list(seen.keys())
        self.lemma2idx: Dict[str, int] = seen

        # One shared noun embedding per unique lemma.
        self.noun_embeddings = nn.ParameterList([nn.Parameter(torch.empty(embedding_dim)) for _ in self.unique_lemmas])
        for p in self.noun_embeddings:
            nn.init.normal_(p, std=0.5)

        # Per-symbol bond factors (empty ParameterList for pure nouns).
        self.bond_factors = nn.ModuleList([self._init_bond_factors(size) for size in sizes])

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_bond_factors(self, size: tuple[int, ...]) -> nn.ParameterList:
        bond_positions = [i for i, d in enumerate(size) if d == self.bond_dim]
        n_bond = len(bond_positions)

        if n_bond == 0:
            return nn.ParameterList([])

        if n_bond == 1:
            p = nn.Parameter(torch.empty(self.bond_dim))
            nn.init.normal_(p, std=0.1)
            return nn.ParameterList([p])

        # Two or more bond dims: one [bond_rank, bond_dim] factor matrix per bond dim.
        factors = []
        for _ in range(n_bond):
            f = nn.Parameter(torch.empty(self.bond_rank, self.bond_dim))
            nn.init.normal_(f, std=0.1)
            factors.append(f)
        return nn.ParameterList(factors)

    # ------------------------------------------------------------------
    # Tensor materialisation
    # ------------------------------------------------------------------

    def _materialise(self, sym: Symbol) -> torch.Tensor:
        i = self.sym2idx[sym]
        size = self.sizes[i]
        noun = self.noun_embeddings[self.lemma2idx[self.sym_lemmas[i]]]
        bond_params = self.bond_factors[i]
        ndim = len(size)

        if ndim == 1:
            return noun

        emb_pos = next(j for j, d in enumerate(size) if d == self.embedding_dim)

        if ndim == 2:
            bond_vec = bond_params[0]
            if emb_pos == 0:
                return torch.einsum("s,b->sb", noun, bond_vec)
            else:
                return torch.einsum("b,s->bs", bond_vec, noun)

        if ndim == 3:
            u, v = bond_params[0], bond_params[1]
            # Bond matrix over the two bond dimensions.
            M = torch.einsum("ra,rc->ac", u, v)  # [bond_dim, bond_dim]
            if emb_pos == 0:
                return torch.einsum("s,ab->sab", noun, M)
            elif emb_pos == 1:
                return torch.einsum("ab,s->asb", M, noun)
            else:
                return torch.einsum("ab,s->abs", M, noun)

        raise ValueError(f"Unsupported tensor rank {ndim} for symbol '{sym}' with size {size}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_single(self, input: tuple) -> torch.Tensor:
        einsum_expr, symbols = input[0], input[1]
        tensors = [self._materialise(sym) for sym in symbols]
        x = einsum(einsum_expr, *tensors)
        if x.ndim != 1:
            raise RuntimeError(f"Expected 1D output, got shape {tuple(x.shape)}\n  expr: {einsum_expr}")
        return nn.functional.normalize(x, dim=-1)

    def forward(self, inputs: List[tuple]) -> torch.Tensor:
        return torch.stack([self._forward_single(inp) for inp in inputs])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        base = super().state_dict(*args, **kwargs)
        base["symbols_list"] = self.symbols
        base["sizes_list"] = self.sizes
        base["embedding_dim"] = self.embedding_dim
        base["bond_dim"] = self.bond_dim
        base["bond_rank"] = self.bond_rank
        return base

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        mapping = {
            "symbols_list": "symbols",
            "sizes_list": "sizes",
            "embedding_dim": "embedding_dim",
            "bond_dim": "bond_dim",
            "bond_rank": "bond_rank",
        }
        for src, dst in mapping.items():
            if src in state_dict:
                setattr(self, dst, state_dict.pop(src))
        self.sym2idx = {sym: i for i, sym in enumerate(self.symbols)}
        self.sym_lemmas = [_extract_lemma(sym) for sym in self.symbols]
        seen: dict[str, int] = {}
        for lemma in self.sym_lemmas:
            if lemma not in seen:
                seen[lemma] = len(seen)
        self.unique_lemmas = list(seen.keys())
        self.lemma2idx = seen
        return super().load_state_dict(state_dict, strict=strict)
