"""LemmaTiedEinsumModel — rank-R lemma-tied (CP-decomposition) EinsumModel.

The original `EinsumModel` stores one free `nn.Parameter` per typed CCG symbol.
With 80 622 typed symbols and a long-tail occurrence distribution (36 % seen
exactly once, 71 % seen ≤ 10 times in 14.1 M total occurrences), most per-symbol
tensors get too little gradient signal to generalise.

This model factorises every typed tensor as a **rank-R sum of outer products**
with a shared lemma factor across typed variants of the same word:

    T(lemma=w, type=t)  =  sum_{r=1..R}  bond_left[t,r]  ⊗  lemma_emb[w,r]  ⊗  bond_right[t,r]

where bond_left/right are absent when the type has no bond on that side, and
the sum is over R independent rank-1 components.

  * R=1 (pure rank-1) collapses every typed variant of a lemma to a single
    embedding direction — empirically too restrictive on COCO (training itself
    fails to converge because all outputs collinear).
  * R=4-8 retains the parameter-sharing across typed variants while letting
    different variants project into different combinations of R lemma
    directions. Each variant's effective embedding is a learned linear
    combination of the lemma's R rank-components, with the combination weights
    coming from the per-type bond factors.

Effects:
  * Lemma signal shared: every typed variant of word `w` updates the same
    `(R, D)` lemma matrix.
  * Type signal shared: every lemma using type `t` updates the same per-type
    `(R, b)` bond factors.
  * Param budget (V=25k lemmas, T=32 types, R=4, D=512, b=10):
    25 k·R·D + T·R·b·2 = ~51 M trainable, vs original 1.04 B (~20× compression).

Interface:
  * Mirrors EinsumModel: `__init__(symbols, sizes, rank=...)`, `forward(inputs)`,
    `state_dict`, `load_state_dict`. Drops into `ContrastiveVLM` unchanged.
  * Embedding dim and bond dim are inferred from the supplied `sizes`.
  * `rank` is stored in state_dict so checkpoints round-trip cleanly.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from cotengra import einsum
from lambeq import Symbol

torch.serialization.add_safe_globals([Symbol])


# Pattern for CCG symbol names from the lambeq compiler: <lemma>_<N>__<type>.
_NAME_RE = re.compile(r"^(.+)_(\d+)__(.+)$")


def _parse_name(name: str) -> Tuple[str, str]:
    """Split a Symbol.name into (lemma, type_key).

    Examples:
        "dog_0__n"              -> ("dog", "n")
        "dog_2__n.r@B"          -> ("dog", "n.r@B")
        "be_1__B.r@s@B"         -> ("be", "B.r@s@B")
    """
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"Cannot parse Symbol.name into <lemma>_<N>__<type>: {name!r}")
    return m.group(1), m.group(3)


def _bond_axes(size: Tuple[int, ...], embedding_dim: int) -> Tuple[int, int, int]:
    """Return (embedding_axis, num_left_bonds, num_right_bonds) for a given shape.

    Every observed shape has exactly one embedding-sized axis; the rest are
    bond-sized.
    """
    axes_at_emb = [i for i, d in enumerate(size) if d == embedding_dim]
    if len(axes_at_emb) != 1:
        raise ValueError(
            f"Expected exactly one embedding-dim axis in size {size}, "
            f"got {len(axes_at_emb)} (embedding_dim={embedding_dim})"
        )
    emb_axis = axes_at_emb[0]
    return emb_axis, emb_axis, len(size) - emb_axis - 1


class LemmaTiedEinsumModel(nn.Module):
    """Rank-R lemma-tied EinsumModel via CP decomposition.

    Args:
        symbols: list of lambeq Symbols (typed CCG symbols).
        sizes: parallel list of tensor shapes, one per symbol.
        rank: CP-decomposition rank R. Default 4. R=1 collapses to the
            rank-1 outer product (empirically too restrictive). R≥4 lets
            different typed variants of the same lemma occupy different
            directions in a learned R-dim subspace.

    Builds one (R, D) trainable matrix per unique lemma and per-type bond
    factors of shape (R, b). Reconstructs each typed symbol's tensor
    on-the-fly during forward via a small einsum.
    """

    def __init__(
        self,
        symbols: List[Symbol] = (),
        sizes: List[Tuple[int, ...]] = (),
        rank: int = 4,
    ):
        super().__init__()
        if len(symbols) != len(sizes):
            raise ValueError("Symbols and sizes must have the same length.")
        if len(set(symbols)) != len(symbols):
            raise ValueError("Symbols must be unique.")
        if rank < 1:
            raise ValueError(f"rank must be ≥ 1, got {rank}")

        self.symbols: List[Symbol] = list(symbols)
        self.sizes: List[Tuple[int, ...]] = [tuple(s) for s in sizes]
        self.rank = int(rank)

        # Infer dims. Each size must contain exactly one "large" axis (the
        # embedding dim) and the rest "small" axes (the bond dim).
        all_dims = sorted({d for s in self.sizes for d in s})
        if not all_dims:
            self.embedding_dim = 512
            self.bond_dim = 10
        elif len(all_dims) == 1:
            self.embedding_dim = all_dims[0]
            self.bond_dim = 10  # placeholder; no bond factors will be referenced
        else:
            self.embedding_dim = all_dims[-1]
            self.bond_dim = all_dims[0]

        self._build_vocab_and_recipes()

        # CP-decomposition parameters.
        # lemma_emb : (V, R, D)  shared across all typed variants of each lemma
        # bond_left : (T, R, b)  per-type left-bond factor (rank-R)
        # bond_right: (T, R, b)  per-type right-bond factor (rank-R)
        self.lemma_emb = nn.Parameter(
            torch.empty(len(self.lemma_to_idx), self.rank, self.embedding_dim)
        )
        self.bond_left = nn.Parameter(
            torch.empty(len(self.type_to_idx), self.rank, self.bond_dim)
        )
        self.bond_right = nn.Parameter(
            torch.empty(len(self.type_to_idx), self.rank, self.bond_dim)
        )

        self.reset_parameters()

    def _build_vocab_and_recipes(self) -> None:
        """Build lemma / type vocabularies and per-symbol reconstruction recipes."""
        self.lemma_to_idx: Dict[str, int] = {}
        self.type_to_idx: Dict[str, int] = {}
        # For each symbol i: (lemma_idx, type_idx, emb_axis, n_left, n_right).
        self.recipes: List[Tuple[int, int, int, int, int]] = []

        for sym, size in zip(self.symbols, self.sizes):
            lemma, type_key = _parse_name(sym.name)
            if lemma not in self.lemma_to_idx:
                self.lemma_to_idx[lemma] = len(self.lemma_to_idx)
            if type_key not in self.type_to_idx:
                self.type_to_idx[type_key] = len(self.type_to_idx)
            emb_axis, n_left, n_right = _bond_axes(size, self.embedding_dim)
            self.recipes.append(
                (self.lemma_to_idx[lemma], self.type_to_idx[type_key], emb_axis, n_left, n_right)
            )

        # Convenience: by-symbol view for forward.
        self.sym2recipe: Dict[Symbol, Tuple[int, int, int, int, int]] = {
            sym: rec for sym, rec in zip(self.symbols, self.recipes)
        }

    def reset_parameters(self) -> None:
        """Initialise the tied parameters.

        For rank-R sum of outer products, we want each rank component's
        contribution to the reconstructed tensor to have entries with std ~ 0.33
        (matching the original EinsumModel's (10, 512) init scale), and the sum
        of R independent components gives variance ≈ R times one component.

        Setting each factor matrix to uniform[-1, 1] gives per-rank entry
        std = (1/sqrt(3))**(n_factors), where n_factors ∈ {1, 2, 3} depending on
        shape. After summing R rank-components and normalising, the long
        cotengra contraction stays well-conditioned. We init the lemma_emb a
        bit smaller (uniform[-0.5, 0.5]) than the bond factors to keep the
        per-lemma magnitudes more like a typical embedding layer.
        """
        nn.init.uniform_(self.lemma_emb, -0.5, 0.5)
        nn.init.uniform_(self.bond_left, -1.0, 1.0)
        nn.init.uniform_(self.bond_right, -1.0, 1.0)

    def _reconstruct(self, recipe: Tuple[int, int, int, int, int]) -> torch.Tensor:
        """Build the rank-R CP reconstruction of a single typed tensor.

        recipe = (lemma_idx, type_idx, embedding_axis, n_left_bonds, n_right_bonds)

        Output shape matches the original `size`:
            (D,)           → sum_r lemma_emb[v, r, :]
            (b, D)         → einsum('rb,rd->bd', bond_left[t], lemma_emb[v])
            (D, b)         → einsum('rd,rb->db', lemma_emb[v], bond_right[t])
            (b, D, b)      → einsum('rb,rd,rB->bdB', bl[t], lemma_emb[v], br[t])
        """
        lemma_idx, type_idx, _emb_axis, n_left, n_right = recipe
        emb = self.lemma_emb[lemma_idx]  # (R, D)
        if n_left == 0 and n_right == 0:
            # Shape (D,) — sum the R rank components.
            return emb.sum(dim=0)
        if n_left == 1 and n_right == 0:
            bl = self.bond_left[type_idx]  # (R, b)
            return torch.einsum("rb,rd->bd", bl, emb)
        if n_left == 0 and n_right == 1:
            br = self.bond_right[type_idx]  # (R, b)
            return torch.einsum("rd,rb->db", emb, br)
        if n_left == 1 and n_right == 1:
            bl = self.bond_left[type_idx]  # (R, b)
            br = self.bond_right[type_idx]  # (R, b)
            return torch.einsum("rb,rd,rB->bdB", bl, emb, br)
        raise NotImplementedError(
            f"Unsupported bond counts: left={n_left}, right={n_right}. "
            f"Need to extend reconstruction for this shape pattern."
        )

    def _forward_single(self, input: Tuple[str, List[Symbol]]) -> torch.Tensor:
        einsum_expr, symbols = input
        tensors = [self._reconstruct(self.sym2recipe[sym]) for sym in symbols]
        x = einsum(einsum_expr, *tensors)
        return nn.functional.normalize(x, dim=-1)

    def forward(self, inputs: List[Tuple[str, List[Symbol]]]) -> torch.Tensor:
        return torch.stack([self._forward_single(input) for input in inputs])

    # ---------- state_dict plumbing (mirrors EinsumModel's contract) ----------

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
            self.sizes = [tuple(s) for s in state_dict.pop("sizes_list")]
        if "rank" in state_dict:
            self.rank = int(state_dict.pop("rank"))
        self._build_vocab_and_recipes()
        # Rebuild parameters with the new sizes so the load matches.
        self.lemma_emb = nn.Parameter(
            torch.empty(len(self.lemma_to_idx), self.rank, self.embedding_dim)
        )
        self.bond_left = nn.Parameter(
            torch.empty(len(self.type_to_idx), self.rank, self.bond_dim)
        )
        self.bond_right = nn.Parameter(
            torch.empty(len(self.type_to_idx), self.rank, self.bond_dim)
        )
        return super().load_state_dict(state_dict, strict=strict)
