from functools import partial
from typing import Any, Dict, List

import torch
import torch.nn as nn
from lambeq import Symbol
from opt_einsum import contract as einsum

from qnlp.core.non_linear_contraction.atom import IntermediateTooLargeError
from qnlp.core.non_linear_contraction.einsum_interface import contract_einsum_non_linearly

torch.serialization.add_safe_globals([Symbol])


def _init_uniform_bound(directed_cod: int) -> float:
    """Half-width of the uniform init for a tensor whose output leg has size
    `directed_cod`. Tuned (empirically) so a fresh multilinear contraction stays
    ~O(1). Also used to derive the per-symbol target norm for the linear-mode
    weight-norm layer."""

    def mean(size: int) -> float:
        if size < 6:
            correction_factor = [0, 3, 2.6, 2, 1.6, 1.3][size]
        else:
            correction_factor = 1 / (0.16 * size - 0.04)
        return (size / 3 - 1 / (15 - correction_factor)) ** 0.5

    return 1 / mean(directed_cod)


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
            # Global scalar residual gate, clamped to ≥ 0.1 in the forward pass so
            # the model always applies some non-linearity. Init at the floor so
            # gradients flow from the first step.
            self.nonlinear_gate = nn.Parameter(torch.tensor(0.1))

        self._setup_contractions_function()
        self.reset_parameters()
        self.sym2weight = self.compute_sym2weight()
        self._path_cache: dict[tuple, list | None] = {}
        # Cache of per-symbol target Frobenius norms for the linear-mode weight-norm.
        self._weight_scale_cache: dict = {}

    def _setup_contractions_function(self):
        if self.non_linear_contractions:
            self.contractions_function = partial(contract_einsum_non_linearly, non_linear_fn=nn.functional.gelu)
        else:
            self.contractions_function = lambda expr, tensors, path=None, gate=None: einsum(expr, *tensors)

    def compute_sym2weight(self) -> Dict[Symbol, nn.Parameter]:
        return {sym: weight for sym, weight in zip(self.symbols, self.weights)}

    def reset_parameters(self, symbols: List[Symbol] = None):
        for sym, weight in zip(self.symbols, self.weights):
            if symbols is not None and sym not in symbols:
                continue
            bound = _init_uniform_bound(sym.directed_cod)
            nn.init.uniform_(weight, -bound, bound)

    def _target_norm(self, sym: Symbol) -> float:
        """Frobenius norm the init would give this symbol's tensor.

        For uniform(-b, b) the per-element std is b/sqrt(3), so the expected
        Frobenius norm is sqrt(numel) * b/sqrt(3). Cached (depends only on shape
        and directed_cod, which are fixed per symbol)."""
        cached = self._weight_scale_cache.get(sym)
        if cached is None:
            w = self.sym2weight[sym]
            std = _init_uniform_bound(sym.directed_cod) / (3.0**0.5)
            cached = (w.numel() ** 0.5) * std
            self._weight_scale_cache[sym] = cached
        return cached

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

    def _get_path(self, einsum_expr: str, tensors: list[torch.Tensor]) -> list | None:
        """Recompute a contraction path from actual tensor shapes via opt_einsum branch-2.

        Only called when no stored path is available (e.g. cluster training where
        bond_dim at training time exceeds bond_dim at dataset-creation time).
        """
        key = (einsum_expr, tuple(tuple(t.shape) for t in tensors))
        if key not in self._path_cache:
            from qnlp.core.non_linear_contraction.determine_optimal_contraction_path import (
                MAX_INTERMEDIATE_ELEMENTS,
                get_contraction_path_and_cost,
            )

            try:
                path, largest = get_contraction_path_and_cost(einsum_expr, tuple(tuple(t.shape) for t in tensors))
                self._path_cache[key] = path if largest <= MAX_INTERMEDIATE_ELEMENTS else None
            except Exception:
                self._path_cache[key] = None
        return self._path_cache[key]

    def _forward_single(self, input: tuple) -> torch.Tensor:
        # input = (einsum_expr, symbols) for linear, or (einsum_expr, symbols, path) for NLC.
        einsum_expr, symbols = input[0], input[1]
        stored_path = input[2] if len(input) > 2 else None

        tensors = [self.sym2weight[sym] for sym in symbols]
        gate = self.nonlinear_gate.clamp(min=0.1) if self.non_linear_contractions else None

        if self.non_linear_contractions:
            # Prefer the stored path from the dataset — it was verified feasible at
            # creation time and at training shapes it can only be cheaper (all dims
            # are remapped to ≤ their creation values locally). If no stored path is
            # present (rare: sample was added without path computation, or cluster
            # run where bond_dim increased), fall back to recomputing.
            if stored_path is not None:
                path = stored_path
            else:
                path = self._get_path(einsum_expr, tensors)
            if path is None:
                # Infeasible: skip without attempting the contraction to avoid OOM.
                output_idx = einsum_expr.split("->")[1]
                size_map = {
                    c: dim
                    for repr_, t in zip(einsum_expr.split("->")[0].split(","), tensors)
                    for c, dim in zip(repr_, t.shape)
                }
                out_dim = size_map[output_idx[0]] if output_idx else 1
                return torch.full((out_dim,), float("nan"), device=tensors[0].device, dtype=tensors[0].dtype)
        else:
            path = None
            # Weight-norm layer (linear mode only): rescale each input tensor to its
            # init Frobenius norm before the contraction. The raw parameters are never
            # mutated — this is a functional reparameterisation in the forward graph,
            # so gradients flow through it. Because the contraction is multilinear and
            # the output is finally normalised, per-symbol scale is a gauge freedom:
            # this leaves the normalised embedding direction identical while pinning
            # magnitudes so the contraction can no longer drift into float overflow.
            # (NLC mode is left untouched: its gate is non-linear, so rescaling inputs
            # would change the represented function.)
            tensors = [self._target_norm(sym) * t / (t.norm() + 1e-8) for sym, t in zip(symbols, tensors)]

        try:
            x = self.contractions_function(einsum_expr, tensors, path, gate=gate)
        except IntermediateTooLargeError:
            output_idx = einsum_expr.split("->")[1]
            size_map = {
                c: dim
                for repr_, t in zip(einsum_expr.split("->")[0].split(","), tensors)
                for c, dim in zip(repr_, t.shape)
            }
            out_dim = size_map[output_idx[0]] if output_idx else 1
            return torch.full((out_dim,), float("nan"), device=tensors[0].device, dtype=tensors[0].dtype)
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
        self._path_cache = {}
        return super().load_state_dict(state_dict, strict=strict)

    def _get_chain_pieces(self, word: str, symbols: List[Symbol]) -> List[Symbol] | None:
        """Find the ordered MPS chain pieces for `word` within one row's own
        (already-resolved) symbol list. `CustomMPSAnsatz._split_ar`
        (qnlp/discoviz/parser/asnsatz.py) names pieces f"{ar.name}_{i}", so
        the base word is the token before the first underscore and the piece
        index is the token right after it. Returns None if `word` doesn't
        appear in `symbols` (Route A's role-resolve-rate gate — see
        TTN_CIFAR_EXPERIMENTS.md — is < 100%, so callers must handle this)."""
        pieces = []
        for sym in symbols:
            name = sym.name
            if "_" not in name:
                continue
            base, rest = name.split("_", 1)
            if base != word:
                continue
            idx_str = rest.split("_", 1)[0]
            if not idx_str.isdigit():
                continue
            pieces.append((int(idx_str), sym))
        if not pieces:
            return None
        pieces.sort(key=lambda p: p[0])
        return [sym for _, sym in pieces]

    def get_role_tensor(self, word: str, symbols: List[Symbol]) -> torch.Tensor | None:
        """Contract `word`'s local MPS chain into a single tensor with only
        its semantic legs — the bond legs joining consecutive chain pieces
        are internal to this one word and always fully contracted regardless
        of the rest of the sentence (TTN_CIFAR_EXPERIMENTS.md, Phase 0
        findings). For a single-piece word (e.g. a bare noun) this is just
        its raw parameter tensor. Route A's subject/object vectors use this
        directly; the verb uses `get_verb_chain` instead (see its docstring
        for why). Returns None if `word` isn't in `symbols`."""
        pieces = self._get_chain_pieces(word, symbols)
        if pieces is None:
            return None
        tensors = [self.sym2weight[sym] for sym in pieces]
        result = tensors[0]
        for t in tensors[1:]:
            result = torch.tensordot(result, t, dims=1)
        return result

    def get_verb_chain(self, word: str, symbols: List[Symbol]) -> List[torch.Tensor] | None:
        """Raw (uncontracted) MPS chain pieces for `word`. Route A's
        role-grounded score head needs to project the chain's two outer
        semantic legs into region-space *before* the internal bond-dim
        contraction — multilinear, so the order doesn't change the result,
        but avoids ever materialising the fully dense verb tensor (e.g.
        embedding_dim^3 elements for a 3-piece chain — infeasible at batch
        scale; see TTN_CIFAR_EXPERIMENTS.md's Route A implementation notes).
        Returns None if `word` isn't in `symbols`."""
        pieces = self._get_chain_pieces(word, symbols)
        if pieces is None:
            return None
        return [self.sym2weight[sym] for sym in pieces]

    def forward_roles(
        self, symbols_batch: List[List[Symbol]], roles_batch: List[tuple[str, str, str]]
    ) -> Dict[str, torch.Tensor]:
        """Per-row role-tensor extraction for Route A. `roles_batch[i]` is
        that row's (subj, verb, obj) words (already lowercased to match
        symbol naming). Subject/object become [B, noun_dim] tensors; the
        verb's chain is returned split into its three pieces (left/mid/right)
        rather than pre-contracted, since Route A's score head needs to
        project its outer legs before contracting (see `get_verb_chain`).

        v1 scope: only the mainline 3-piece transitive-verb chain is
        supported (the common SVO-Probes case). A row whose subject/object
        don't resolve, or whose verb chain isn't exactly 3 pieces, is marked
        invalid via a NaN sentinel on every returned tensor for that row —
        the same convention `ContrastiveVLM._safe_text_embed` uses — so
        callers can drop it with a standard isfinite check before the loss.
        """
        subj_list: list[torch.Tensor | None] = []
        obj_list: list[torch.Tensor | None] = []
        verb_chains: list[List[torch.Tensor] | None] = []
        for symbols, (subj_w, verb_w, obj_w) in zip(symbols_batch, roles_batch):
            s = self.get_role_tensor(subj_w, symbols)
            o = self.get_role_tensor(obj_w, symbols)
            v_chain = self.get_verb_chain(verb_w, symbols)
            ok = s is not None and o is not None and v_chain is not None and len(v_chain) == 3
            subj_list.append(s if ok else None)
            obj_list.append(o if ok else None)
            verb_chains.append(v_chain if ok else None)

        def _stack(tensors: list[torch.Tensor | None]) -> torch.Tensor:
            valid = [t for t in tensors if t is not None]
            if not valid:
                raise ValueError("No row in this batch resolved this role — check subj/verb/obj column lemmatisation.")
            shapes = [t.shape for t in valid]
            target_shape = max(set(shapes), key=shapes.count)
            ref = valid[0]
            out = [
                t if (t is not None and t.shape == target_shape) else torch.full(target_shape, float("nan"))
                for t in tensors
            ]
            out = [t.to(device=ref.device, dtype=ref.dtype) for t in out]
            return torch.stack(out)

        subj = _stack(subj_list)
        obj = _stack(obj_list)

        valid_chains = [c for c in verb_chains if c is not None]
        if not valid_chains:
            raise ValueError("No row in this batch had a resolvable 3-piece verb chain.")
        verb_pieces = []
        for pos in range(3):
            piece_tensors = [(chain[pos] if chain is not None else None) for chain in verb_chains]
            verb_pieces.append(_stack(piece_tensors))

        return {
            "subj": subj,
            "obj": obj,
            "verb_left": verb_pieces[0],
            "verb_mid": verb_pieces[1],
            "verb_right": verb_pieces[2],
        }
