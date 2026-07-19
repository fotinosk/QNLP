import string
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
    ~O(1)."""

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
        # Cache of pre-compiled ContractExpression graphs for linear-mode contraction,
        # shared by the per-sample and batched-same-topology paths (their expr strings
        # never collide, since the batched one has an extra leading batch index).
        self._expression_cache: dict[tuple, Any] = {}
        # Cache of "ab,bc->ac" -> "zab,zbc->zac" batched-expr rewrites, keyed by the
        # plain (unbatched) einsum expression.
        self._batched_expr_cache: dict[str, str] = {}
        # Diagnostic counters: how many forward() calls used the batched
        # same-topology fast path vs. the per-sample fallback, and how many
        # individual samples went through a grouped (batched) contraction vs.
        # a one-sample sequential contraction.
        self.fast_path_batches = 0
        self.fallback_batches = 0
        self.fast_path_samples = 0
        self.fallback_samples = 0
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

    def _rescale_single(self, t: torch.Tensor, sym: Symbol) -> torch.Tensor:
        """Rescale one linear-mode symbol tensor to its expected init Frobenius
        norm before the contraction. The raw parameter is never mutated — this
        is a functional reparameterisation in the forward graph, so gradients
        flow through it. Restores the O(1)-typical-magnitude calibration
        `_init_uniform_bound` was tuned for (unlike pinning every symbol to an
        arbitrary constant, which either overflows or underflows depending on
        chain length). The contraction itself runs in float64 (see
        `_forward_single`/`_forward_batch_same_topology`) so the wider dynamic
        range absorbs residual drift from weight growth or correlated
        (non-random) directions during training, without touching the
        opt_einsum path or algorithm. Because the contraction is multilinear
        and the output is finally L2-normalised, per-symbol scale is a gauge
        freedom — this leaves the normalised embedding direction unaffected."""
        norm = torch.linalg.vector_norm(t)
        return self._target_norm(sym) * t / (norm + 1e-8)

    def _rescale_batched(self, stacked: torch.Tensor, syms: List[Symbol]) -> torch.Tensor:
        """Same rescale as `_rescale_single`, but for a [B, *shape] stack where
        each row may be a DIFFERENT symbol/word — so each row gets its OWN
        target norm, not a single shared one."""
        dims = tuple(range(1, stacked.ndim))
        norm = torch.linalg.vector_norm(stacked, ord=2, dim=dims, keepdim=True)
        target = torch.tensor([self._target_norm(s) for s in syms], device=stacked.device, dtype=stacked.dtype)
        target = target.view(-1, *([1] * (stacked.ndim - 1)))
        return target * stacked / (norm + 1e-8)

    def _batched_expr(self, einsum_expr: str) -> str:
        """Rewrite "ab,bc->ac" into "zab,zbc->zac" (add a shared leading batch
        index to every operand and the output), picking a letter not already
        used anywhere in the expression."""
        cached = self._batched_expr_cache.get(einsum_expr)
        if cached is not None:
            return cached

        lhs, rhs = einsum_expr.split("->")
        operands = lhs.split(",")
        used = {c for c in einsum_expr if c.isalpha()}
        batch_letter = next(c for c in string.ascii_letters if c not in used)
        batched = f"{','.join(batch_letter + op for op in operands)}->{batch_letter}{rhs}"
        self._batched_expr_cache[einsum_expr] = batched
        return batched

    def _forward_batch_same_topology(self, inputs: list[tuple]) -> torch.Tensor:
        """Batched fast path: every input shares the same einsum expression
        (diagram), so this runs ONE opt_einsum call across the whole batch
        instead of one call per sample. Linear mode only — see forward().

        Relies on an assumption verified offline against the actual training
        data (not just assumed): for any two samples sharing the same
        einsum_expr, symbols[i] has the same shape at every position i (see
        qnlp/domain/datasets/topology_bucket_sampler.py). Re-checked here
        defensively — any violation falls back to the per-sample path rather
        than risk silently contracting mismatched legs together.
        """
        einsum_expr = inputs[0][0]
        symbols_per_sample = [inp[1] for inp in inputs]
        n_positions = len(symbols_per_sample[0])

        if any(len(syms) != n_positions for syms in symbols_per_sample):
            return torch.stack([self._forward_single(inp) for inp in inputs])

        orig_dtype = self.sym2weight[symbols_per_sample[0][0]].dtype
        stacked_tensors = []
        for i in range(n_positions):
            column = [self.sym2weight[syms[i]] for syms in symbols_per_sample]
            ref_shape = column[0].shape
            if any(t.shape != ref_shape for t in column[1:]):
                return torch.stack([self._forward_single(inp) for inp in inputs])
            syms_at_position = [syms[i] for syms in symbols_per_sample]
            stacked_tensors.append(self._rescale_batched(torch.stack(column, dim=0), syms_at_position))

        batched_expr = self._batched_expr(einsum_expr)
        shapes = tuple(t.shape for t in stacked_tensors)
        key = (batched_expr, shapes)
        expr_obj = self._expression_cache.get(key)
        if expr_obj is None:
            import opt_einsum

            expr_obj = opt_einsum.contract_expression(batched_expr, *shapes)
            self._expression_cache[key] = expr_obj

        # Contract (and normalise) in float64: with per-symbol target-norm
        # scaling restored, per-tensor magnitudes are back to ~O(100)s, and a
        # chain of a dozen-plus such tensors can drift past float32's overflow
        # ceiling as training correlates weight directions. float64 gives ~270
        # orders of magnitude more headroom — same opt_einsum path/algorithm,
        # pure dtype change. Cast back to the model dtype only after
        # normalize, whose output is safely bounded in [-1, 1].
        x = expr_obj(*[t.double() for t in stacked_tensors])  # [B, out_dim]
        if x.ndim != 2:
            raise RuntimeError(f"Expected 2D batched output, got shape {tuple(x.shape)}\n  diagram: {einsum_expr}")
        return nn.functional.normalize(x, dim=-1, eps=1e-35).to(orig_dtype)

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
            # Weight-norm layer (linear mode only): rescale each input tensor to
            # its expected init Frobenius norm before the contraction — see
            # _rescale_single for why (restores the O(1)-typical-magnitude
            # calibration rather than an arbitrary constant). The contraction
            # runs in float64 (below) for headroom against overflow as training
            # correlates weight directions. (NLC mode is left untouched: its
            # gate is non-linear, so rescaling inputs would change the
            # represented function.)
            tensors = [self._rescale_single(t, sym) for sym, t in zip(symbols, tensors)]

        try:
            if self.non_linear_contractions:
                x = self.contractions_function(einsum_expr, tensors, path, gate=gate)
            else:
                shapes = tuple(t.shape for t in tensors)
                key = (einsum_expr, shapes)
                expr_obj = self._expression_cache.get(key)
                if expr_obj is None:
                    import opt_einsum

                    expr_obj = opt_einsum.contract_expression(einsum_expr, *shapes)
                    self._expression_cache[key] = expr_obj
                # float64 contraction — see _rescale_single docstring.
                x = expr_obj(*[t.double() for t in tensors])
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
        result = nn.functional.normalize(x, dim=-1, eps=1e-35)
        return result.to(tensors[0].dtype) if not self.non_linear_contractions else result

    def forward(self, inputs: List[tuple[str, List[Symbol]]]) -> torch.Tensor:
        # Fast path: linear mode, batch size > 1, and every sample shares the same
        # diagram (e.g. batches built by TopologyBucketSampler) — one contraction
        # call for the whole batch instead of one per sample. NLC mode always uses
        # the per-sample path (its gate is non-linear, so batching would change the
        # represented function).
        if self.non_linear_contractions or len(inputs) == 1:
            self.fallback_batches += 1
            self.fallback_samples += len(inputs)
            return torch.stack([self._forward_single(input) for input in inputs])

        groups: dict[str, list[int]] = {}
        for i, inp in enumerate(inputs):
            groups.setdefault(inp[0], []).append(i)

        if len(groups) == 1:
            self.fast_path_batches += 1
            self.fast_path_samples += len(inputs)
            return self._forward_batch_same_topology(inputs)

        # Heterogeneous batch (tail batches from TopologyBucketSampler, default
        # random batching, eval loaders, ...): contract per same-diagram GROUP
        # rather than per sample. One opt_einsum call per unique diagram in the
        # batch — per-call cost is roughly independent of group size (it's
        # dispatch-bound for these small tensors), so this cuts the dominant
        # per-sample dispatch cost by the mean group multiplicity. Samples whose
        # diagram is unique within the batch still take the single-sample path.
        self.fallback_batches += 1
        out: list[torch.Tensor | None] = [None] * len(inputs)
        for idxs in groups.values():
            if len(idxs) == 1:
                self.fallback_samples += 1
                out[idxs[0]] = self._forward_single(inputs[idxs[0]])
            else:
                self.fast_path_samples += len(idxs)
                embeddings = self._forward_batch_same_topology([inputs[i] for i in idxs])
                for row, i in enumerate(idxs):
                    out[i] = embeddings[row]
        return torch.stack(out)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        base = super().state_dict(*args, **kwargs)
        base["symbols_list"] = self.symbols
        base["sizes_list"] = self.sizes
        base["non_linear_contractions"] = self.non_linear_contractions
        return base

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        # Preserve whatever device (and dtype) the model was already on — the
        # weights are about to be replaced wholesale below, and torch.empty()
        # defaults to CPU, which would silently strand the model on CPU on
        # every reload regardless of an earlier .to(device) call.
        device = self.weights[0].device if len(self.weights) > 0 else torch.device("cpu")
        dtype = self.weights[0].dtype if len(self.weights) > 0 else torch.get_default_dtype()

        if "symbols_list" in state_dict:
            loaded_symbols = state_dict.pop("symbols_list")
            self.symbols = list(loaded_symbols)
        if "sizes_list" in state_dict:
            loaded_sizes = state_dict.pop("sizes_list")
            self.sizes = list(loaded_sizes)
        if "non_linear_contractions" in state_dict:
            self.non_linear_contractions = state_dict.pop("non_linear_contractions")
            self._setup_contractions_function()

        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(size, device=device, dtype=dtype)) for size in self.sizes]
        )

        self.sym2weight = self.compute_sym2weight()
        self._path_cache = {}
        return super().load_state_dict(state_dict, strict=strict)
