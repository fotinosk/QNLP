"""
NODE_ARCHITECTURE_PLAN.md's node candidates NODE-1, NODE-2, NODE-4.

Each replaces `CPQuadRankLayer` (qnlp/discoviz/models/cp_node.py) as the
per-node computation inside `TTNImageModel`'s quadtree, keeping the same
forward interface: input `[batch, nodes, 4_children, in_dim]`, output
`[batch, nodes, out_dim]`. Residual, gain, dropout, and Route N's gated
non-linearity are identical across all three and to the original CP node
(factored into `_QuadNodeBase`); only how `merged` is computed from the
four RMS-normalised child projections differs.

NODE-3 (isometry maintained during training) needs no new class: it is
`CPQuadRankLayer(tied=True, ...)` with `torch.nn.utils.parametrizations
.orthogonal` applied to its five factor tensors after construction — see
`apply_isometric_parametrization` at the bottom of this file.
"""

import torch
from torch import nn


def _expand_leading(t: torch.Tensor, num_nodes: int, tied: bool) -> torch.Tensor:
    """Broadcast a tied (leading-dim-1) parameter to num_nodes for einsum,
    which -- unlike plain multiplication -- does not numpy-broadcast a
    size-1 named dim automatically. `.expand` is a view; autograd sums
    gradients back into the size-1 parameter."""
    if not tied:
        return t
    return t.expand(num_nodes, *([-1] * (t.ndim - 1)))


class _QuadNodeBase(nn.Module):
    """Shared scaffolding: dropout, output projection, Route N gate,
    residual. Subclasses implement `_compute_merged` and set `self.rank_out`
    (the dimension `merged` has, consumed by `factor_out`)."""

    def __init__(
        self,
        num_nodes,
        in_dim,
        out_dim,
        rank,
        dropout_p=0.0,
        use_residual=True,
        gain_factor=1.0,
        use_isometric_init=True,
        nonlinearity="none",
        tied=False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.dropout_p = dropout_p
        self.use_residual = use_residual
        self.use_isometric_init = use_isometric_init
        self.nonlinearity = nonlinearity
        self.tied = tied
        self._param_nodes = 1 if tied else num_nodes
        if nonlinearity != "none":
            self.gate = nn.Parameter(torch.zeros(1))
        self.gain = nn.Parameter(torch.full((self._param_nodes, 1), gain_factor))
        if use_residual:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def _rms_norm(self, t, eps=1e-6):
        rms = torch.sqrt(torch.mean(t**2, dim=-1, keepdim=True) + eps)
        return t / rms

    def _expand(self, t):
        return _expand_leading(t, self.num_nodes, self.tied)

    def _orthogonal_init(self, *factors):
        with torch.no_grad():
            if self.use_isometric_init:
                for f in factors:
                    for i in range(f.shape[0]):
                        nn.init.orthogonal_(f[i])
            else:
                for f in factors:
                    nn.init.orthogonal_(f)

    def _compute_merged(self, x) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x):
        merged = self._compute_merged(x)
        self._last_merged = merged.detach()  # NODE_ARCHITECTURE_PLAN.md row 0/4's kurtosis measurement

        if self.training and self.dropout_p > 0:
            merged = nn.functional.dropout(merged, p=self.dropout_p)

        out = torch.einsum("bnr, nro -> bno", merged, self._expand(self.factor_out))

        if self.nonlinearity == "born":
            out = out + self.gate * (out * out)
        elif self.nonlinearity == "gelu":
            out = out + self.gate * nn.functional.gelu(out)

        if self.use_residual:
            return out + self.res_proj(x.mean(dim=2))
        return out


class PairwiseBinaryNode(_QuadNodeBase):
    """NODE-1: the standard TTN construction. Two nested binary
    contractions with an intermediate RMS-normalisation, instead of a
    single 4-way CP product:

        a   = RMSNorm( (A_tl x_tl) * (A_tr x_tr) )
        b   = RMSNorm( (A_bl x_bl) * (A_br x_br) )
        out = V( (B_a a) * (B_b b) )
    """

    def __init__(self, num_nodes, in_dim, out_dim, rank, **kwargs):
        super().__init__(num_nodes, in_dim, out_dim, rank, **kwargs)
        n = self._param_nodes
        self.factor_tl = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_tr = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_bl = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_br = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_ba = nn.Parameter(torch.empty(n, rank, rank))  # second-stage: a -> rank
        self.factor_bb = nn.Parameter(torch.empty(n, rank, rank))  # second-stage: b -> rank
        self.factor_out = nn.Parameter(torch.empty(n, rank, out_dim))
        self._orthogonal_init(
            self.factor_tl,
            self.factor_tr,
            self.factor_bl,
            self.factor_br,
            self.factor_ba,
            self.factor_bb,
            self.factor_out,
        )

    def _compute_merged(self, x):
        p_tl = torch.einsum("bni, nri -> bnr", x[:, :, 0, :], self._expand(self.factor_tl))
        p_tr = torch.einsum("bni, nri -> bnr", x[:, :, 1, :], self._expand(self.factor_tr))
        p_bl = torch.einsum("bni, nri -> bnr", x[:, :, 2, :], self._expand(self.factor_bl))
        p_br = torch.einsum("bni, nri -> bnr", x[:, :, 3, :], self._expand(self.factor_br))

        a = self._rms_norm(p_tl * p_tr)
        b = self._rms_norm(p_bl * p_br)

        a2 = torch.einsum("bnr, nsr -> bns", a, self._expand(self.factor_ba))
        b2 = torch.einsum("bnr, nsr -> bns", b, self._expand(self.factor_bb))

        merged = a2 * b2
        merged = merged * self.gain.unsqueeze(0)
        return merged


class DegreeReducedNode(_QuadNodeBase):
    """NODE-2: replace the degree-4 product with a sum over the six
    pairwise products of the four RMS-normalised child projections
    (degree 2), each with a learnable scalar weight. Gated on
    NODE_ARCHITECTURE_PLAN.md's kurtosis measurement showing `merged` is
    heavy-tailed (it is: excess kurtosis 163-9301 across layers on the
    trained checkpoint)."""

    def __init__(self, num_nodes, in_dim, out_dim, rank, **kwargs):
        super().__init__(num_nodes, in_dim, out_dim, rank, **kwargs)
        n = self._param_nodes
        self.factor_tl = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_tr = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_bl = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_br = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_out = nn.Parameter(torch.empty(n, rank, out_dim))
        self.pair_weights = nn.Parameter(torch.ones(n, 6))
        self._orthogonal_init(self.factor_tl, self.factor_tr, self.factor_bl, self.factor_br, self.factor_out)

    def _compute_merged(self, x):
        p_tl = torch.einsum("bni, nri -> bnr", x[:, :, 0, :], self._expand(self.factor_tl))
        p_tr = torch.einsum("bni, nri -> bnr", x[:, :, 1, :], self._expand(self.factor_tr))
        p_bl = torch.einsum("bni, nri -> bnr", x[:, :, 2, :], self._expand(self.factor_bl))
        p_br = torch.einsum("bni, nri -> bnr", x[:, :, 3, :], self._expand(self.factor_br))
        p_tl, p_tr, p_bl, p_br = map(self._rms_norm, [p_tl, p_tr, p_bl, p_br])

        w = self._expand(self.pair_weights.unsqueeze(1)).squeeze(1)  # [num_nodes, 6]
        pairs = [p_tl * p_tr, p_tl * p_bl, p_tl * p_br, p_tr * p_bl, p_tr * p_br, p_bl * p_br]
        merged = sum(w[:, i].unsqueeze(0).unsqueeze(-1) * pairs[i] for i in range(6))
        merged = merged * self.gain.unsqueeze(0)
        return merged


class TuckerNode(_QuadNodeBase):
    """NODE-4: a full Tucker core instead of CP's diagonal core, allowing
    cross-terms between rank components that CP structurally forbids.
    Requires node-tying (the core has rank^5 entries; only affordable
    shared across a whole level) -- raises if tied=False."""

    def __init__(self, num_nodes, in_dim, out_dim, rank, **kwargs):
        super().__init__(num_nodes, in_dim, out_dim, rank, **kwargs)
        if not self.tied:
            raise ValueError("TuckerNode (NODE-4) requires tied=True -- the core has rank^5 entries per node.")
        n = self._param_nodes
        self.factor_tl = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_tr = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_bl = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_br = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_out = nn.Parameter(torch.empty(n, rank, out_dim))
        self.core = nn.Parameter(torch.empty(n, rank, rank, rank, rank, rank))
        self._orthogonal_init(self.factor_tl, self.factor_tr, self.factor_bl, self.factor_br, self.factor_out)
        # A CP-equivalent init for the core (diagonal in the first four
        # indices, identity on the output index) keeps the layer close to
        # today's CP node at init rather than starting from unstructured
        # noise; scaled by rank**-1.5 to keep merged's initial RMS ~ CP's.
        with torch.no_grad():
            core = torch.zeros(n, rank, rank, rank, rank, rank)
            diag = torch.arange(rank)
            core[:, diag, diag, diag, diag, diag] = rank**-1.5
            self.core.copy_(core)

    def _compute_merged(self, x):
        p_tl = torch.einsum("bni, nri -> bnr", x[:, :, 0, :], self._expand(self.factor_tl))
        p_tr = torch.einsum("bni, nri -> bnr", x[:, :, 1, :], self._expand(self.factor_tr))
        p_bl = torch.einsum("bni, nri -> bnr", x[:, :, 2, :], self._expand(self.factor_bl))
        p_br = torch.einsum("bni, nri -> bnr", x[:, :, 3, :], self._expand(self.factor_br))
        p_tl, p_tr, p_bl, p_br = map(self._rms_norm, [p_tl, p_tr, p_bl, p_br])

        core = _expand_leading(self.core, self.num_nodes, self.tied)
        # Distinct letters for batch (z) vs the four rank indices (a-d) --
        # reusing 'b' for both batch and a rank index is a silent bug here,
        # since einsum has no concept of "this letter means two different
        # things in different operands".
        merged = torch.einsum("zna,znb,znc,znd,nabcde->zne", p_tl, p_tr, p_bl, p_br, core)
        merged = merged * self.gain.unsqueeze(0)
        return merged


def apply_isometric_parametrization(layer: nn.Module) -> nn.Module:
    """NODE-3: keep the node's projections on the Stiefel manifold
    throughout training (canonical-form TTN), not just at init.
    `layer` must be a tied CPQuadRankLayer (its factors are then 2D-batch
    tensors of shape [1, rank, dim], which orthogonal parametrisation
    treats as a single 2D matrix)."""
    if not getattr(layer, "tied", False):
        raise ValueError("NODE-3 (isometry maintained) requires a tied layer -- see NODE_ARCHITECTURE_PLAN.md.")
    for name in ("factor_tl", "factor_tr", "factor_bl", "factor_br", "factor_out"):
        torch.nn.utils.parametrizations.orthogonal(layer, name=name)
    return layer
