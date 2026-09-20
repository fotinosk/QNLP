import torch
from torch import nn


class CPQuadRankLayer(nn.Module):
    """
    Optimized Quadtree Layer with Internal Factor Normalization.
    """

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
        # NODE_ARCHITECTURE_PLAN.md's cross-cutting tying option: one shared
        # tensor per level instead of one per node. Implemented by storing
        # the factor's leading dim as 1 (not num_nodes) and `.expand`-ing it
        # back to num_nodes at forward time -- a broadcasting view, not a
        # copy, so the einsums below are identical either way and autograd
        # naturally sums gradients back into the size-1 parameter.
        self.tied = tied
        self._param_nodes = 1 if tied else num_nodes
        # TTN_CIFAR_EXPERIMENTS.md Route N: measures what the multilinear
        # constraint costs, mirroring the text tower's NLC exactly. gate
        # inits at EXACTLY 0.0 (not text tower's 0.1-floor-clamped variant)
        # so the model is bit-for-bit today's multilinear one at init --
        # the gate's learned trajectory IS the measurement of how much
        # non-linearity this task demands.
        self.nonlinearity = nonlinearity
        if nonlinearity != "none":
            self.gate = nn.Parameter(torch.zeros(1))

        # Factor weights: [Nodes (or 1 if tied), Rank, Input_Dim]
        n = self._param_nodes
        self.factor_tl = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_tr = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_bl = nn.Parameter(torch.empty(n, rank, in_dim))
        self.factor_br = nn.Parameter(torch.empty(n, rank, in_dim))

        self.factor_out = nn.Parameter(torch.empty(n, rank, out_dim))

        # Learnable gain per node (or shared, if tied) to replace static scaling
        self.gain = nn.Parameter(torch.full((n, 1), gain_factor))

        if use_residual:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

        self._initialize()

    def _initialize(self):
        # nn.init.orthogonal_ on a 3D [num_nodes, rank, in_dim] tensor flattens
        # dims [1:], so it orthogonalises NODES against each other (each node's
        # whole [rank, in_dim] block becomes one unit-norm "row") rather than
        # giving each node its own semi-orthogonal [rank, in_dim] matrix. That
        # leaves every node a heavily down-scaled random projection (Frobenius
        # norm 1 over the whole block, not per-row) instead of a canonical-form
        # TTN tensor. Fix (Stage A1, default on): orthogonalise each node's own
        # matrix individually — the standard TN-canonical (isometric tensor)
        # prescription. use_isometric_init=False reproduces the original
        # (verified-defective) behaviour, kept only so Stage A1 can be
        # ablated against A1+B1 with everything else held fixed — see
        # TTN_CIFAR_EXPERIMENTS.md's parallel batch plan, row 2.
        with torch.no_grad():
            factors = [self.factor_tl, self.factor_tr, self.factor_bl, self.factor_br, self.factor_out]
            if self.use_isometric_init:
                for f in factors:
                    for i in range(f.shape[0]):
                        nn.init.orthogonal_(f[i])
            else:
                for f in factors:
                    nn.init.orthogonal_(f)

    def _rms_norm(self, t, eps=1e-6):
        # Normalizes across the Bond/Rank dimension to keep energy at 1.0
        rms = torch.sqrt(torch.mean(t**2, dim=-1, keepdim=True) + eps)
        return t / rms

    def _expand(self, factor):
        # torch.einsum does not numpy-broadcast a size-1 named dim against a
        # larger one, so a tied factor's leading dim must be expanded to
        # num_nodes explicitly before the einsum. `.expand` is a view (no
        # copy); autograd sums gradients back into the size-1 parameter.
        return factor.expand(self.num_nodes, -1, -1) if self.tied else factor

    def forward(self, x):
        # x shape: [batch, nodes, 4_children, in_dim]

        # 1. Project to Rank space (Internal Legs)
        p_tl = torch.einsum("bni, nri -> bnr", x[:, :, 0, :], self._expand(self.factor_tl))
        p_tr = torch.einsum("bni, nri -> bnr", x[:, :, 1, :], self._expand(self.factor_tr))
        p_bl = torch.einsum("bni, nri -> bnr", x[:, :, 2, :], self._expand(self.factor_bl))
        p_br = torch.einsum("bni, nri -> bnr", x[:, :, 3, :], self._expand(self.factor_br))

        # 2. FIX #2: INTERNAL FACTOR RMS NORM
        # Prevents the "Vanishing Product" between layers
        p_tl, p_tr, p_bl, p_br = map(self._rms_norm, [p_tl, p_tr, p_bl, p_br])

        # 3. Multilinear Product with Gain
        merged = p_tl * p_tr * p_bl * p_br
        # Plain multiplication (not einsum) DOES numpy-broadcast a size-1
        # dim automatically, so a tied gain ([1, 1]) needs no explicit expand.
        merged = merged * self.gain.unsqueeze(0)
        # NODE_ARCHITECTURE_PLAN.md's prerequisite measurement: excess
        # kurtosis of this 4-way product, per layer, on a trained
        # checkpoint. Cheap capture (detached, no grad-graph cost); read by
        # qnlp/discoviz/diagnostic/node_kurtosis.py after a forward pass.
        self._last_merged = merged.detach()

        # 4. Dropout and Output Projection
        if self.training and self.dropout_p > 0:
            merged = nn.functional.dropout(merged, p=self.dropout_p)

        out = torch.einsum("bnr, nro -> bno", merged, self._expand(self.factor_out))

        # 4b. Route N: gated non-linearity, applied before the residual add
        # (mirrors the text tower's NLC placement). At gate=0 this is exactly
        # a no-op regardless of which _f is selected.
        if self.nonlinearity == "born":
            out = out + self.gate * (out * out)
        elif self.nonlinearity == "gelu":
            out = out + self.gate * nn.functional.gelu(out)

        # 5. Residual (Optional for early layers)
        if self.use_residual:
            return out + self.res_proj(x.mean(dim=2))
        return out
