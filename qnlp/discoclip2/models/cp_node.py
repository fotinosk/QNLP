import torch
from torch import nn


class CPQuadRankLayer(nn.Module):
    """
    Optimized Quadtree Layer with Internal Factor Normalization.
    """

    def __init__(self, num_nodes, in_dim, out_dim, rank, dropout_p=0.0, use_residual=True, gain_factor=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.dropout_p = dropout_p
        self.use_residual = use_residual

        # Factor weights: [Nodes, Rank, Input_Dim]
        self.factor_tl = nn.Parameter(torch.empty(num_nodes, rank, in_dim))
        self.factor_tr = nn.Parameter(torch.empty(num_nodes, rank, in_dim))
        self.factor_bl = nn.Parameter(torch.empty(num_nodes, rank, in_dim))
        self.factor_br = nn.Parameter(torch.empty(num_nodes, rank, in_dim))

        self.factor_out = nn.Parameter(torch.empty(num_nodes, rank, out_dim))

        # Learnable gain per node to replace static scaling
        self.gain = nn.Parameter(torch.full((num_nodes, 1), gain_factor))

        if use_residual:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

        self._initialize()

    def _initialize(self):
        with torch.no_grad():
            for f in [self.factor_tl, self.factor_tr, self.factor_bl, self.factor_br]:
                nn.init.orthogonal_(f)
            nn.init.orthogonal_(self.factor_out)

    def _rms_norm(self, t, eps=1e-6):
        # Normalizes across the Bond/Rank dimension to keep energy at 1.0
        rms = torch.sqrt(torch.mean(t**2, dim=-1, keepdim=True) + eps)
        return t / rms

    def forward(self, x):
        # x shape: [batch, nodes, 4_children, in_dim]

        # 1. Project to Rank space (Internal Legs)
        p_tl = torch.einsum("bni, nri -> bnr", x[:, :, 0, :], self.factor_tl)
        p_tr = torch.einsum("bni, nri -> bnr", x[:, :, 1, :], self.factor_tr)
        p_bl = torch.einsum("bni, nri -> bnr", x[:, :, 2, :], self.factor_bl)
        p_br = torch.einsum("bni, nri -> bnr", x[:, :, 3, :], self.factor_br)

        # 2. FIX #2: INTERNAL FACTOR RMS NORM
        # Prevents the "Vanishing Product" between layers
        p_tl, p_tr, p_bl, p_br = map(self._rms_norm, [p_tl, p_tr, p_bl, p_br])

        # 3. Multilinear Product with Gain
        prod_path = p_tl * p_tr * p_bl * p_br
        merged = prod_path * self.gain.unsqueeze(0)

        if self.training and self.dropout_p > 0:
            merged = nn.functional.dropout(merged, p=self.dropout_p)

        out = torch.einsum("bnr, nro -> bno", merged, self.factor_out)

        if self.use_residual:
            return out + self.res_proj(x.mean(dim=2))
        return out
