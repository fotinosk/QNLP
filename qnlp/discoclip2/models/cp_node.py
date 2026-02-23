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

        self.factors = nn.Parameter(torch.empty(4, num_nodes, rank, in_dim))
        self.factor_out = nn.Parameter(torch.empty(num_nodes, rank, out_dim))

        # Learnable gain per node to replace static scaling
        self.gain = nn.Parameter(torch.full((num_nodes, 1), gain_factor))

        if use_residual:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

        with torch.no_grad():
            nn.init.orthogonal_(self.factors)  # PyTorch handles this across the 1st dim
            nn.init.orthogonal_(self.factor_out)

    def _rms_norm(self, t, eps=1e-6):
        # Normalizes across the Bond/Rank dimension to keep energy at 1.0
        rms = torch.sqrt(torch.mean(t**2, dim=-1, keepdim=True) + eps)
        return t / rms

    def forward(self, x):
        projected = torch.einsum("bnci, cnri -> bncr", x, self.factors)
        projected = self._rms_norm(projected)
        p_tl, p_tr, p_bl, p_br = projected.unbind(dim=2)
        merged = p_tl * p_tr * p_bl * p_br
        merged = merged * self.gain.unsqueeze(0)
        if self.training and self.dropout_p > 0:
            merged = nn.functional.dropout(merged, p=self.dropout_p)

        out = torch.einsum("bnr, nro -> bno", merged, self.factor_out)

        # 5. Residual (Optional for early layers)
        if self.use_residual:
            return out + self.res_proj(x.mean(dim=2))
        return out
