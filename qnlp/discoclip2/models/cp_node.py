import torch
from torch import nn


class CPQuadRankLayer(nn.Module):
    """
    Quadtree Layer: Merges 4 children (2x2 block) into 1 parent.
    """

    def __init__(self, num_nodes, in_dim, out_dim, rank, dropout_p=0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.dropout_p = dropout_p

        # ADDED: Pre-Norm for gradient stability
        self.norm = nn.LayerNorm(in_dim)

        self.factor_tl = nn.Parameter(torch.empty(num_nodes, rank, in_dim))
        self.factor_tr = nn.Parameter(torch.empty(num_nodes, rank, in_dim))
        self.factor_bl = nn.Parameter(torch.empty(num_nodes, rank, in_dim))
        self.factor_br = nn.Parameter(torch.empty(num_nodes, rank, in_dim))

        self.factor_out = nn.Parameter(torch.empty(num_nodes, rank, out_dim))
        self.scale = nn.Parameter(torch.ones(num_nodes, rank) * (1.0 / rank))

        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.res_proj = nn.Identity()

        self._initialize()

    def _initialize(self):
        with torch.no_grad():
            for f in [self.factor_tl, self.factor_tr, self.factor_bl, self.factor_br]:
                nn.init.xavier_uniform_(f)
            nn.init.xavier_uniform_(self.factor_out)

    def forward(self, x):
        res = self.res_proj(x.mean(dim=2))

        x = self.norm(x)

        x_tl = x[:, :, 0, :]
        x_tr = x[:, :, 1, :]
        x_bl = x[:, :, 2, :]
        x_br = x[:, :, 3, :]

        p_tl = torch.einsum("bni,nri->bnr", x_tl, self.factor_tl)
        p_tr = torch.einsum("bni,nri->bnr", x_tr, self.factor_tr)
        p_bl = torch.einsum("bni,nri->bnr", x_bl, self.factor_bl)
        p_br = torch.einsum("bni,nri->bnr", x_br, self.factor_br)

        merged = self.scale.unsqueeze(0) * p_tl * p_tr * p_bl * p_br

        if self.training and self.dropout_p > 0:
            mask = torch.bernoulli(torch.full_like(merged, 1 - self.dropout_p))
            merged = merged * mask / (1 - self.dropout_p)

        out = torch.einsum("bnr,nro->bno", merged, self.factor_out)

        return out + res
