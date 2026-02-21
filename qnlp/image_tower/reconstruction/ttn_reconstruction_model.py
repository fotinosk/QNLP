import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
BATCH_SIZE = 256
LEARNING_RATE = 0.002
EPOCHS = 10
BOND_DIM = 64
CP_RANK = 16
DROPOUT = 0.0  # set to zero to make deterministic
PATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"Running on device: {DEVICE}")


class CPQuadUnfoldLayer(nn.Module):
    """
    Inverse Quadtree Layer: Expands 1 parent into 4 children (2x2 block).
    """

    def __init__(self, num_nodes, in_dim, out_dim, rank, dropout_p=0.0):
        super().__init__()
        self.num_nodes = num_nodes  # Number of parent nodes
        self.rank = rank

        # Factor to project input into the CP rank space
        self.factor_in = nn.Parameter(torch.randn(num_nodes, rank, in_dim))

        # 4 Factors to generate the 4 children
        self.factor_tl = nn.Parameter(torch.randn(num_nodes, rank, out_dim))
        self.factor_tr = nn.Parameter(torch.randn(num_nodes, rank, out_dim))
        self.factor_bl = nn.Parameter(torch.randn(num_nodes, rank, out_dim))
        self.factor_br = nn.Parameter(torch.randn(num_nodes, rank, out_dim))

        self.scale = nn.Parameter(torch.ones(num_nodes, rank))
        self._initialize()

    def _initialize(self):
        with torch.no_grad():
            for f in [
                self.factor_in,
                self.factor_tl,
                self.factor_tr,
                self.factor_bl,
                self.factor_br,
            ]:
                f.data /= f.data.norm(dim=-1, keepdim=True) + 1e-8
            self.scale.data.normal_(1.0, 0.02)

    def forward(self, x):
        # x shape: [Batch, Num_Nodes, In_Dim]

        # Project parent to rank space
        latents = torch.einsum("bni,nri->bnr", x, self.factor_in)
        latents = latents * self.scale.unsqueeze(0)

        # Generate 4 children
        out_tl = torch.einsum("bnr,nro->bno", latents, self.factor_tl)
        out_tr = torch.einsum("bnr,nro->bno", latents, self.factor_tr)
        out_bl = torch.einsum("bnr,nro->bno", latents, self.factor_bl)
        out_br = torch.einsum("bnr,nro->bno", latents, self.factor_br)

        # Stack into [Batch, Num_Nodes, 4, Out_Dim]
        # Residual: Add parent identity to children
        out = torch.stack([out_tl, out_tr, out_bl, out_br], dim=2)
        out = out + x.unsqueeze(2)

        return out


class CPQuadRankLayer(nn.Module):
    """
    Quadtree Layer: Merges 4 children (2x2 block) into 1 parent.
    """

    def __init__(self, num_nodes, in_dim, out_dim, rank, dropout_p=0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.dropout_p = dropout_p

        # We now have 4 input factors corresponding to the 2x2 spatial block
        self.factor_tl = nn.Parameter(torch.randn(num_nodes, rank, in_dim))  # Top-Left
        self.factor_tr = nn.Parameter(torch.randn(num_nodes, rank, in_dim))  # Top-Right
        self.factor_bl = nn.Parameter(torch.randn(num_nodes, rank, in_dim))  # Bottom-Left
        self.factor_br = nn.Parameter(torch.randn(num_nodes, rank, in_dim))  # Bottom-Right

        self.factor_out = nn.Parameter(torch.randn(num_nodes, rank, out_dim))
        self.scale = nn.Parameter(torch.ones(num_nodes, rank))

        self._initialize()

    def _initialize(self):
        with torch.no_grad():
            for f in [
                self.factor_tl,
                self.factor_tr,
                self.factor_bl,
                self.factor_br,
                self.factor_out,
            ]:
                f.data /= f.data.norm(dim=-1, keepdim=True) + 1e-8
            self.scale.data.normal_(1.0, 0.02)

    def forward(self, x):
        # x shape expected: [Batch, Num_Nodes, 4, In_Dim]
        # The 4 dimension corresponds to: 0:TL, 1:TR, 2:BL, 3:BR

        x_tl = x[:, :, 0, :]
        x_tr = x[:, :, 1, :]
        x_bl = x[:, :, 2, :]
        x_br = x[:, :, 3, :]

        # using 'bni' (batch, node, in) -> 'bnr' (batch, node, rank)
        p_tl = torch.einsum("bni,nri->bnr", x_tl, self.factor_tl)
        p_tr = torch.einsum("bni,nri->bnr", x_tr, self.factor_tr)
        p_bl = torch.einsum("bni,nri->bnr", x_bl, self.factor_bl)
        p_br = torch.einsum("bni,nri->bnr", x_br, self.factor_br)

        # This approximates a high-order tensor interaction
        merged = self.scale.unsqueeze(0) * p_tl * p_tr * p_bl * p_br

        if self.training and self.dropout_p > 0:
            mask = torch.bernoulli(torch.full_like(merged, 1 - self.dropout_p))
            merged = merged * mask / (1 - self.dropout_p)

        # 3. Project to Output
        out = torch.einsum("bnr,nro->bno", merged, self.factor_out)

        # 4. Residual Connection
        # Sum of all inputs added to output (Skip connection)
        out = out + x_tl + x_tr + x_bl + x_br

        return out


class QuadTreeEncoder(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=1,
        bond_dim=64,
        cp_rank=16,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_side = img_size // patch_size
        num_patches = self.num_patches_side**2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, bond_dim),
        )

        self.depth = int(math.log(num_patches, 4))
        self.layers = nn.ModuleList()
        current_nodes = num_patches // 4

        for _ in range(self.depth):
            self.layers.append(CPQuadRankLayer(current_nodes, bond_dim, bond_dim, cp_rank, dropout))
            current_nodes //= 4

    def forward(self, x):
        x = self.patch_embed(x)
        grid_dim = self.num_patches_side
        for layer in self.layers:
            x = rearrange(x, "b (h w) c -> b c h w", h=grid_dim)
            x = rearrange(x, "b c (h h2) (w w2) -> b (h w) (h2 w2) c", h2=2, w2=2)
            x = layer(x)
            grid_dim //= 2
        return x  # Shape: [Batch, 1, bond_dim]


class QuadTreeDecoder(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        out_channels=1,
        bond_dim=64,
        cp_rank=16,
        dropout_p=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_side = img_size // patch_size
        num_patches = self.num_patches_side**2
        self.depth = int(math.log(num_patches, 4))

        self.layers = nn.ModuleList()
        # Decoder works in reverse order of nodes
        # e.g., 1 -> 4 -> 16 -> 64
        current_nodes = 1
        for _ in range(self.depth):
            self.layers.append(CPQuadUnfoldLayer(current_nodes, bond_dim, bond_dim, cp_rank, dropout_p))
            current_nodes *= 4

        self.head = nn.Sequential(
            nn.Linear(bond_dim, out_channels * patch_size * patch_size),
            Rearrange(
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                h=self.num_patches_side,
                w=self.num_patches_side,
                p1=patch_size,
                p2=patch_size,
            ),
        )

    def forward(self, x):
        # x: [Batch, 1, bond_dim]
        grid_dim = 1
        for layer in self.layers:
            x = layer(x)  # [B, Nodes, 4, Dim]
            # Reshape 4-children back into a 2D grid sequence
            x = rearrange(
                x,
                "b (h w) (h2 w2) c -> b (h h2 w w2) c",
                h=grid_dim,
                w=grid_dim,
                h2=2,
                w2=2,
            )
            grid_dim *= 2

        return self.head(x)


class QuadTreeAutoencoder(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=1,
        bond_dim=64,
        cp_rank=16,
        dropout=0.0,
    ):
        super().__init__()
        self.encoder = QuadTreeEncoder(img_size, patch_size, in_channels, bond_dim, cp_rank, dropout)
        self.decoder = QuadTreeDecoder(img_size, patch_size, in_channels, bond_dim, cp_rank, dropout)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
