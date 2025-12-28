import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


BATCH_SIZE = 256
LEARNING_RATE = 0.002
EPOCHS = 10
BOND_DIM = 64          
CP_RANK = 16           
DROPOUT = 0.1          
PATCH_SIZE = 4         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TITLE = "binary-tree-classification"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"Running on device: {DEVICE}")

class CPLowRankLayer(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, rank, dropout_p=0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.dropout_p = dropout_p
        
        self.factor_left = nn.Parameter(torch.randn(num_nodes, rank, in_dim))
        self.factor_right = nn.Parameter(torch.randn(num_nodes, rank, in_dim))
        self.factor_out = nn.Parameter(torch.randn(num_nodes, rank, out_dim))
        
        self.scale = nn.Parameter(torch.ones(num_nodes, rank))

        self._initialize()

    def _initialize(self):
        with torch.no_grad():
            self.factor_left.data /= (self.factor_left.data.norm(dim=-1, keepdim=True) + 1e-8)
            self.factor_right.data /= (self.factor_right.data.norm(dim=-1, keepdim=True) + 1e-8)
            self.factor_out.data /= (self.factor_out.data.norm(dim=-1, keepdim=True) + 1e-8)
            self.scale.data.normal_(1.0, 0.02)

    def forward(self, x):
        B = x.size(0)  # batch size
        
        x = x.view(B, self.num_nodes, 2, -1)
        x_l = x[:, :, 0, :] # (B, N, In)
        x_r = x[:, :, 1, :] # (B, N, In)

        proj_l = torch.einsum('bni,nri->bnr', x_l, self.factor_left)
        proj_r = torch.einsum('bni,nri->bnr', x_r, self.factor_right)
        
        merged = self.scale.unsqueeze(0) * proj_l * proj_r
        
        if self.training and self.dropout_p > 0:
            mask = torch.bernoulli(torch.full_like(merged, 1 - self.dropout_p))
            merged = merged * mask / (1 - self.dropout_p)

        out = torch.einsum('bnr,nro->bno', merged, self.factor_out)
        out = out + x_l + x_r
        
        return out

class PatchTTN(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=1, 
                 num_classes=10, bond_dim=64, cp_rank=16, dropout=0.1):
        super().__init__()
        
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                    p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, bond_dim)
        )
        
        self.depth = int(torch.log2(torch.tensor(float(num_patches))))
        
        print(f"Tree Structure: {num_patches} leaves -> Depth {self.depth}")
        
        self.layers = nn.ModuleList()
        current_nodes = num_patches // 2
        
        for _ in range(self.depth):
            layer = CPLowRankLayer(
                num_nodes=current_nodes,
                in_dim=bond_dim,
                out_dim=bond_dim,
                rank=cp_rank,
                dropout_p=dropout
            )
            self.layers.append(layer)
            current_nodes //= 2
            
        self.norm = nn.LayerNorm(bond_dim)
        self.head = nn.Linear(bond_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(1)
        x = self.norm(x)
        return self.head(x)
