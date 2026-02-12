import math
import torch
from torch import nn
from torchvision.transforms import v2
from einops.layers.torch import Rearrange
from einops import rearrange



BOND_DIM = 64
CP_RANK = 16           
DROPOUT = 0.3
PATCH_SIZE = 4  
IMAGE_SIZE = 32
# IMAGE_SIZE = 64


preprocess = v2.Compose([
    v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)), # Model can't memorize fixed positions
    v2.ColorJitter(brightness=0.2, contrast=0.2), # Model can't memorize exact color stats    
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Grayscale(),
])


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
        # CHANGED: Switched to Xavier initialization for stable variance
        with torch.no_grad():
            for f in [self.factor_tl, self.factor_tr, self.factor_bl, self.factor_br]:
                nn.init.xavier_uniform_(f)
            nn.init.xavier_uniform_(self.factor_out)

    def forward(self, x):
        # CHANGED: Residual is now the MEAN to control magnitude
        res = self.res_proj(x.mean(dim=2))
        
        # ADDED: Pre-Norm application
        x = self.norm(x)

        x_tl = x[:, :, 0, :]
        x_tr = x[:, :, 1, :]
        x_bl = x[:, :, 2, :]
        x_br = x[:, :, 3, :]

        p_tl = torch.einsum('bni,nri->bnr', x_tl, self.factor_tl)
        p_tr = torch.einsum('bni,nri->bnr', x_tr, self.factor_tr)
        p_bl = torch.einsum('bni,nri->bnr', x_bl, self.factor_bl)
        p_br = torch.einsum('bni,nri->bnr', x_br, self.factor_br)
        
        merged = self.scale.unsqueeze(0) * p_tl * p_tr * p_bl * p_br
        
        if self.training and self.dropout_p > 0:
            mask = torch.bernoulli(torch.full_like(merged, 1 - self.dropout_p))
            merged = merged * mask / (1 - self.dropout_p)

        out = torch.einsum('bnr,nro->bno', merged, self.factor_out)
        
        return out + res

class TTNImageModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int
        ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_patches_side =  IMAGE_SIZE // PATCH_SIZE
        num_patches = self.num_patches_side ** 2
        patch_dim = PATCH_SIZE * PATCH_SIZE
        
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                    p1=PATCH_SIZE, p2=PATCH_SIZE),
            nn.Linear(patch_dim, BOND_DIM)
        )
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, BOND_DIM))
        
        # Calculate Depth (Log base 4 because we reduce nodes by 4 each time)
        # e.g., 64 patches -> 16 nodes -> 4 nodes -> 1 node (Depth 3)
        self.depth = int(math.log(num_patches, 4))
        
        print(f"QuadTree Structure: {num_patches} leaves -> Depth {self.depth}")
        
        self.layers = nn.ModuleList()
        current_nodes = num_patches // 4 # First layer outputs this many nodes

        in_dim = BOND_DIM
        
        for _ in range(self.depth):
            layer = CPQuadRankLayer(
                num_nodes=current_nodes,
                in_dim=in_dim,
                out_dim=in_dim * 2,
                rank=CP_RANK,
                dropout_p=DROPOUT
            )
            self.layers.append(layer)
            current_nodes //= 4
            in_dim *= 2
            
        self.norm = nn.LayerNorm(in_dim)
        self.head = nn.Linear(in_dim, self.embedding_dim)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.positional_embedding
        
        current_grid_dim = self.num_patches_side

        for layer in self.layers:
            x = rearrange(x, 'b (h w) c -> b c h w', h=current_grid_dim)
            x = rearrange(x, 'b c (h h2) (w w2) -> b (h w) (h2 w2) c', h2=2, w2=2)
            
            x = layer(x) 
            
            current_grid_dim //= 2
            
        x = x.squeeze(1) 
        x = self.norm(x)
        x = self.head(x)
        
        return nn.functional.normalize(x, p=2, dim=-1)
