import math
import torch
from torch import nn
from torchvision.transforms import v2
from einops.layers.torch import Rearrange
from einops import rearrange



BOND_DIM = 20       
CP_RANK = 16           
DROPOUT = 0.3
PATCH_SIZE = 4  
IMAGE_SIZE = 32
# IMAGE_SIZE = 64


preprocess = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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
        
        # We now have 4 input factors corresponding to the 2x2 spatial block
        self.factor_tl = nn.Parameter(torch.randn(num_nodes, rank, in_dim)) # Top-Left
        self.factor_tr = nn.Parameter(torch.randn(num_nodes, rank, in_dim)) # Top-Right
        self.factor_bl = nn.Parameter(torch.randn(num_nodes, rank, in_dim)) # Bottom-Left
        self.factor_br = nn.Parameter(torch.randn(num_nodes, rank, in_dim)) # Bottom-Right
        
        self.factor_out = nn.Parameter(torch.randn(num_nodes, rank, out_dim))
        self.scale = nn.Parameter(torch.ones(num_nodes, rank))

        self._initialize()

    def _initialize(self):
        with torch.no_grad():
            for f in [self.factor_tl, self.factor_tr, self.factor_bl, self.factor_br, self.factor_out]:
                f.data /= (f.data.norm(dim=-1, keepdim=True) + 1e-8)
            self.scale.data.normal_(1.0, 0.02)

    def forward(self, x):
        # x shape expected: [Batch, Num_Nodes, 4, In_Dim]
        # The 4 dimension corresponds to: 0:TL, 1:TR, 2:BL, 3:BR
        
        x_tl = x[:, :, 0, :]
        x_tr = x[:, :, 1, :]
        x_bl = x[:, :, 2, :]
        x_br = x[:, :, 3, :]

        # using 'bni' (batch, node, in) -> 'bnr' (batch, node, rank)
        p_tl = torch.einsum('bni,nri->bnr', x_tl, self.factor_tl)
        p_tr = torch.einsum('bni,nri->bnr', x_tr, self.factor_tr)
        p_bl = torch.einsum('bni,nri->bnr', x_bl, self.factor_bl)
        p_br = torch.einsum('bni,nri->bnr', x_br, self.factor_br)
        
        # This approximates a high-order tensor interaction
        merged = self.scale.unsqueeze(0) * p_tl * p_tr * p_bl * p_br
        
        if self.training and self.dropout_p > 0:
            mask = torch.bernoulli(torch.full_like(merged, 1 - self.dropout_p))
            merged = merged * mask / (1 - self.dropout_p)

        # 3. Project to Output
        out = torch.einsum('bnr,nro->bno', merged, self.factor_out)
        
        # 4. Residual Connection
        # Sum of all inputs added to output (Skip connection)
        out = out + x_tl + x_tr + x_bl + x_br
        return out


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
        
                # Standard patch embedding
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                    p1=PATCH_SIZE, p2=PATCH_SIZE),
            nn.Linear(patch_dim, BOND_DIM)
        )
        
        # Calculate Depth (Log base 4 because we reduce nodes by 4 each time)
        # e.g., 64 patches -> 16 nodes -> 4 nodes -> 1 node (Depth 3)
        self.depth = int(math.log(num_patches, 4))
        
        print(f"QuadTree Structure: {num_patches} leaves -> Depth {self.depth}")
        
        self.layers = nn.ModuleList()
        current_nodes = num_patches // 4 # First layer outputs this many nodes
        
        for _ in range(self.depth):
            layer = CPQuadRankLayer(
                num_nodes=current_nodes,
                in_dim=BOND_DIM,
                out_dim=BOND_DIM,
                rank=CP_RANK,
                dropout_p=DROPOUT
            )
            self.layers.append(layer)
            current_nodes //= 4
            
        self.norm = nn.LayerNorm(BOND_DIM)
        self.head = nn.Linear(BOND_DIM, self.embedding_dim)
        
    def forward(self, x):
        x = self.patch_embed(x) # [B, 64, Dim]
        
        current_grid_dim = self.num_patches_side # e.g., 8 (for 8x8=64 patches)

        for layer in self.layers:
            # CRITICAL STEP: 
            # We must group the flat sequence into 2x2 spatial blocks.
            # 1. 'b (h w) c' -> restore to 2D grid 'b c h w'
            # 2. chop into 2x2 blocks -> 'b (h_new w_new) 4 c'
            
            x = rearrange(x, 'b (h w) c -> b c h w', h=current_grid_dim)
            
            # h2=2, w2=2 means we are grabbing 2x2 blocks
            x = rearrange(x, 'b c (h h2) (w w2) -> b (h w) (h2 w2) c', h2=2, w2=2)
            
            # Now x is [Batch, New_Num_Nodes, 4, Dim]. 
            # The '4' contains TL, TR, BL, BR spatially.
            
            x = layer(x) # Returns [Batch, New_Num_Nodes, Dim]
            
            # Update grid dimension for next layer (grid shrinks by half on each side)
            current_grid_dim //= 2
            
        x = x.squeeze(1) # [B, 1, Dim] -> [B, Dim]
        x = self.norm(x)
        x = self.head(x)
        return nn.functional.normalize(x, dim=-1)
