import math

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import nn
from torchvision.transforms import v2

from qnlp.discoclip2.models.cp_node import CPQuadRankLayer


class ImageModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IMAGE_MODEL_")
    use_color: bool = True
    bond_dim: int = 64
    cp_rank: int = 32
    dropout: float = 0.3
    patch_size: int = 4
    image_size: int = 64


image_model_hyperparams = ImageModelSettings()

transforms = [
    v2.RandomResizedCrop(image_model_hyperparams.image_size, scale=(0.8, 1.0)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
]
if not image_model_hyperparams.use_color:
    transforms.append(v2.Grayscale())
preprocess = v2.Compose(transforms)


class TTNImageModel(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.in_channels = 3 if image_model_hyperparams.use_color else 1
        self.embedding_dim = embedding_dim
        self.bond_dim = image_model_hyperparams.bond_dim
        self.patch_size = image_model_hyperparams.patch_size
        
        num_patches_side = image_model_hyperparams.image_size // self.patch_size
        num_patches = num_patches_side**2

        # FIX #3: BILINEAR PATCH EMBEDDING
        # Separates Color (What) and Space (Where) to boost initial Variance
        self.color_factor = nn.Parameter(torch.empty(self.in_channels, self.bond_dim))
        self.pixel_factor = nn.Parameter(torch.empty(self.patch_size**2, self.bond_dim))
        nn.init.xavier_uniform_(self.color_factor)
        nn.init.xavier_uniform_(self.pixel_factor)

        # FIX #1: GATED POSITIONAL EMBEDDING
        # Prevents position from drowning out image signal (SNR Fix)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, self.bond_dim))
        self.pos_scale = nn.Parameter(torch.tensor(0.05)) # Initialize to 5% of signal

        self.depth = int(math.log(num_patches, 4))
        self.layers = nn.ModuleList()
        
        current_nodes = num_patches // 4
        in_dim = self.bond_dim

        for i in range(self.depth):
            # Pruning: Remove residuals from Layer 0 & 1 to force feature learning
            use_res = True if i > 1 else False
            
            self.layers.append(CPQuadRankLayer(
                num_nodes=current_nodes,
                in_dim=in_dim,
                out_dim=in_dim * 2,
                rank=image_model_hyperparams.cp_rank,
                dropout_p=image_model_hyperparams.dropout,
                use_residual=use_res
            ))
            current_nodes //= 4
            in_dim *= 2

        self.final_norm = nn.LayerNorm(in_dim)
        self.head = nn.Linear(in_dim, self.embedding_dim)

    def forward(self, x):
        # 1. Bilinear Patch Mapping
        # [b, c, (h p1), (w p2)] -> [b, n, c, p]
        patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) c (p1 p2)", 
                            p1=self.patch_size, p2=self.patch_size)
        
        # Entangle Color and Pixels
        c_feat = torch.einsum("bncp, ck -> bnk", patches, self.color_factor)
        p_feat = torch.einsum("bncp, pk -> bnk", patches, self.pixel_factor)
        x = c_feat * p_feat # Bilinear Interaction
        
        # 2. Add Gated Position
        x = x + (self.positional_embedding * self.pos_scale)

        # 3. Tree Contraction
        current_grid_dim = int(math.sqrt(x.shape[1]))
        for layer in self.layers:
            # Reshape into 2x2 blocks for QuadTree contraction
            x = rearrange(x, "b (h w) c -> b c h w", h=current_grid_dim)
            x = rearrange(x, "b c (h h2) (w w2) -> b (h w) (h2 w2) c", h2=2, w2=2)
            
            x = layer(x)
            current_grid_dim //= 2

        # 4. Global Head
        x = x.squeeze(1)
        x = self.final_norm(x)
        x = self.head(x)

        return nn.functional.normalize(x, p=2, dim=-1)