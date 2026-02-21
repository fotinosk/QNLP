import math
import torch
from torch import nn
from torchvision.transforms import v2
from einops.layers.torch import Rearrange
from einops import rearrange
from pydantic_settings import BaseSettings, SettingsConfigDict

from qnlp.discoclip2.models.cp_node import CPQuadRankLayer


class ImageModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IMAGE_MODEL_"
    )
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
    def __init__(
        self,
        embedding_dim: int
    ):
        super().__init__()
        self.in_channels = 3 if image_model_hyperparams.use_color else 1
        self.embedding_dim = embedding_dim
        self.num_patches_side =  image_model_hyperparams.image_size // image_model_hyperparams.patch_size
        num_patches = self.num_patches_side ** 2
        patch_dim = image_model_hyperparams.patch_size * image_model_hyperparams.patch_size * self.in_channels
        
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                    p1=image_model_hyperparams.patch_size, p2=image_model_hyperparams.patch_size),
            nn.Linear(patch_dim, image_model_hyperparams.bond_dim)
        )
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, image_model_hyperparams.bond_dim))
        
        self.depth = int(math.log(num_patches, 4))
        
        print(f"QuadTree Structure: {num_patches} leaves -> Depth {self.depth}")
        
        self.layers = nn.ModuleList()
        current_nodes = num_patches // 4 # First layer outputs this many nodes

        in_dim = image_model_hyperparams.bond_dim
        
        for _ in range(self.depth):
            layer = CPQuadRankLayer(
                num_nodes=current_nodes,
                in_dim=in_dim,
                out_dim=in_dim * 2,
                rank=image_model_hyperparams.cp_rank,
                dropout_p=image_model_hyperparams.dropout
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
