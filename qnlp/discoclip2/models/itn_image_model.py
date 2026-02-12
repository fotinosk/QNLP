"""Inhomogeneous tensor network image model
Each patch gets a different CP layer
"""
import math
import torch
from torch import nn

from einops.layers.torch import Rearrange
from einops import rearrange

from qnlp.discoclip2.models.cp_node import CPQuadRankLayer
from qnlp.discoclip2.models.image_model import IMAGE_SIZE, PATCH_SIZE, BOND_DIM, DROPOUT, CP_RANK


class InhomogeneousTTNImageModel(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2

        patch_dim = PATCH_SIZE * PATCH_SIZE

        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=PATCH_SIZE, p2=PATCH_SIZE),
            nn.Linear(patch_dim, BOND_DIM)
        )
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, BOND_DIM))

        # Calculate Depth (Log base 4 because we reduce nodes by 4 each time)
        # e.g., 64 patches -> 16 nodes -> 4 nodes -> 1 node (Depth 3)
        self.depth = int(math.log(self.num_patches, 4))

        print(f"Inhomogeneous QuadTree Structure: {self.num_patches} leaves -> Depth {self.depth}")

        self.layers = nn.ModuleList()
        current_nodes = self.num_patches // 4

        for _ in range(self.depth):
            layer_nodes = nn.ModuleList()
            for _ in range(current_nodes):
                node = CPQuadRankLayer(
                    num_nodes=1,
                    in_dim=BOND_DIM,
                    out_dim=BOND_DIM,
                    rank=CP_RANK,
                    dropout_p=DROPOUT
                )
                layer_nodes.append(node)
            self.layers.append(layer_nodes)
            current_nodes //= 4

        self.norm = nn.LayerNorm(BOND_DIM)
        self.head = nn.Linear(BOND_DIM, self.embedding_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.positional_embedding

        current_grid_dim = IMAGE_SIZE // PATCH_SIZE
        for layer_nodes in self.layers:
            x = rearrange(x, 'b (h w) c -> b c h w', h=current_grid_dim)
            x = rearrange(x, 'b c (h h2) (w w2) -> b (h w) (h2 w2) c', h2=2, w2=2)
            node_inputs = torch.chunk(x, len(layer_nodes), dim=1)
            level_outputs = []
            for node_data, node_module in zip(node_inputs, layer_nodes):
                level_outputs.append(node_module(node_data))
            x = torch.cat(level_outputs, dim=1)

            current_grid_dim //= 2

        x = x.squeeze(1)
        x = self.norm(x)
        x = self.head(x)

        return nn.functional.normalize(x, p=2, dim=-1)
