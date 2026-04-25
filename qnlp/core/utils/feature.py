import torch
import torch.nn as nn


class FeatureMap(nn.Module):
    """Quantum-inspired feature map for encoding scalar pixel values into a 2D feature vector [cos, sin]."""

    def __init__(self):
        super().__init__()
        self.register_buffer("factor", torch.tensor(torch.pi / 2.0))

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.cat([torch.cos(self.factor * x), torch.sin(self.factor * x)], dim=-1)
