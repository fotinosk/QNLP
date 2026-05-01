import torch.nn as nn
from torch import Tensor


class VICRegProjector(nn.Module):
    """
    Expander MLP used by VICReg between the backbone and the loss.

    Maps L2-normalised backbone embeddings to a higher-dimensional
    non-normalised space where VICReg's variance/covariance terms
    can operate correctly. Not used at evaluation time.

    Architecture follows the original VICReg paper:
    Linear → BatchNorm → ReLU → Linear → BatchNorm → ReLU → Linear
    (no activation or norm on the final layer so the output is unconstrained)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 1024, output_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
