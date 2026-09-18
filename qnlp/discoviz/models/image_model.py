import math

import torch
from einops import rearrange
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import nn

from qnlp.discoviz.models.cp_node import CPQuadRankLayer


class ImageModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IMAGE_MODEL_")
    use_color: bool = True
    bond_dim: int = 64
    cp_rank: int = 32
    dropout: float = 0.3
    patch_size: int = 4
    image_size: int = 64
    # TTN_CIFAR_EXPERIMENTS.md Stage B1: replace the learned bilinear patch
    # embedding (a degenerate rank-1 quadratic form, confirmed to destroy
    # most class structure in one operation via the structure trace) with a
    # fixed per-pixel angle-encoding feature map + one learned linear layer.
    # Default False — this is still an active investigation scoped to the
    # CIFAR-10 capacity probe; SVO/ARO/COCO keep today's behaviour unless
    # explicitly opted in via IMAGE_MODEL_USE_B1_FEATURE_MAP=true.
    use_b1_feature_map: bool = False
    # Stage A1 (default True, the verified fix). False reproduces the
    # original defective per-node init, kept only for the parallel batch
    # plan's row 2 ablation (B1 alone vs A1+B1).
    use_isometric_init: bool = True
    # Stage A4 (default False): batch-mean-center the ~[0,1] pixel values
    # right before the (fixed or bilinear) patch embedding. Uses batch
    # statistics, not precomputed dataset statistics -- a direct, cheap
    # proxy matching what tower_spread_trace.py's --mean-center ablation
    # already measured, not a claim of the exact "dataset mean" phrasing
    # in the original Stage A4 write-up.
    mean_center_input: bool = False
    # Stage A5 (default False): zero the positional embedding's
    # contribution entirely, regardless of the learned pos_scale value.
    zero_pos_scale: bool = False


image_model_hyperparams = ImageModelSettings()


class TTNImageModel(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.in_channels = 3 if image_model_hyperparams.use_color else 1
        self.embedding_dim = embedding_dim
        self.bond_dim = image_model_hyperparams.bond_dim
        self.patch_size = image_model_hyperparams.patch_size

        num_patches_side = image_model_hyperparams.image_size // self.patch_size
        num_patches = num_patches_side**2

        self.use_b1_feature_map = image_model_hyperparams.use_b1_feature_map
        self.mean_center_input = image_model_hyperparams.mean_center_input
        self.zero_pos_scale = image_model_hyperparams.zero_pos_scale
        if self.use_b1_feature_map:
            # Stage B1: fixed angle encoding phi(x) = [cos(pi*x/2), sin(pi*x/2)],
            # x in [0,1], applied per raw pixel value -- the quantum-inspired
            # analogue of a qubit angle encoding (state preparation is where
            # the non-linearity belongs, not the circuit; see
            # TTN_CIFAR_EXPERIMENTS.md's "What multilinear does and does not
            # forbid"). Callers pass ImageNet-normalised tensors (every
            # existing data pipeline in this project does), so forward()
            # inverts that normalisation back to ~[0,1] before applying phi --
            # this keeps every external dataset/transform contract unchanged.
            self.register_buffer("_pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.feature_proj = nn.Linear(self.in_channels * self.patch_size**2 * 2, self.bond_dim)
        else:
            # FIX #3: BILINEAR PATCH EMBEDDING
            # Separates Color (What) and Space (Where) to boost initial Variance
            self.color_factor = nn.Parameter(torch.empty(self.in_channels, self.bond_dim))
            self.pixel_factor = nn.Parameter(torch.empty(self.patch_size**2, self.bond_dim))
            nn.init.xavier_uniform_(self.color_factor)
            nn.init.xavier_uniform_(self.pixel_factor)

        # FIX #1: GATED POSITIONAL EMBEDDING
        # Prevents position from drowning out image signal (SNR Fix)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, self.bond_dim))
        self.pos_scale = nn.Parameter(torch.tensor(0.05))  # Initialize to 5% of signal

        self.depth = int(math.log(num_patches, 4))
        self.layers = nn.ModuleList()

        current_nodes = num_patches // 4
        in_dim = self.bond_dim

        gains = [2.0, 1.5, 1.0, 1.0]

        for i in range(self.depth):
            # Pruning: Remove residuals from Layer 0 & 1 to force feature learning
            use_res = True if i > 1 else False
            gain = gains[i]

            self.layers.append(
                CPQuadRankLayer(
                    num_nodes=current_nodes,
                    in_dim=in_dim,
                    out_dim=in_dim * 2,
                    rank=image_model_hyperparams.cp_rank,
                    dropout_p=image_model_hyperparams.dropout,
                    use_residual=use_res,
                    gain_factor=gain,
                    use_isometric_init=image_model_hyperparams.use_isometric_init,
                )
            )
            current_nodes //= 4
            in_dim *= 2

        self.final_norm = nn.LayerNorm(in_dim)
        self.head = nn.Linear(in_dim, self.embedding_dim)

    def forward(self, x, normalize: bool = True):
        # normalize=False exposes the pre-L2-norm head output. Contrastive
        # training (every caller elsewhere) wants the default: cosine
        # similarity is scale-invariant so L2-norm is free there. A
        # classification probe is NOT scale-invariant — a TN classifier's
        # output magnitude can carry class signal — so
        # qnlp/discoviz/diagnostic/ttn_supervised_probe.py reads the raw
        # head output instead. See TTN_CIFAR_EXPERIMENTS.md Stage 0.2.
        if self.use_b1_feature_map:
            # Stage B1: invert ImageNet normalisation back to ~[0,1] pixel
            # intensities, apply the fixed angle encoding per pixel, then one
            # learned linear map from the per-patch phi-stack into bond_dim.
            x01 = (x * self._pixel_std + self._pixel_mean).clamp(0.0, 1.0)
            if self.mean_center_input:
                # Stage A4: batch-mean-centre before the feature map.
                x01 = (x01 - x01.mean(dim=0, keepdim=True)).clamp(-1.0, 1.0)
            patches = rearrange(x01, "b c (h p1) (w p2) -> b (h w) c (p1 p2)", p1=self.patch_size, p2=self.patch_size)
            phi = torch.stack([torch.cos(math.pi / 2 * patches), torch.sin(math.pi / 2 * patches)], dim=-1)
            x = self.feature_proj(phi.flatten(2))
        else:
            # 1. Bilinear Patch Mapping
            # [b, c, (h p1), (w p2)] -> [b, n, c, p]
            patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) c (p1 p2)", p1=self.patch_size, p2=self.patch_size)

            # Entangle Color and Pixels
            c_feat = torch.einsum("bncp, ck -> bnk", patches, self.color_factor)
            p_feat = torch.einsum("bncp, pk -> bnk", patches, self.pixel_factor)
            x = c_feat * p_feat  # Bilinear Interaction

        # 2. Add Gated Position
        pos_scale = 0.0 if self.zero_pos_scale else self.pos_scale  # Stage A5
        x = x + (self.positional_embedding * pos_scale)

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

        return nn.functional.normalize(x, p=2, dim=-1) if normalize else x
