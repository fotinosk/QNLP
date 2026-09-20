"""
DTTN (Deep Tree Tensor Network) image tower — DTTN_IMPLEMENTATION_GUIDE.md.

Reimplemented from "Deep Tree Tensor Networks" (Chang Nie, NeurIPS 2025,
arXiv:2502.09928; official code github.com/NieCha/deep_tree_tensor_network,
DTTN.py). A drop-in alternative to TTNImageModel (same interface), selected
via IMAGE_MODEL_IMAGE_BACKBONE=dttn. Not wired into SVO/ARO training yet
(DTTN_IMPLEMENTATION_GUIDE.md's "Out of scope for this task").

Reported 95.0% CIFAR-10 / 77.9-82.4% ImageNet-1K with no ReLU/GELU-style
nonlinear activation anywhere (LayerNorm/BatchNorm are used, which are not
linear operations -- "no nonlinear activation functions" is the accurate
claim, not "strictly multilinear"). That external result is this project's
evidence that the node/backbone, not the multilinear constraint, has been
the ceiling -- see NODE_ARCHITECTURE_PLAN.md.
"""

import torch
import torch.nn as nn
from einops import rearrange

_VARIANTS = {
    "T": {"transitions": [False, True, True, True], "layers": [6, 6, 16, 6], "dims": [96, 128, 160, 192]},
    "S": {"transitions": [False, True, True, True], "layers": [6, 6, 24, 8], "dims": [96, 128, 192, 192]},
    "L": {"transitions": [False, True, True, True], "layers": [8, 8, 32, 8], "dims": [128, 192, 256, 384]},
}


class AimBlock(nn.Module):
    """Antisymmetric Interaction Module: two branches applying the same two
    ops (grouped-3x3, pointwise-1x1) in reversed order, multiplied
    together. The reversal is the "antisymmetric" idea -- copied verbatim
    from the reference; do not "simplify" the two branches to match, that
    changes the architecture."""

    def __init__(self, embed_dim: int, expansion: bool, scale: int = 3, use_ln: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_features = embed_dim * scale if expansion else embed_dim
        self.R1 = nn.Conv2d(self.embed_dim, self.hidden_features, kernel_size=1, padding=0, stride=1)
        self.R2 = nn.Conv2d(
            self.hidden_features, self.hidden_features, kernel_size=3, padding=1, stride=1, groups=self.hidden_features
        )
        self.L1 = nn.Conv2d(
            self.embed_dim, self.hidden_features, kernel_size=3, padding=1, stride=1, groups=self.embed_dim
        )
        self.L2 = nn.Conv2d(self.hidden_features, self.hidden_features, kernel_size=1, padding=0, stride=1, groups=1)
        self.Pro = nn.Conv2d(self.hidden_features, self.embed_dim, kernel_size=1, padding=0, stride=1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.use_ln = use_ln
        if not self.use_ln:
            self.norm1 = nn.BatchNorm2d(self.hidden_features, eps=1e-5)
            self.norm2 = nn.BatchNorm2d(self.hidden_features, eps=1e-5)
        else:
            self.norm1 = nn.LayerNorm(self.hidden_features, eps=1e-6)
        self.norm3 = nn.BatchNorm2d(self.embed_dim, eps=1e-5)
        self._init_weights()

    def _init_weights(self):
        for conv in (self.L1, self.L2, self.R1, self.R2, self.Pro):
            nn.init.kaiming_normal_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        if self.use_ln:
            out = self.L2(self.L1(x))
            out_z = self.R2(self.R1(x))
            out = out * out_z
            out = rearrange(out, "b c h w -> b h w c")
            out = self.norm1(out)
            out = rearrange(out, "b h w c -> b c h w")
        else:
            out = self.norm1(self.L2(self.L1(x)))
            out_z = self.norm2(self.R2(self.R1(x)))
            out = out * out_z
        out = self.Pro(out)
        return x + self.alpha * self.norm3(out)


class BasicBlocks(nn.Module):
    """One stage: `nums` AimBlocks, expansion alternating per block
    (block 0 no expansion, block 1 expanded, block 2 no expansion, ...) --
    copied from the reference, not a simplification."""

    def __init__(self, nums: int, embed_dim: int, scale: int, use_ln: bool):
        super().__init__()
        self.model = nn.Sequential(
            *[AimBlock(embed_dim, expansion=(i % 2 != 0), scale=scale, use_ln=use_ln) for i in range(nums)]
        )

    def forward(self, x):
        return self.model(x)


class DTTNImageModel(nn.Module):
    """Drop-in alternative to TTNImageModel -- same forward/forward_regions
    interface. See DTTN_IMPLEMENTATION_GUIDE.md for the full spec and the
    resolution-adaptation rationale (the reference's 32x total downsample
    collapses a 32x32/64x64 input to <=1x1 unless the stem/transitions are
    adjusted for this project's much smaller inputs)."""

    def __init__(
        self,
        embedding_dim: int,
        variant: str = "T",
        stem_patch: int = 2,
        transitions: str | None = None,
        use_ln: bool = True,
        scale: int = 3,
        in_channels: int = 3,
    ):
        super().__init__()
        cfg = _VARIANTS[variant]
        dims = cfg["dims"]
        layers = cfg["layers"]
        trans = [c == "T" for c in transitions] if transitions is not None else cfg["transitions"]
        if len(trans) != 4:
            raise ValueError(f"transitions must have length 4, got {trans!r}")

        self.embedding_dim = embedding_dim
        self.dims = dims

        # Stem: two convs at stride `stem_patch` each (4x total downsample
        # at the reference's stem_patch=2). For 32x32 inputs, callers pass
        # stem_patch=1 (see DTTN_IMPLEMENTATION_GUIDE.md's resolution table)
        # so the stem does not spatially reduce at all, only the 4 stage
        # transitions do.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=stem_patch, stride=stem_patch),
            nn.Conv2d(dims[0], dims[0], kernel_size=stem_patch, stride=stem_patch),
        )

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_dim = dims[0]
        for i in range(4):
            if trans[i]:
                self.downsamples.append(nn.Conv2d(prev_dim, dims[i], kernel_size=2, stride=2))
            else:
                self.downsamples.append(nn.Identity() if prev_dim == dims[i] else nn.Conv2d(prev_dim, dims[i], 1))
            self.stages.append(BasicBlocks(layers[i], dims[i], scale, use_ln))
            prev_dim = dims[i]

        self.head = nn.Linear(dims[-1], embedding_dim)

    def _stage_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run stem + all 4 stages, returning each stage's output feature
        map in order (index 0 = first stage, index -1 = last/coarsest) --
        forward_regions reads this list from the end, mirroring
        TTNImageModel's root-indexed layer_outputs convention."""
        x = self.stem(x)
        outputs = []
        for downsample, stage in zip(self.downsamples, self.stages):
            x = downsample(x)
            x = stage(x)
            outputs.append(x)
        min_spatial = min(outputs[-1].shape[-2:])
        if min_spatial < 2:
            raise ValueError(
                f"DTTN's final feature map collapsed to {tuple(outputs[-1].shape[-2:])} "
                "(< 2x2) -- reduce dttn_stem_patch or the number of 'T' transitions "
                "for this input resolution. See DTTN_IMPLEMENTATION_GUIDE.md's resolution table."
            )
        return outputs

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        outputs = self._stage_outputs(x)
        pooled = outputs[-1].mean(dim=(-2, -1))  # global average pool
        out = self.head(pooled)
        return nn.functional.normalize(out, p=2, dim=-1) if normalize else out

    def forward_regions(self, x: torch.Tensor, level: int = 1) -> torch.Tensor:
        """Root-indexed region tensors, mirroring TTNImageModel.forward_regions:
        level=0 is the last stage pooled to 1 vector; level=1 the second-to-
        last stage pooled to 2x2=4 regions; level=2 the third-to-last pooled
        to 4x4=16 regions. Each level is projected to a common `dim_at_level`
        (that stage's own channel count) via a lazily-created per-level
        nn.Linear only when dims actually differ across levels -- here every
        level already has a fixed channel count from `self.dims`, so no
        projection is needed; consumers get raw per-stage-dim region
        tensors just like TTNImageModel's forward_regions does."""
        outputs = self._stage_outputs(x)
        n_stages = len(outputs)
        if not 0 <= level < n_stages:
            raise ValueError(f"level must be in [0, {n_stages - 1}], got {level}")
        stage_out = outputs[n_stages - 1 - level]
        grid = 2**level
        pooled = nn.functional.adaptive_avg_pool2d(stage_out, (grid, grid))
        regions = rearrange(pooled, "b c h w -> b (h w) c")
        expected_nodes = 4**level
        assert (
            regions.shape[1] == expected_nodes
        ), f"forward_regions(level={level}): expected {expected_nodes} regions, got {regions.shape[1]}"
        return regions
