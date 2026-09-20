import math
from typing import Literal

import torch
from einops import rearrange
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import nn

from qnlp.discoviz.models.cp_node import CPQuadRankLayer
from qnlp.discoviz.models.node_variants import (
    DegreeReducedNode,
    PairwiseBinaryNode,
    TuckerNode,
    apply_isometric_parametrization,
)


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
    # Track 2, proposal C-c (default 0 = disabled, use patch_size as stride):
    # overlapping patches. Must be paired with a patch_size/image_size combo
    # that keeps num_patches = (image_size // stride)**2 a power of 4 -- e.g.
    # patch_size=4, stride=2, image_size=32 gives 16x16=256 leaves, depth 4,
    # the same tree shape as the non-overlapping B1 baseline (deliberately
    # NOT depth 5, which is the config that regressed in Stage B2). Requires
    # use_b1_feature_map (overlapping bilinear patches were never
    # implemented, since B1 is the only patch path this document still
    # recommends).
    patch_stride: int = 0
    patch_pad: int = 0
    # Track 2, proposal C-a (default False, requires use_b1_feature_map):
    # replace B1's single nn.Linear feature_proj -- shared across every
    # patch position -- with a position-specific [num_patches, in_features,
    # bond_dim] tensor, one independent projection per patch. The quadtree
    # itself is already per-node (CPQuadRankLayer indexes factors by node),
    # so this closes the one part of the model that was still weight-tied
    # across space. See TTN_CIFAR_EXPERIMENTS.md's "Plan forward".
    use_per_patch_embedding: bool = False
    # Route N (default "none"): gated non-linearity per quadtree layer,
    # mirroring the text tower's NLC. "born" = x + gate*x^2 (the real-valued
    # analogue of the Born rule, the principled quantum-inspired choice);
    # "gelu" = x + gate*GELU(x) (an upper bound on what any non-linearity
    # buys). gate inits at exactly 0.0 per layer, so this is a measurement,
    # not a capitulation -- at gate=0 the model is bit-for-bit unchanged.
    nonlinearity: str = "none"
    # NODE_ARCHITECTURE_PLAN.md: per-node computation. "cp" (default) is
    # today's 4-way CP-decomposed node, unchanged. "pairwise" = NODE-1
    # (standard TTN, two nested binary contractions). "degree2" = NODE-2
    # (sum of six pairwise products, gated on the kurtosis measurement
    # showing `merged` is heavy-tailed). "isometric" = NODE-3 (today's CP
    # node, tied, kept on the Stiefel manifold throughout training via
    # orthogonal parametrisation -- requires tie_nodes != "none").
    # "tucker" = NODE-4 (full Tucker core instead of CP's diagonal core --
    # requires tie_nodes != "none", the core has rank^5 entries).
    node_type: Literal["cp", "pairwise", "degree2", "isometric", "tucker"] = "cp"
    # Cross-cutting tying option: one shared tensor per level instead of
    # one per node. "none" (default, today's behaviour) | "all" (every
    # layer) | "fine" (layers 0-1 only, leaving the 4-node/1-node layers
    # per-node). Composes with every node_type; "isometric"/"tucker"
    # require "all" or "fine" (never "none").
    tie_nodes: Literal["none", "all", "fine"] = "none"
    # DTTN_IMPLEMENTATION_GUIDE.md: drop-in alternative image tower. "ttn"
    # (default) is TTNImageModel, unchanged. "dttn" selects DTTNImageModel
    # instead -- construct via `build_image_model`, not TTNImageModel(...)
    # directly, wherever a caller should respect this switch.
    # "clip" selects CLIPImageModel -- a frozen, externally-pretrained
    # sanity check (qnlp/discoviz/models/clip_image_model.py's docstring),
    # not a candidate architecture: it answers "is the task learnable at
    # all with a known-excellent image representation", not "should we
    # adopt this backbone".
    image_backbone: Literal["ttn", "dttn", "clip"] = "ttn"
    dttn_variant: Literal["T", "S", "L"] = "T"
    dttn_stem_patch: int = 2  # 1 for 32x32 inputs -- see the guide's resolution table
    dttn_transitions: str = "FTTT"  # per-stage flags, 'T'/'F', length 4
    dttn_use_ln: bool = True
    dttn_scale: int = 3  # expansion ratio
    clip_model_name: str = "openai/clip-vit-base-patch32"


image_model_hyperparams = ImageModelSettings()


def build_image_model(embedding_dim: int) -> nn.Module:
    """Factory respecting IMAGE_MODEL_IMAGE_BACKBONE. Prefer this over
    constructing TTNImageModel directly at any new call site so DTTN stays
    a config switch rather than a per-caller branch (DTTN_IMPLEMENTATION_
    GUIDE.md)."""
    if image_model_hyperparams.image_backbone == "dttn":
        from qnlp.discoviz.models.dttn_image_model import DTTNImageModel

        return DTTNImageModel(
            embedding_dim,
            variant=image_model_hyperparams.dttn_variant,
            stem_patch=image_model_hyperparams.dttn_stem_patch,
            transitions=image_model_hyperparams.dttn_transitions,
            use_ln=image_model_hyperparams.dttn_use_ln,
            scale=image_model_hyperparams.dttn_scale,
            in_channels=3 if image_model_hyperparams.use_color else 1,
        )
    if image_model_hyperparams.image_backbone == "clip":
        from qnlp.discoviz.models.clip_image_model import CLIPImageModel

        return CLIPImageModel(embedding_dim, model_name=image_model_hyperparams.clip_model_name)
    return TTNImageModel(embedding_dim)


class TTNImageModel(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.in_channels = 3 if image_model_hyperparams.use_color else 1
        self.embedding_dim = embedding_dim
        self.bond_dim = image_model_hyperparams.bond_dim
        self.patch_size = image_model_hyperparams.patch_size
        # Track 2, proposal C-c: stride < patch_size gives overlapping
        # patches. 0 means "no override" -> stride == patch_size, exactly
        # today's non-overlapping behaviour.
        self.patch_stride = image_model_hyperparams.patch_stride or self.patch_size
        self.patch_pad = image_model_hyperparams.patch_pad
        self.overlapping_patches = self.patch_stride != self.patch_size or self.patch_pad != 0

        num_patches_side = (
            image_model_hyperparams.image_size + 2 * self.patch_pad - self.patch_size
        ) // self.patch_stride + 1
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
            in_features = self.in_channels * self.patch_size**2 * 2
            self.use_per_patch_embedding = image_model_hyperparams.use_per_patch_embedding
            if self.use_per_patch_embedding:
                # C-a: one independent [in_features, bond_dim] projection per
                # patch position, instead of one shared across all of them.
                # Weight stored as [num_patches, in_features, bond_dim]
                # (transposed vs. nn.Linear's [out, in]) so the forward pass
                # is a plain per-patch matmul; init directly with nn.Linear's
                # default bound (1/sqrt(fan_in)) rather than calling
                # kaiming_uniform_, which would infer fan_in from the wrong
                # dimension for this shape.
                self.patch_weight = nn.Parameter(torch.empty(num_patches, in_features, self.bond_dim))
                self.patch_bias = nn.Parameter(torch.empty(num_patches, self.bond_dim))
                bound = 1 / math.sqrt(in_features)
                nn.init.uniform_(self.patch_weight, -bound, bound)
                nn.init.uniform_(self.patch_bias, -bound, bound)
            else:
                self.feature_proj = nn.Linear(in_features, self.bond_dim)
        else:
            self.use_per_patch_embedding = False
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

        node_type = image_model_hyperparams.node_type
        tie_mode = image_model_hyperparams.tie_nodes
        if node_type in ("isometric", "tucker") and tie_mode == "none":
            raise ValueError(f"node_type={node_type!r} requires tie_nodes != 'none' (see NODE_ARCHITECTURE_PLAN.md).")

        for i in range(self.depth):
            # Pruning: Remove residuals from Layer 0 & 1 to force feature learning
            use_res = True if i > 1 else False
            # gains only has hand-tuned values for the first 4 layers (the
            # depth every prior config used); deeper trees (e.g. B2's
            # per-pixel leaves, depth 5) fall back to the untuned default.
            gain = gains[i] if i < len(gains) else 1.0
            # "fine" ties only layers 0-1 (the many-node layers, where
            # per-position parameters are both most numerous and least
            # plausibly meaningful); "all" ties every layer.
            tied = tie_mode == "all" or (tie_mode == "fine" and i < 2)

            layer_kwargs = dict(
                num_nodes=current_nodes,
                in_dim=in_dim,
                out_dim=in_dim * 2,
                rank=image_model_hyperparams.cp_rank,
                dropout_p=image_model_hyperparams.dropout,
                use_residual=use_res,
                gain_factor=gain,
                use_isometric_init=image_model_hyperparams.use_isometric_init,
                nonlinearity=image_model_hyperparams.nonlinearity,
                tied=tied,
            )
            if node_type == "cp":
                layer = CPQuadRankLayer(**layer_kwargs)
            elif node_type == "isometric":
                layer = apply_isometric_parametrization(CPQuadRankLayer(**layer_kwargs))
            elif node_type == "pairwise":
                layer = PairwiseBinaryNode(**layer_kwargs)
            elif node_type == "degree2":
                layer = DegreeReducedNode(**layer_kwargs)
            elif node_type == "tucker":
                layer = TuckerNode(**layer_kwargs)
            else:
                raise ValueError(f"Unknown node_type: {node_type!r}")
            self.layers.append(layer)
            current_nodes //= 4
            in_dim *= 2

        self.final_norm = nn.LayerNorm(in_dim)
        self.head = nn.Linear(in_dim, self.embedding_dim)

    def nonlinear_gates(self) -> list[float] | None:
        """Route N: per-layer gate values, for logging their trajectory
        during training (the headline figure for this route). None if
        nonlinearity="none"."""
        if image_model_hyperparams.nonlinearity == "none":
            return None
        return [layer.gate.item() for layer in self.layers]

    def _patch_embed(self, x):
        if self.use_b1_feature_map:
            # Stage B1: invert ImageNet normalisation back to ~[0,1] pixel
            # intensities, apply the fixed angle encoding per pixel, then one
            # learned linear map from the per-patch phi-stack into bond_dim.
            x01 = (x * self._pixel_std + self._pixel_mean).clamp(0.0, 1.0)
            if self.mean_center_input:
                # Stage A4: batch-mean-centre before the feature map.
                x01 = (x01 - x01.mean(dim=0, keepdim=True)).clamp(-1.0, 1.0)
            if self.overlapping_patches:
                # Track 2, proposal C-c: stride < patch_size gives
                # overlapping patches. unfold's output column order is
                # row-major over the output spatial grid (same convention
                # as rearrange's "(h w)" below), so the rest of the forward
                # pass (positional embedding, quadtree reshape) needs no
                # further change.
                b = x01.shape[0]
                cols = nn.functional.unfold(
                    x01, kernel_size=self.patch_size, stride=self.patch_stride, padding=self.patch_pad
                )  # [b, c*patch_size**2, num_patches]
                patches = cols.view(b, self.in_channels, self.patch_size**2, -1).permute(0, 3, 1, 2)
            else:
                patches = rearrange(
                    x01, "b c (h p1) (w p2) -> b (h w) c (p1 p2)", p1=self.patch_size, p2=self.patch_size
                )
            phi = torch.stack([torch.cos(math.pi / 2 * patches), torch.sin(math.pi / 2 * patches)], dim=-1)
            phi_flat = phi.flatten(2)
            if self.use_per_patch_embedding:
                # C-a: independent projection per patch position instead of
                # one nn.Linear shared across all of them.
                x = torch.einsum("bnf,nfd->bnd", phi_flat, self.patch_weight) + self.patch_bias.unsqueeze(0)
            else:
                x = self.feature_proj(phi_flat)
        else:
            # 1. Bilinear Patch Mapping
            # [b, c, (h p1), (w p2)] -> [b, n, c, p]
            patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) c (p1 p2)", p1=self.patch_size, p2=self.patch_size)

            # Entangle Color and Pixels
            c_feat = torch.einsum("bncp, ck -> bnk", patches, self.color_factor)
            p_feat = torch.einsum("bncp, pk -> bnk", patches, self.pixel_factor)
            x = c_feat * p_feat  # Bilinear Interaction

        return x

    def _run_tree(self, x) -> list:
        """Run the quadtree contraction, returning every layer's raw output
        in order (index 0 = first layer below the leaves, index -1 = root,
        pre-squeeze/final_norm/head). P1: consumers index this list from the
        END to get root-indexed, depth-invariant region tensors — see
        forward_regions."""
        # 2. Add Gated Position
        pos_scale = 0.0 if self.zero_pos_scale else self.pos_scale  # Stage A5
        x = x + (self.positional_embedding * pos_scale)

        # 3. Tree Contraction
        current_grid_dim = int(math.sqrt(x.shape[1]))
        layer_outputs = []
        for layer in self.layers:
            # Reshape into 2x2 blocks for QuadTree contraction
            x = rearrange(x, "b (h w) c -> b c h w", h=current_grid_dim)
            x = rearrange(x, "b c (h h2) (w w2) -> b (h w) (h2 w2) c", h2=2, w2=2)

            x = layer(x)
            layer_outputs.append(x)
            current_grid_dim //= 2
        return layer_outputs

    def forward(self, x, normalize: bool = True):
        # normalize=False exposes the pre-L2-norm head output. Contrastive
        # training (every caller elsewhere) wants the default: cosine
        # similarity is scale-invariant so L2-norm is free there. A
        # classification probe is NOT scale-invariant — a TN classifier's
        # output magnitude can carry class signal — so
        # qnlp/discoviz/diagnostic/ttn_supervised_probe.py reads the raw
        # head output instead. See TTN_CIFAR_EXPERIMENTS.md Stage 0.2.
        x = self._patch_embed(x)
        layer_outputs = self._run_tree(x)

        # 4. Global Head
        x = layer_outputs[-1].squeeze(1)
        x = self.final_norm(x)
        x = self.head(x)

        return nn.functional.normalize(x, p=2, dim=-1) if normalize else x

    def forward_regions(self, x, level: int = 1):
        """P1 (TTN_CIFAR_EXPERIMENTS.md, shared prerequisite for Routes A/B):
        return [B, 4**level, dim_at_level] root-indexed region tensors.
        level=0 is the root (equivalent to forward()'s pooled output, before
        final_norm/head). Root-indexed rather than leaf-indexed because tree
        depth varies with image size, so this interface is depth-invariant:
        level=1 is always 4 regions, level=2 is always 16, regardless of how
        many quadtree layers the model has. Does not change forward()'s
        behaviour or outputs — runs the same _patch_embed/_run_tree path."""
        if not 0 <= level < len(self.layers):
            raise ValueError(f"level must be in [0, {len(self.layers) - 1}], got {level}")
        x = self._patch_embed(x)
        layer_outputs = self._run_tree(x)
        regions = layer_outputs[len(self.layers) - 1 - level]
        expected_nodes = 4**level
        assert regions.shape[1] == expected_nodes, (
            f"forward_regions(level={level}): expected {expected_nodes} regions, got {regions.shape[1]} "
            "-- check the root-indexing arithmetic against the model's actual depth."
        )
        return regions
