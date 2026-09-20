# Implementation guide: DTTN as an alternative image tower

**Audience: an implementing agent with no prior context on this project.**
Everything needed is in this document. Follow it top to bottom.

## Goal

Add `DTTNImageModel` as a **drop-in alternative** to the existing
`TTNImageModel` (`qnlp/discoviz/models/image_model.py`), selectable by
config. It must satisfy the same interface so that every existing caller
(CIFAR probe, SVO training, ARO training, diagnostics) works unchanged.

Do **not** delete, fork, or modify `TTNImageModel`. It stays the default.
Everything you add is opt-in and off by default.

## Source

Architecture from "Deep Tree Tensor Networks" (Chang Nie, NeurIPS 2025),
arXiv:2502.09928, official code at
`github.com/NieCha/deep_tree_tensor_network` (file `DTTN.py`).
Reported: CIFAR-10 95.0%, ImageNet-1K 77.9-82.4% at 7.1M-35.9M params,
with no ReLU/GELU-style activations anywhere.

**Do not clone the repo into this project.** Reimplement from the
reference below, in this project's style and file layout.

## Reference implementation (verbatim from the paper's repo)

```python
class AimBlock(nn.Module):
    def __init__(self, embed_dim, expansion, scale=3, use_ln=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_features = embed_dim * scale if expansion else embed_dim
        self.R1 = nn.Conv2d(self.embed_dim, self.hidden_features, kernel_size=1, padding=0, stride=1)
        self.R2 = nn.Conv2d(self.hidden_features, self.hidden_features, kernel_size=3, padding=1, stride=1, groups=self.hidden_features)
        self.L1 = nn.Conv2d(self.embed_dim, self.hidden_features, kernel_size=3, padding=1, stride=1, groups=self.embed_dim)
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
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.L1.weight)
        nn.init.kaiming_normal_(self.L2.weight)
        nn.init.kaiming_normal_(self.R1.weight)
        nn.init.kaiming_normal_(self.R2.weight)
        nn.init.kaiming_normal_(self.Pro.weight)
        nn.init.zeros_(self.L1.bias)
        nn.init.zeros_(self.L2.bias)
        nn.init.zeros_(self.R1.bias)
        nn.init.zeros_(self.R2.bias)
        nn.init.zeros_(self.Pro.bias)

    def forward(self, x):
        if self.use_ln:
            out = (self.L2(self.L1(x)))
            out_z = (self.R2(self.R1(x)))
            out = (out * out_z)
            out = rearrange(out, 'b c h w -> b h w c')
            out = self.norm1(out)
            out = rearrange(out, 'b h w c -> b c h w')
        else:
            out = self.norm1(self.L2(self.L1(x)))
            out_z = self.norm2(self.R2(self.R1(x)))
            out = (out * out_z)
        out = self.Pro(out)
        return x + self.alpha * self.norm3(out)


class Basic_blocks(nn.Module):
    def __init__(self, idx, nums, embed_dim):
        super().__init__()
        self.model = nn.Sequential(
            *[nn.Sequential(AimBlock(embed_dim, expansion=(i % 2 != 0))) for i in range(nums)]
        )

    def forward(self, x):
        return self.model(x)
```

Details that are easy to get wrong — copy them exactly:

- `L1` uses `groups=self.embed_dim` (**not** `groups=hidden_features`).
  `R2` uses `groups=self.hidden_features`. They are deliberately different.
- The two branches are the same two ops in **reversed order**: left is
  `grouped-3x3 -> pointwise-1x1`, right is `pointwise-1x1 -> depthwise-3x3`.
  This reversal is the whole "antisymmetric" idea.
- `expansion` **alternates per block**: `expansion=(i % 2 != 0)`, so block 0
  has `hidden = embed_dim`, block 1 has `hidden = embed_dim * 3`, and so on.
- `alpha` is initialised to **ones**, not zeros.
- With `use_ln=True` the LayerNorm is applied **after** the Hadamard product,
  with a permute to channels-last and back. `norm2` does not exist in that
  branch — do not create it.
- `norm3` (BatchNorm2d) is applied to `Pro`'s output, inside the residual.
- `rearrange` is from `einops`, already a dependency of this project.

## Full model assembly

```
stem:   Conv2d(in_chans, dims[0], k=patch, s=patch)
        Conv2d(dims[0],  dims[0], k=patch, s=patch)      # 4x downsample total at patch=2

body:   for stage_idx in range(4):
            if transitions[stage_idx]:
                downsample: Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
            Basic_blocks(stage_idx, layers[stage_idx], dims[stage_idx])

head:   global average pool -> Linear(dims[-1], embedding_dim)
```

Variant configs from the paper's repo:

| variant | transitions | layers | embed_dims |
|---|---|---|---|
| DTTN_T | [False, True, True, True] | [6, 6, 16, 6] | [96, 128, 160, 192] |
| DTTN_S | [False, True, True, True] | [6, 6, 24, 8] | [96, 128, 192, 192] |
| DTTN_L | [False, True, True, True] | [8, 8, 32, 8] | [128, 192, 256, 384] |

## CRITICAL adaptation: input resolution

The reference model targets ImageNet at 224x224. This project runs at
**64x64** (SVO/ARO) and **32x32** (CIFAR). The default stem plus three
downsamples reduces spatial dims by **32x**, which destroys small inputs:

| input | after stem (4x) | after 3 transitions | usable? |
|---|---|---|---|
| 224 | 56 | 7 | yes (reference) |
| 64 | 16 | 2 | marginal |
| 32 | 8 | 1 | **no — collapses to 1x1 before the last stages** |

**Required:** make the stem and transitions resolution-aware.

- For **32x32**: use `patch_size=1` in the stem (two 1x1 convs, no spatial
  reduction) so the stem output is 32x32, then the three transitions give
  16 -> 8 -> 4. Alternatively keep `patch_size=2` for the first stem conv
  only (16x16 after stem) and set `transitions=[False, True, True, False]`.
- For **64x64**: `patch_size=2` on the first stem conv only gives 32x32,
  then transitions give 16 -> 8 -> 4.

Expose this as config (see below) rather than hardcoding. Add an assertion
that the final feature map is at least 2x2 and raise a clear error naming
the offending config if not — silent collapse to 1x1 is exactly the class
of bug this project has been bitten by before.

## Required interface

Match `TTNImageModel` exactly. Read that class before writing anything.

```python
class DTTNImageModel(nn.Module):
    def __init__(self, embedding_dim: int): ...

    def forward(self, x, normalize: bool = True):
        """x: [B, 3, H, W] (ImageNet-normalised).
        Returns [B, embedding_dim]. L2-normalised iff normalize=True."""

    def forward_regions(self, x, level: int = 1):
        """Returns [B, 4**level, dim_at_level] root-indexed region tensors.
        level=0 is a single global vector."""
```

- `normalize: bool = True` default is load-bearing. Contrastive training
  needs the normalised output; the supervised CIFAR probe calls
  `backbone(x, normalize=False)` because softmax classification needs the
  unnormalised magnitude. Getting this wrong silently produces
  chance-level CIFAR accuracy — it has already happened once in this
  project.
- `forward_regions` is used by the Route A/B score heads. Map DTTN's
  **stage outputs** to root-indexed levels: the last stage's feature map
  is level 0 (pool it to 1 vector), the second-to-last gives level 1
  (adaptive-avg-pool its map to 2x2 = 4 regions), third-to-last gives
  level 2 (pool to 4x4 = 16 regions). Project each level to a common dim
  with a per-level `nn.Linear`, since stage dims differ.

## Config

Add to `ImageModelSettings` in `qnlp/discoviz/models/image_model.py`
(env prefix `IMAGE_MODEL_`):

```python
image_backbone: str = "ttn"        # "ttn" (default, unchanged) | "dttn"
dttn_variant: str = "T"            # "T" | "S" | "L"
dttn_stem_patch: int = 2           # 1 for 32x32 inputs
dttn_transitions: str = "FTTT"     # per-stage flags, 'T'/'F', length 4
dttn_use_ln: bool = True
dttn_scale: int = 3                # expansion ratio
```

Then, at every construction site of `TTNImageModel(...)`, select the class
by `image_backbone`. Construction sites to update (search for
`TTNImageModel(` — there are several):

- `qnlp/scripts/svo/run.py`
- `qnlp/scripts/aro_contrastive/run.py`
- `qnlp/discoviz/diagnostic/ttn_supervised_probe.py`
- `qnlp/discoviz/diagnostic/tower_spread_trace.py`
- `qnlp/discoviz/diagnostic/image_ablation.py`
- `qnlp/scripts/coco_multi_caption/evaluate.py`

Prefer a single factory, e.g. `build_image_model(embedding_dim)` in
`image_model.py`, and call it from all of the above, rather than repeating
the branch six times.

## What must not change

- `TTNImageModel` stays the default. With `IMAGE_MODEL_IMAGE_BACKBONE`
  unset, every existing run must be **bit-for-bit identical** to today.
- Do not change the data pipeline, transforms, losses, training loop, or
  any config defaults other than adding the new fields above.
- Do not add a DTTN classifier head. The head here is
  `pool -> Linear(dims[-1], embedding_dim)`; classification is the probe's
  job, via its own `TTNClassifier`.

## Validation, in order — do not skip

1. **Shape smoke test (local, CPU).** Construct at `embedding_dim=512` for
   32x32 and 64x64 inputs. Check `forward` returns `[B, 512]`, that
   `normalize=True` gives unit-norm rows and `normalize=False` does not,
   and that `forward_regions(x, level=1)` returns `[B, 4, dim]`.
2. **Backward smoke test.** One forward+backward on random data; confirm
   finite gradients on `stem`, every stage, and the head.
3. **Parameter count.** Log it. DTTN-T at ImageNet config is ~7.1M; at
   this project's smaller resolutions it will differ, but a count that is
   orders of magnitude off means the config is wrong.
4. **Regression check.** Run the existing CIFAR probe with the backbone
   flag **unset** and confirm the TTN number still reproduces (~0.5857
   with the GELU-gated config). If it moved, you changed a shared path.
5. **CIFAR-10 benchmark.** Run the probe with
   `IMAGE_MODEL_IMAGE_BACKBONE=dttn`, variant T, `--arch ttn` slot.
   Reference points: TTN 0.5857, CNN 0.7842, ResNet18 0.7569, logreg
   0.3784. The paper reports 95.0% for DTTN-S at 224x224 with full
   ImageNet-style augmentation and a long schedule — **do not expect that
   here**. At 32x32 with this project's short schedule, anything above
   0.70 is a strong result and above 0.60 already beats the current tower.

## Known gotchas

- **`rearrange` import**: `from einops import rearrange`, already used in
  `image_model.py`.
- **BatchNorm with small batches**: `norm3` and the non-LN path use
  BatchNorm2d. SVO training uses `batch_size=128` (fine), but evaluation
  and the ablation scripts sometimes run smaller batches — make sure
  `model.eval()` is called in all diagnostic paths, otherwise BatchNorm
  will use batch statistics and results will be unstable. The existing
  diagnostics do call `.eval()`; verify rather than assume.
- **The "multilinear" claim**: DTTN has no ReLU/GELU, but it does use
  LayerNorm/BatchNorm, which are not linear operations. If anyone writes
  about this, the accurate phrasing is "no nonlinear activation
  functions", not "strictly multilinear".
- **No pretrained weights exist.** The official repo publishes no
  checkpoints and `hubconf.py` only re-exports timm's registry. Any
  transfer experiment must first train DTTN here (CIFAR-10 is the cheap
  option) and warm-start from that, following the existing
  `_warm_start_from_aro` pattern in `qnlp/scripts/svo/run.py`.

## Out of scope for this task

Do not wire DTTN into SVO/ARO training runs yet, and do not attempt any
transfer or warm-start experiment. This task ends when DTTN is selectable,
passes validation steps 1-4, and has a CIFAR-10 number from step 5.
