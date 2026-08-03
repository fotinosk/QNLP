"""CLEVR shape-binding composites — Task C6, the closing experiment.

WHY THIS TASK EXISTS
--------------------
`mlp_reference` beats every TTN arm on every head of C3, and beats them on C4's
relations (65.4 vs classical_bare 49.6, quantum_none 46.2). Read naively that is
evidence against the project's premise -- that tensor networks capture
COMPOSITIONAL structure better than the bag-of-features behaviour which makes
frozen CLIP fail on ARO/SugarCrepe.

It is not, because NEITHER TASK IS COMPOSITIONAL:

  * C1/C3 attribute classification is pure perception -- one object, one label,
    nothing to bind.
  * C4 relations, as built, is "locate the off-centre blob, report its
    direction" -- spatial, but still a single-referent readout.

`MLPReference` flattens POSITIONED patch embeddings into a fully-connected trunk,
so it is position-aware and close to ideal for both. Neither task has the
property that breaks a bag-of-features model: THE SAME FEATURES ARRANGED TWO
WAYS, YIELDING TWO DIFFERENT LABELS. This module builds a task that does.

THE TASK
--------
One object of shape A and one of shape B per image; the label is WHICH ONE IS ON
THE LEFT. Two classes, chance 50%.

    class 0 : shape A left,  shape B right
    class 1 : shape B left,  shape A right

**The marginals are identical across classes by construction.** Every image
contains exactly one A and exactly one B, so the global shape content carries
zero information and only the BINDING of shape to side separates the classes. A
model using unbound features is at chance provably, not merely empirically --
which is what makes this a compositional probe rather than another perception
task dressed up as one.

WHY COMPOSITES AND NOT NATURAL CROPS -- THIS IS FORCED, NOT PREFERRED
---------------------------------------------------------------------
Two independent reasons, and the first one nearly got missed:

1. **Natural relation crops are CENTRED ON THE REFERENCE OBJECT** (see
   `clevr_objects.iter_relation_crops`). Building the binding task on them would
   reduce it to "what shape is in the middle?" -- perception again, in a new
   costume, and it would have looked like a binding experiment while testing
   nothing of the sort.
2. **Natural two-object crops are 3.2% of valid pairs** -- about 830 balanced
   crops from the entire 85k-scene atlas, far below the 1024/512 protocol.

Composition also removes distractor objects and referent ambiguity outright.
That ambiguity is what killed the first C4: 96.8% of its crops held a distractor
and in 87.9% at least one sat farther from the centre than the labelled object,
so the label named one object out of ~5 with nothing in the image saying which.

Cost: the composites are SYNTHETIC. State that as a limitation; the natural C4
relation task stands beside this one as the ecological-validity companion.

DESIGN CHOICES THAT CARRY THE TASK
----------------------------------
* **Placement is JITTERED, never aligned to the tree's 2x2 level-1 blocks.**
  Aligned placement would hand each level-1 block exactly one object -- precisely
  the TTN's claimed inductive bias. A reviewer would call that rigged, and they
  would be right.
* **Canvas is 32x32 with patch_size 8, NOT 16x16 with patch_size 4.** The patch
  grid stays 4x4, so the tree still uses **16 qubits** and the circuit is
  unchanged -- this is C5 route (a), already regression-tested by
  `test_bigger_patches_keep_the_tree_at_16_qubits`. The extra pixels buy room to
  place two objects without shrinking either: an object fills 0.25-0.50 of its
  source crop, so at a 14 px cell it spans ~4-7 px, against ~2-4 px if both were
  squeezed into a 16x16 canvas. Record it as resolution scaling, not quantum
  scaling: the circuit sees exactly as many wires as before.
* **The two cells cannot overlap**, enforced geometrically rather than hoped for.
* **Source crops keep their own CLEVR floor background**, so the composite reads
  as one scene rather than two cut-outs on a synthetic field.

WHICH TWO SHAPES?
-----------------
`shape` is stored as an integer and this module does NOT hardcode which integer
is a cube. Two distinct classes is all the task needs -- the names matter only
for the write-up. Build the montage, look at it, and record the mapping in
`research_log.md` rather than assuming it. Assuming a label mapping is exactly
the kind of silent error that yields a balanced, plausible-looking dataset and a
meaningless result -- the same class of mistake as the relation-axis error C0
caught only because someone looked at the pictures.

Run: conda run -n qnlp python -m qnlp.image_tower.classification.clevr.build_clevr_binding
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from qnlp.utils.data.clevr_objects import CACHE_DIR
from qnlp.utils.data.clevr_objects import cache_path as objects_cache_path

# Two DISTINCT shape classes. Not named here on purpose -- see the docstring.
DEFAULT_SHAPE_PAIR = (0, 1)

# Composite geometry. Defaults chosen so the two cells tile the canvas with a
# little slack for jitter and can never overlap; see `_cell_origins`.
CANVAS = 32
CELL = 14
JITTER_X = 2
JITTER_Y = 6
SOURCE_RES = 32  # Source object crops to resize down from, for the least loss.

BINDING_CLASSES = ("A_left", "B_left")


def _cell_origins(rng, canvas=CANVAS, cell=CELL, jitter_x=JITTER_X, jitter_y=JITTER_Y):
    """Top-left corners for the left and right cells, jittered, never overlapping.

    The left cell is confined to [0, canvas/2 - cell] and the right one starts at
    or after canvas/2, so `left_x + cell <= canvas/2 <= right_x` holds for every
    draw. Overlap is impossible by arithmetic rather than by luck -- two objects
    bleeding into each other would make "which is on the left" ambiguous, and
    ambiguity of exactly that kind is what invalidated the first C4.
    """
    half = canvas // 2
    lx_max, rx_min, rx_max = half - cell, half, canvas - cell
    lx = int(rng.integers(max(0, lx_max - jitter_x), lx_max + 1))
    rx = int(rng.integers(rx_min, min(rx_max, rx_min + jitter_x) + 1))
    y_mid = (canvas - cell) // 2
    ly = int(rng.integers(max(0, y_mid - jitter_y), min(canvas - cell, y_mid + jitter_y) + 1))
    ry = int(rng.integers(max(0, y_mid - jitter_y), min(canvas - cell, y_mid + jitter_y) + 1))
    return (lx, ly), (rx, ry)


def _resize(crop, cell):
    return np.asarray(Image.fromarray(crop).resize((cell, cell), Image.BILINEAR), dtype=np.uint8)


def build_composites(
    images,
    shapes,
    n,
    shape_pair=DEFAULT_SHAPE_PAIR,
    seed=0,
    canvas=CANVAS,
    cell=CELL,
    jitter_x=JITTER_X,
    jitter_y=JITTER_Y,
):
    """Compose `n` two-object images, exactly balanced over the two classes.

    Returns (images [n, canvas, canvas, 3] uint8, labels [n] int64, meta).

    Every composite holds exactly one object of each shape in `shape_pair`, so
    the class-conditional shape marginals are identical BY CONSTRUCTION. That is
    the whole point: it makes the bag-of-features shortcut provably worthless
    rather than merely unlikely, so a model scoring above chance must be using
    the binding of shape to side.
    """
    a, b = shape_pair
    if a == b:
        raise ValueError(f"shape_pair must be two DISTINCT classes, got {shape_pair}")
    idx_a = np.flatnonzero(np.asarray(shapes) == a)
    idx_b = np.flatnonzero(np.asarray(shapes) == b)
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError(f"shape classes {shape_pair} are not both present (found {len(idx_a)}, {len(idx_b)})")

    rng = np.random.default_rng(seed)
    # Exactly balanced: alternate the classes rather than sampling them, so the
    # 50% floor is exact and no model can beat chance on the prior alone.
    labels = np.tile([0, 1], n // 2 + 1)[:n]
    out = np.zeros((n, canvas, canvas, 3), dtype=np.uint8)

    for i, lab in enumerate(labels):
        ia = int(rng.choice(idx_a))
        ib = int(rng.choice(idx_b))
        # label 0 = shape A on the left; label 1 = shape B on the left.
        left, right = (ia, ib) if lab == 0 else (ib, ia)
        (lx, ly), (rx, ry) = _cell_origins(rng, canvas, cell, jitter_x, jitter_y)
        out[i, ly : ly + cell, lx : lx + cell] = _resize(images[left], cell)
        out[i, ry : ry + cell, rx : rx + cell] = _resize(images[right], cell)

    meta = {
        "shape_pair": [int(a), int(b)],
        "canvas": canvas,
        "cell": cell,
        "jitter_x": jitter_x,
        "jitter_y": jitter_y,
        "classes": list(BINDING_CLASSES),
        "n_source_a": int(len(idx_a)),
        "n_source_b": int(len(idx_b)),
    }
    return out, labels.astype(np.int64), meta


def binding_cache_path(img_size, split):
    return os.path.join(CACHE_DIR, f"clevr_binding_{img_size}_{split}.npz")


def build_split(split, n, shape_pair=DEFAULT_SHAPE_PAIR, seed=0, source_res=SOURCE_RES, **geom):
    """Build one split from the cached single-object crops.

    Source crops come from `clevr_objects_<source_res>_<split>.npz`, so train and
    val inherit their DIFFERENT CLEVR shards and no scene is shared -- the
    property `test_train_and_val_come_from_different_scenes` guards for objects.
    """
    path = objects_cache_path("objects", source_res, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build the object crops first:\n"
            f"  conda run -n qnlp python -m qnlp.image_tower.classification.clevr.build_clevr_crops "
            f"--tasks objects"
        )
    with np.load(path) as z:
        images, shapes = z["images"], z["shape"]
    return build_composites(images, shapes, n, shape_pair=shape_pair, seed=seed, **geom)


def save_split(split, images, labels, canvas=CANVAS):
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = binding_cache_path(canvas, split)
    np.savez_compressed(p, images=images, binding=labels)
    return p


def shuffle_patches(x, patch_size, generator=None):
    """Randomly permute the patch grid of each image INDEPENDENTLY.

    THIS IS A MANIPULATION CHECK ON THE TASK, NOT A PER-ARCHITECTURE SCORE.
    An earlier version of the C6 plan proposed `acc(normal) - acc(shuffled)` as a
    compositionality score per architecture. That is wrong here: because the
    marginals are controlled, shuffling destroys the only signal present, so
    EVERY architecture must land at exactly chance and the "score" would be
    identical by construction.

    What it does test is the DATA: if any arm scores resolvably above chance on
    shuffled input, some non-positional cue has been baked into the composites --
    a lighting difference, an intensity gradient, a pasting artifact correlated
    with the class. That invalidates the task and it must be rebuilt before any
    result is read. Given this phase's history with tasks that looked fine and
    were not, that check is worth its (negligible) cost.

    `x` is [B, C, S, S]; the permutation is a genuine permutation of patches, so
    the pixel multiset of each image is preserved exactly.
    """
    b, c, s, _ = x.shape
    g = s // patch_size
    p = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    p = p.permute(0, 2, 3, 1, 4, 5).reshape(b, g * g, c, patch_size, patch_size)
    perm = torch.argsort(torch.rand(b, g * g, generator=generator), dim=1)
    p = torch.gather(p, 1, perm[:, :, None, None, None].expand_as(p))
    p = p.reshape(b, g, g, c, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5)
    return p.reshape(b, c, s, s)


def get_binding_loaders(
    img_size=CANVAS,
    batch_size=32,
    train_samples=1024,
    test_samples=512,
    seed=42,
    shuffled=False,
    patch_size=8,
):
    """Mirrors `clevr_objects.get_clevr_loaders`, one head named `binding`.

    `shuffled=True` applies the patch-shuffle manipulation check to BOTH splits.
    """
    loaders = []
    for split, n in (("train", train_samples), ("val", test_samples)):
        path = binding_cache_path(img_size, split)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Build it first:\n"
                f"  conda run -n qnlp python -m qnlp.image_tower.classification.clevr.build_clevr_binding"
            )
        with np.load(path) as z:
            images, labels = z["images"], z["binding"]
        total = images.shape[0]
        if n > total:
            raise ValueError(f"asked for {n} {split} samples but the cache holds {total}")
        pick = np.random.default_rng(seed if split == "train" else seed + 10_000).permutation(total)[:n]
        x = torch.from_numpy(images[pick]).float().permute(0, 3, 1, 2) / 255.0
        if shuffled:
            x = shuffle_patches(x, patch_size, generator=torch.Generator().manual_seed(seed))
        y = torch.from_numpy(labels[pick]).long()
        loaders.append(DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=(split == "train")))

    def collate(loader):
        for imgs, lab in loader:
            yield imgs, {"binding": lab}

    return _Wrapped(loaders[0], collate), _Wrapped(loaders[1], collate)


class _Wrapped:
    """Keeps `len()` working while re-shaping batches into (images, label dict)."""

    def __init__(self, loader, collate):
        self._loader, self._collate = loader, collate

    def __iter__(self):
        return iter(self._collate(self._loader))

    def __len__(self):
        return len(self._loader)


def load_binding_manifest():
    with open(os.path.join(CACHE_DIR, "clevr_binding_manifest.json")) as f:
        return json.load(f)
