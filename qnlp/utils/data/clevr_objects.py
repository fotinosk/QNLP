"""CLEVR object-crop datasets for the Phase 2 (CLEVR) image-tower experiments.

WHY CROPS, AND NOT WHOLE SCENES
-------------------------------
The Phase 2 plan (quantum_investigation_roadmap.md Section 3, task C0) says
"filter to scenes with exactly one object" and "... exactly two objects".
**No such scenes exist.** CLEVR renders 3-10 objects per scene by construction,
verified against `dpdl-benchmark/clevr` on HuggingFace (2026-07-29); both filters
return zero rows.

So the single-object and two-object datasets are built by CROPPING objects out of
full scenes, using the per-object `pixel_coords = [x, y, depth]` the parquet
already carries. This also sidesteps the resolution warning in
quantum_implementation_plan.md:79, which assumed whole 480x320 scenes downsampled
to 16x16 (where an object is a handful of pixels and `material` is hopeless).
A crop puts one object in the frame instead.

THE CROP MUST BE WORLD-SCALED, NOT TIGHT
----------------------------------------
The crop side is `CROP_K / depth`, never the object's own apparent extent. This
is the single most important design decision in this module and it exists to keep
the `size` head learnable:

  apparent radius (px)  =  F_PX * r_world / depth
  crop side      (px)  =  CROP_K / depth
  ------------------------------------------------ divide
  fraction of the box the object fills = 2 * F_PX * r_world / CROP_K

The depth cancels. A large object (r=0.70) therefore fills the same fraction of
its box wherever it sits in the scene, and a small one (r=0.35) fills exactly
half that fraction -- so "how much of the frame is filled" is a clean, distance-
invariant cue for `size`. A tight bounding-box crop would normalise that cue away
and make the `size` head unlearnable by construction.

CROP_K IS CALIBRATED ONCE AND FROZEN
------------------------------------
See `build_clevr_crops.py --calibrate`. Do not re-tune it per experiment; a
moving data definition is how `classical_bare` acquired three different
"measurements" in Phase 1.

Run: conda run -n qnlp python -m qnlp.image_tower.classification.clevr.build_clevr_crops
"""

import io
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

CLEVR_REPO = "dpdl-benchmark/clevr"
# One shard each is plenty: ~3700 train scenes x 3-10 objects is tens of
# thousands of crops, against a 1024-sample training protocol. The full dataset
# is 11.4 GB and there is no reason to pull it.
TRAIN_SHARD = "data/train-00000-of-00019.parquet"
TEST_SHARD = "data/test-00000-of-00005.parquet"

IMG_W, IMG_H = 480, 320

# Blender's default camera as CLEVR configures it: 35 mm lens on a 32 mm sensor,
# rendered 480 px wide. f_px = 35/32 * 480. Used only to reason about crop
# geometry (and to assert it in tests); nothing depends on it at runtime.
F_PX = 35.0 / 32.0 * IMG_W  # 525.0

# Crop side in pixels at unit depth: side_px = CROP_K / depth.
# Chosen so a large object (r_world = 0.7) fills ~50% of its box and a small one
# (r_world = 0.35) ~25%: CROP_K = 2 * F_PX * 0.7 / 0.5 = 1470.
# CALIBRATED AND FROZEN 2026-07-30 -- see the module docstring.
CROP_K = 1470.0

# CLEVR's canonical direction vectors (render_images.py). These are the CAMERA-
# rotated axes, not the world axes.
#
# NOTE: `quantum_implementation_plan.md:173-180` proposes comparing raw
# `3d_coords` x and y. That is WRONG for CLEVR and mislabels a large fraction of
# pairs -- the scene's left/right axis is rotated ~49 degrees from world x.
DIR_RIGHT = np.array([0.6563112735748291, 0.7544902563095093, 0.0])
DIR_FRONT = np.array([0.7544902563095093, -0.6563112735748291, 0.0])

ATTRIBUTES = {"color": 8, "shape": 3, "material": 2, "size": 2}
RELATIONS = ("left", "right", "front", "behind")

RESOLUTIONS = (16, 32, 64)
CACHE_DIR = "data/datasets"

# An object is dropped if a NEARER object's centre falls within this fraction of
# its crop side -- i.e. something is standing in front of it. CLEVR scenes are
# dense, so without this a "single-object" crop routinely contains two objects
# and the labels are simply wrong.
OCCLUSION_FRAC = 0.5
# Reject a crop whose box runs more than this fraction of its side outside the
# frame; anything less is padded by edge replication.
MAX_OUT_OF_FRAME = 0.10
# A relation is kept only if the dominant axis beats the other by this factor.
# Near-diagonal pairs have no defensible left-vs-front label, and including them
# would put a ceiling on accuracy that has nothing to do with the model.
RELATION_MARGIN = 1.5
# Widest relation crop, in source pixels. A pair separated by more than this
# would shrink both objects to a few pixels once resized, so accuracy would
# measure resolution rather than relational reasoning.
#
# Both relation limits are set from the measured distribution of required crop
# sides (median 456 px over 1246 valid pairs), not guessed. Centring on the
# reference object means the box must reach past the second object, so it is
# about twice as wide as the pair's own extent -- and the source frame is only
# 320 px tall, which makes the out-of-frame allowance the binding constraint.
# 480 px / 0.25 keeps ~40% of valid pairs; tightening to 0.10 keeps 8% and
# starves the dataset.
MAX_RELATION_SIDE = 480.0
# Relation crops are allowed more edge padding than object crops. The padded
# region replicates the CLEVR floor, which is a smooth grey gradient, so this is
# visually benign -- but it IS a deviation and is recorded in the manifest.
MAX_OUT_OF_FRAME_RELATION = 0.25


def decode_image(image_field):
    """HF parquet stores images as a struct {bytes, path} or as raw bytes."""
    raw = image_field["bytes"] if isinstance(image_field, dict) else image_field
    return Image.open(io.BytesIO(raw)).convert("RGB")


def crop_side_px(depth):
    return CROP_K / float(depth)


def fill_fraction(r_world):
    """Fraction of the crop box an object of this world radius occupies, at ANY
    depth. Depth-independent by construction -- this is what makes `size`
    learnable from a crop. `r_world` is `3d_coords[2]`: CLEVR stores an object's
    z-centre, which equals its radius since objects rest on the floor.
    """
    return 2.0 * F_PX * float(r_world) / CROP_K


def is_occluded(idx, pixel_coords):
    """Is object `idx` hidden by something closer to the camera?"""
    px, py, depth = pixel_coords[idx]
    reach = OCCLUSION_FRAC * crop_side_px(depth)
    for j, (qx, qy, qdepth) in enumerate(pixel_coords):
        if j == idx:
            continue
        if qdepth < depth and np.hypot(qx - px, qy - py) < reach:
            return True
    return False


def _crop_box(cx, cy, side):
    half = side / 2.0
    return cx - half, cy - half, cx + half, cy + half


def _out_of_frame_frac(box, side):
    x0, y0, x1, y1 = box
    over = max(0.0, -x0, -y0, x1 - IMG_W, y1 - IMG_H)
    return over / side


def _crop_and_resize(img, box, resolutions):
    """Crop, replicating edge pixels where the box leaves the frame, then resize
    to each requested resolution."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    arr = np.asarray(img)
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - IMG_W), max(0, y1 - IMG_H)
    if pad_l or pad_t or pad_r or pad_b:
        arr = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
        x0, x1, y0, y1 = x0 + pad_l, x1 + pad_l, y0 + pad_t, y1 + pad_t
    patch = Image.fromarray(arr[y0:y1, x0:x1])
    return {s: np.asarray(patch.resize((s, s), Image.BILINEAR), dtype=np.uint8) for s in resolutions}


def relation_label(coords_a, coords_b, margin=RELATION_MARGIN):
    """Spatial relation of b relative to a, in CLEVR's camera-rotated basis.

    Returns None for near-diagonal pairs, where neither axis dominates and the
    label would be arbitrary.
    """
    d = np.asarray(coords_b, dtype=float) - np.asarray(coords_a, dtype=float)
    u = float(d @ DIR_RIGHT)  # +right / -left
    v = float(d @ DIR_FRONT)  # +front / -behind
    hi, lo = (abs(u), abs(v)) if abs(u) >= abs(v) else (abs(v), abs(u))
    if lo > 0 and hi < margin * lo:
        return None
    if abs(u) >= abs(v):
        return "right" if u > 0 else "left"
    return "front" if v > 0 else "behind"


def iter_object_crops(row, resolutions=RESOLUTIONS):
    """Yield (crops_by_resolution, label_dict) for each unoccluded object."""
    objects = row["objects"]
    pixel_coords = [tuple(map(float, p)) for p in objects["pixel_coords"]]
    img = None
    for i in range(len(objects["color"])):
        if is_occluded(i, pixel_coords):
            continue
        cx, cy, depth = pixel_coords[i]
        side = crop_side_px(depth)
        box = _crop_box(cx, cy, side)
        if _out_of_frame_frac(box, side) > MAX_OUT_OF_FRAME:
            continue
        if img is None:
            img = decode_image(row["image"])
        labels = {a: int(objects[a][i]) for a in ATTRIBUTES}
        yield _crop_and_resize(img, box, resolutions), labels


def iter_relation_crops(row, resolutions=RESOLUTIONS, rng=None, max_pairs=2):
    """Yield (crops_by_resolution, {"relation": idx}) for unambiguous pairs.

    THE REFERENCE OBJECT IS THE CENTRED ONE. The label is "where is the other
    object relative to the object in the middle of the frame", and the crop is
    centred on the reference exactly as single-object crops are.

    This is not cosmetic, it is what makes the task well-posed. A crop that
    merely contains two objects carries no information about which is the
    reference, so "b is left of a" and "a is right of b" would be the same
    picture with opposite labels -- an unlearnable 50% of the dataset. Centring
    resolves it, and it lets the (a, b) order be chosen at random, which is what
    keeps all four relations equally frequent. Any deterministic ordering rule
    collapses the task instead: ordering by screen position makes left/right
    almost fully predictable from the convention, and ordering by depth does the
    same to front/behind.

    At most `max_pairs` per scene, sampled, so one dense scene cannot dominate
    the dataset with its O(n^2) pairs.
    """
    objects = row["objects"]
    pixel_coords = [tuple(map(float, p)) for p in objects["pixel_coords"]]
    coords_3d = [tuple(map(float, c)) for c in objects["3d_coords"]]
    n = len(objects["color"])
    keep = [i for i in range(n) if not is_occluded(i, pixel_coords)]
    rng = rng or np.random.default_rng(0)

    candidates = []
    for ai in range(len(keep)):
        for bi in range(ai + 1, len(keep)):
            a, b = keep[ai], keep[bi]
            if rng.random() < 0.5:  # random reference, so all four labels occur
                a, b = b, a
            rel = relation_label(coords_3d[a], coords_3d[b])
            if rel is not None:
                candidates.append((a, b, rel))
    if not candidates:
        return
    idxs = rng.permutation(len(candidates))[:max_pairs]

    img = None
    for k in idxs:
        a, b, rel = candidates[int(k)]
        ax, ay, adepth = pixel_coords[a]
        bx, by, bdepth = pixel_coords[b]
        sa, sb = crop_side_px(adepth), crop_side_px(bdepth)
        # Centred on the reference `a`, wide enough to contain all of `b`.
        reach = np.hypot(bx - ax, by - ay) + sb / 2.0
        side = 2.0 * max(reach, sa / 2.0) * 1.1
        if side > MAX_RELATION_SIDE:
            # Far-apart pairs would shrink both objects to a few pixels; that
            # measures resolution, not relational reasoning.
            continue
        box = _crop_box(ax, ay, side)
        if _out_of_frame_frac(box, side) > MAX_OUT_OF_FRAME_RELATION:
            continue
        if img is None:
            img = decode_image(row["image"])
        yield _crop_and_resize(img, box, resolutions), {"relation": RELATIONS.index(rel)}


# ---------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------


def cache_path(task, img_size, split):
    return os.path.join(CACHE_DIR, f"clevr_{task}_{img_size}_{split}.npz")


def save_split(task, split, images_by_res, labels):
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = []
    for res, imgs in images_by_res.items():
        path = cache_path(task, res, split)
        np.savez_compressed(
            path, images=np.stack(imgs), **{k: np.asarray(v, dtype=np.int64) for k, v in labels.items()}
        )
        paths.append(path)
    return paths


def class_balance(labels, task):
    """Per-attribute class counts. `material` and `size` are binary and could be
    near-degenerate; a head sitting at its majority-class rate looks like a
    result and is not one."""
    card = ATTRIBUTES if task == "objects" else {"relation": len(RELATIONS)}
    out = {}
    for name, k in card.items():
        counts = np.bincount(np.asarray(labels[name]), minlength=k)
        out[name] = {
            "counts": counts.tolist(),
            "majority_rate": float(counts.max() / max(counts.sum(), 1)),
        }
    return out


def balance_classes(labels, task, seed=0):
    """Subsample to equal counts per class on the task's primary label.

    For relations this is required (the plan says so): an unbalanced 4-way
    relation set lets a model score above the 25% chance floor without learning
    anything spatial.
    """
    key = "relation" if task == "relations" else None
    if key is None:
        return None
    y = np.asarray(labels[key])
    rng = np.random.default_rng(seed)
    per = min(int((y == c).sum()) for c in range(len(RELATIONS)))
    idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per] for c in range(len(RELATIONS))])
    return np.sort(rng.permutation(idx))


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------


def get_clevr_loaders(
    task="objects",
    img_size=16,
    batch_size=32,
    train_samples=1024,
    test_samples=512,
    seed=42,
):
    """Mirrors `synthetic_shapes.get_synthetic_shapes_loaders`, but each batch is
    `(images [B,3,S,S] float, labels dict[str, LongTensor])`.

    `seed` selects WHICH cached crops are used (a fresh subsample per seed), so
    seeds vary the data draw as well as the initialisation -- matching how the
    synthetic-shapes loader behaves and keeping seed-to-seed variance honest.
    Train and val come from different CLEVR shards, so no scene is shared.
    """
    heads = list(ATTRIBUTES) if task == "objects" else ["relation"]
    loaders = []
    for split, n in (("train", train_samples), ("val", test_samples)):
        path = cache_path(task, img_size, split)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Build it first:\n"
                f"  conda run -n qnlp python -m qnlp.image_tower.classification.clevr.build_clevr_crops"
            )
        with np.load(path) as z:
            images, labels = z["images"], {h: z[h] for h in heads}
        total = images.shape[0]
        if n > total:
            raise ValueError(f"asked for {n} {split} samples but the cache holds {total}")
        pick = np.random.default_rng(seed if split == "train" else seed + 10_000).permutation(total)[:n]
        x = torch.from_numpy(images[pick]).float().permute(0, 3, 1, 2) / 255.0
        ys = [torch.from_numpy(labels[h][pick]).long() for h in heads]
        ds = TensorDataset(x, *ys)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=(split == "train")))

    def collate(loader):
        for batch in loader:
            yield batch[0], dict(zip(heads, batch[1:]))

    return _Wrapped(loaders[0], collate), _Wrapped(loaders[1], collate)


class _Wrapped:
    """Keeps `len()` working while re-shaping batches into (images, label dict)."""

    def __init__(self, loader, collate):
        self._loader, self._collate = loader, collate

    def __iter__(self):
        return iter(self._collate(self._loader))

    def __len__(self):
        return len(self._loader)


def load_manifest(task):
    path = os.path.join(CACHE_DIR, f"clevr_{task}_manifest.json")
    with open(path) as f:
        return json.load(f)
