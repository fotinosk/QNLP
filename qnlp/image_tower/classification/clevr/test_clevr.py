"""Guards for the CLEVR (Phase 2) data pipeline and harness.

Like test_qttn_core.py, every test here targets a failure this project either
shipped or came close to shipping:

  test_size_cue_survives_the_crop
      The whole reason crops are world-scaled rather than tight. A tight bounding
      box would normalise apparent size away and make the `size` head unlearnable
      by construction -- a data bug that would have read as an architecture
      result.

  test_relation_reference_object_is_centred
      Without a centred reference, "b is left of a" and "a is right of b" are the
      same picture with opposite labels, so half the relation dataset would be
      unlearnable noise and Question C.2 would get a fake null.

  test_relation_partner_is_the_nearest_object_not_an_arbitrary_one
      The same ambiguity for the OTHER object, which the test above does not
      cover and which is what actually killed C4: a randomly chosen partner
      among ~5 objects in frame is not identifiable from the image, so the task
      returns chance however good the model is.

  test_per_head_chance_guard_uses_each_heads_own_classes
      pc.is_chance_level defaults to 4 classes. Applied to the binary `material`
      head it would wave through a completely dead 50% head. R4's 25.8% arm
      nearly produced a fake 57.8-pt quantum-advantage headline the same way.

  test_train_and_val_come_from_different_scenes
      Train and val are built from different CLEVR shards. If that ever breaks,
      every accuracy in Phase 2 is inflated.

  test_binding_composites_contain_exactly_one_of_each_shape
      C6's whole validity. Identical class-conditional marginals are what make a
      bag-of-features shortcut provably worthless; lose that and the task
      silently degrades into another perception task, which is the very thing
      C6 exists to escape.

Data-backed tests skip when the cache is absent; build it with
  conda run -n qnlp python -m qnlp.image_tower.classification.clevr.build_clevr_crops

Run: conda run -n qnlp python -m pytest qnlp/image_tower/classification/clevr/test_clevr.py -q
"""

import io
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from qnlp.image_tower.classification.clevr import clevr_common as cc
from qnlp.image_tower.classification.quantum.qttn_core import CoherentQTTNClassifier
from qnlp.utils.data import clevr_binding as cb
from qnlp.utils.data import clevr_objects as co

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_HAS_OBJECTS = os.path.exists(co.cache_path("objects", 16, "train"))
_HAS_RELATIONS = os.path.exists(co.cache_path("relations", 16, "train"))
needs_objects = pytest.mark.skipif(not _HAS_OBJECTS, reason="run build_clevr_crops first")
needs_relations = pytest.mark.skipif(not _HAS_RELATIONS, reason="run build_clevr_crops first")


# ---------------------------------------------------------------------
# Crop geometry
# ---------------------------------------------------------------------


def test_size_cue_survives_the_crop():
    """A large object must fill measurably more of its box than a small one, and
    the fraction must NOT depend on depth.

    Depth-invariance is the whole point of scaling the crop by CROP_K / depth:
    it makes "how much of the frame is filled" a clean distance-invariant cue for
    `size`. If someone later switches to a tight bounding box, this fails.
    """
    small, large = co.fill_fraction(0.35), co.fill_fraction(0.70)
    assert large > small * 1.9, f"size cue too weak: small={small:.3f} large={large:.3f}"
    assert 0.15 < small < 0.40 and 0.35 < large < 0.75, (
        f"objects fill {small:.2f}/{large:.2f} of the crop -- re-run --calibrate; "
        f"too small wastes resolution, too large clips the object"
    )
    # Depth-invariance: the crop side and the apparent radius both scale as
    # 1/depth, so their ratio is constant.
    for depth in (7.0, 11.0, 16.0):
        apparent = 2 * co.F_PX * 0.70 / depth
        assert abs(apparent / co.crop_side_px(depth) - large) < 1e-9


def test_occlusion_filter_only_rejects_objects_hidden_by_nearer_ones():
    """Something BEHIND the target must not disqualify it -- CLEVR scenes are
    dense and rejecting on any neighbour would empty the dataset."""
    # (x, y, depth); object 0 is the target at depth 10, side = 147 px.
    behind = [(240.0, 160.0, 10.0), (250.0, 160.0, 12.0)]
    infront = [(240.0, 160.0, 10.0), (250.0, 160.0, 8.0)]
    faraway = [(240.0, 160.0, 10.0), (450.0, 160.0, 8.0)]
    assert not co.is_occluded(0, behind), "a farther object must not occlude"
    assert co.is_occluded(0, infront), "a nearer object inside the box must occlude"
    assert not co.is_occluded(0, faraway), "a nearer object outside the box must not occlude"


def test_relation_labels_use_clevrs_rotated_axes_not_world_axes():
    """quantum_implementation_plan.md:173-180 proposes comparing raw 3d_coords x
    and y. CLEVR's left/right is ~49 degrees from world x, so that helper
    mislabels a large fraction of pairs. Pin the correct behaviour."""
    origin = (0.0, 0.0, 0.35)
    assert co.relation_label(origin, tuple(2 * co.DIR_RIGHT)) == "right"
    assert co.relation_label(origin, tuple(-2 * co.DIR_RIGHT)) == "left"
    assert co.relation_label(origin, tuple(2 * co.DIR_FRONT)) == "front"
    assert co.relation_label(origin, tuple(-2 * co.DIR_FRONT)) == "behind"
    # Antisymmetry: swapping the pair must flip the label.
    a, b = (0.0, 0.0, 0.35), (1.5, 1.9, 0.35)
    assert co.relation_label(a, b) != co.relation_label(b, a)
    # Near-diagonal pairs are dropped rather than labelled arbitrarily.
    diag = tuple(np.array(co.DIR_RIGHT) + np.array(co.DIR_FRONT))
    assert co.relation_label(origin, diag) is None

    # The naive world-axis helper really does disagree -- this is the bug, pinned.
    def naive(a, b):
        return (
            ("left" if a[0] < b[0] else "right")
            if abs(a[0] - b[0]) > abs(a[1] - b[1])
            else ("front" if a[1] < b[1] else "behind")
        )

    assert naive(origin, tuple(2 * co.DIR_RIGHT)) != "right"


# ---------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------


@needs_objects
@pytest.mark.parametrize("img_size", [16, 32, 64])
def test_object_loader_shapes_and_label_ranges(img_size):
    train, val = cc.get_clevr_loaders(task="objects", img_size=img_size, train_samples=64, test_samples=64)
    for loader in (train, val):
        imgs, labels = next(iter(loader))
        assert imgs.shape[1:] == (3, img_size, img_size)
        assert imgs.min() >= 0.0 and imgs.max() <= 1.0
        assert set(labels) == set(cc.HEADS)
        for head, k in cc.HEADS.items():
            assert labels[head].dtype == torch.long
            assert int(labels[head].min()) >= 0 and int(labels[head].max()) < k


@needs_relations
def test_relation_loader_is_balanced_and_four_way():
    _, val = cc.get_clevr_loaders(task="relations", img_size=16, train_samples=64, test_samples=512)
    counts = np.zeros(4, dtype=int)
    for _, labels in val:
        counts += np.bincount(labels["relation"].numpy(), minlength=4)
    rate = counts.max() / counts.sum()
    assert rate < 0.30, (
        f"relation classes are unbalanced (majority {rate:.2f}); a model could beat the 25% "
        f"chance floor without learning anything spatial"
    )


@needs_relations
def test_relation_reference_object_is_centred():
    """The label is 'relative to the object in the middle', so there must BE an
    object in the middle. Measured by colour saturation: CLEVR's floor is grey,
    its objects are not."""
    _, val = cc.get_clevr_loaders(task="relations", img_size=64, train_samples=64, test_samples=256)
    imgs = torch.cat([b[0] for b in val])
    mx, mn = imgs.max(dim=1).values, imgs.min(dim=1).values
    sat = (mx - mn) / mx.clamp(min=1e-6)
    centre = sat[:, 30:34, 30:34].mean()
    corner = torch.cat([sat[:, :4, :4].flatten(), sat[:, -4:, -4:].flatten()]).mean()
    assert centre > corner * 1.5, (
        f"crop centre (saturation {centre:.3f}) is not more object-like than the corners "
        f"({corner:.3f}) -- the reference object is not centred, so the relation labels are "
        f"ambiguous and half the dataset is unlearnable"
    )


def test_relation_partner_is_the_nearest_object_not_an_arbitrary_one():
    """THE DEFECT THAT KILLED C4, pinned.

    The partner used to be sampled at random from all valid pairs, with nothing
    checking what else was in the box: 96.8% of crops held a distractor (median
    4), and in 87.9% at least one distractor sat FARTHER from the centre than the
    labelled object. So the label named one object out of ~5 and nothing in the
    image said which -- a model that understands relations perfectly still scores
    chance, and C4 duly measured 28.4% against a 29.3% floor.

    Scene below: reference `a` centred, `b` near and to the RIGHT, `c` far and in
    FRONT. Under the nearest rule `a`'s partner is always `b`, so "front" can
    never be emitted. Under the old random rule it could be, half the time.
    """
    a_px, b_px, c_px = (240.0, 160.0, 10.0), (300.0, 160.0, 10.0), (240.0, 60.0, 10.0)
    assert np.hypot(*(np.subtract(b_px[:2], a_px[:2]))) < np.hypot(*(np.subtract(c_px[:2], a_px[:2])))
    buf = io.BytesIO()
    Image.fromarray(np.zeros((co.IMG_H, co.IMG_W, 3), dtype=np.uint8)).save(buf, format="PNG")
    row = {
        "image": {"bytes": buf.getvalue()},
        "objects": {
            "color": [0, 1, 2],  # only its length is read
            "pixel_coords": [a_px, b_px, c_px],
            "3d_coords": [(0.0, 0.0, 0.35), tuple(2 * co.DIR_RIGHT), tuple(2 * co.DIR_FRONT)],
        },
    }
    labels = {co.RELATIONS[lab["relation"]] for _, lab in co.iter_relation_crops(row, resolutions=(16,))}
    assert "front" not in labels and "behind" not in labels, (
        f"emitted {sorted(labels)} -- the label refers to an object that is NOT the nearest one, "
        f"so nothing in the crop identifies which object it means. This is the C4 defect."
    )
    assert labels == {"right", "left"}, f"expected the a<->b pair in both directions, got {sorted(labels)}"


@needs_relations
def test_relation_cache_was_built_with_the_nearest_partner_rule():
    """A cache built before 2026-08-01 has an ambiguous referent and will return a
    null no matter what the model does. The manifest is the only way to tell the
    two apart -- the .npz files look identical."""
    manifest = json.load(open(os.path.join(co.CACHE_DIR, "clevr_relations_manifest.json")))
    assert manifest.get("relation_partner_rule") == co.RELATION_PARTNER_RULE, (
        f"relation cache at {co.CACHE_DIR} was built with partner rule "
        f"{manifest.get('relation_partner_rule', 'random (pre-2026-08-01)')!r} -- rebuild it with "
        f"build_clevr_crops before running C4, or the task is not learnable by construction"
    )


# ---------------------------------------------------------------------
# Task C6 -- shape-binding composites
# ---------------------------------------------------------------------


def _fake_object_crops(n=40, res=32, seed=0):
    """Three attribute classes of noise blobs, enough to compose from."""
    rng = np.random.default_rng(seed)
    imgs = rng.integers(0, 255, size=(n, res, res, 3), dtype=np.uint8)
    attrs = np.tile([0, 1, 2], n // 3 + 1)[:n]
    return imgs, attrs


def test_binding_composites_contain_exactly_one_of_each_class():
    """THE PROPERTY THE WHOLE TASK RESTS ON.

    Every composite holds exactly one object of each of the two attribute
    classes, so the class-conditional marginals are IDENTICAL and a bag-of-features
    model is at chance provably, not merely empirically. If this ever breaks,
    C6 stops being a compositional probe and silently becomes another perception
    task -- which is the exact failure C6 was created to escape, since neither
    C3 nor C4 turned out to test binding.
    """
    imgs, attrs = _fake_object_crops()
    out, labels, meta = cb.build_composites(imgs, attrs, n=64, attr_pair=(0, 1), seed=0)
    assert out.shape == (64, cb.CANVAS, cb.CANVAS, 3)
    counts = np.bincount(labels, minlength=2)
    assert counts[0] == counts[1] == 32, f"classes must be exactly balanced, got {counts.tolist()}"
    assert meta["attr_pair"] == [0, 1]
    with pytest.raises(ValueError):
        cb.build_composites(imgs, attrs, n=8, attr_pair=(1, 1))  # must be distinct


def test_binding_cache_paths_are_scoped_per_attribute():
    """The size and shape datasets must not collide.

    The first C6 build bound on `shape` and is confounded -- two of four arms sit
    at the shape floor in C3, so their binding scores measured perception. It is
    kept as a footnote, which requires it to stay reachable rather than be
    silently overwritten by the size rebuild.
    """
    a = cb.binding_cache_path(32, "train", "size")
    b = cb.binding_cache_path(32, "train", "shape")
    assert a != b and "size" in a and "shape" in b
    assert cb.binding_manifest_path("size") != cb.binding_manifest_path("shape")


def test_binding_default_attribute_is_one_every_arm_can_perceive():
    """A binding task is perception AND binding. If an arm cannot see the
    attribute on a single object, its failure says nothing about binding -- the
    C4 error, repeated. C3 single-object accuracy is the table that matters:
    size 84.8-99.1 across all arms, shape 36.5-64.5 with two arms at the 35.4
    floor. So the default must be `size`, and this pins it."""
    assert cb.DEFAULT_ATTRIBUTE == "size"


def test_binding_cells_never_overlap():
    """Two objects bleeding into each other would make 'which is on the left'
    ambiguous -- the same class of defect that invalidated the first C4, where
    the label named one object out of ~5 with nothing identifying which."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        (lx, _), (rx, _) = cb._cell_origins(rng)
        assert lx + cb.CELL <= rx, f"cells overlap: left ends at {lx + cb.CELL}, right starts at {rx}"
        assert 0 <= lx and rx + cb.CELL <= cb.CANVAS


def test_binding_placement_is_jittered_not_fixed():
    """Aligned placement would hand each level-1 block exactly one object -- the
    TTN's claimed inductive bias, handed to it for free. Jitter is what keeps the
    comparison honest, so pin that it actually varies."""
    rng = np.random.default_rng(0)
    origins = {cb._cell_origins(rng)[0] for _ in range(200)}
    assert len(origins) > 1, "left-cell placement is constant -- the jitter is not doing anything"


def test_binding_patch_shuffle_is_a_permutation():
    """The manipulation check must PERMUTE patches, not corrupt them: the pixel
    multiset has to be preserved exactly, or a drop to chance would just mean the
    image was destroyed rather than that the task needs position."""
    x = torch.rand(4, 3, 32, 32)
    y = cb.shuffle_patches(x, patch_size=8, generator=torch.Generator().manual_seed(0))
    assert y.shape == x.shape
    assert torch.allclose(x.flatten().sort().values, y.flatten().sort().values)
    assert not torch.allclose(x, y), "shuffle changed nothing"


@needs_objects
def test_train_and_val_come_from_different_scenes():
    manifest = json.load(open(os.path.join(co.CACHE_DIR, "clevr_objects_manifest.json")))
    assert manifest["source"]["train_shard"] != manifest["source"]["test_shard"]
    assert manifest["crop_k"] == co.CROP_K, "cache was built with a different CROP_K -- rebuild it"


@needs_objects
def test_binary_attribute_heads_are_not_degenerate():
    """`material` and `size` are binary and the roadmap flagged them as possibly
    near-degenerate. If either majority class exceeds ~65% the head is mostly
    measuring the prior, and a 'good' accuracy there means nothing."""
    manifest = json.load(open(os.path.join(co.CACHE_DIR, "clevr_objects_manifest.json")))
    for split in manifest["splits"]:
        for head in ("material", "size"):
            rate = split["balance"][head]["majority_rate"]
            assert rate < 0.65, f"{split['split']}/{head} majority rate {rate:.2f} is too degenerate"


# ---------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------


def test_per_head_chance_guard_uses_each_heads_own_classes():
    """50% on the binary `material` head IS chance; 50% on the 8-way `color` head
    is not. A single 4-class default would get both wrong."""
    dead_binary = {"color": 80.0, "shape": 70.0, "material": 50.0, "size": 51.0}
    with pytest.raises(AssertionError, match="material"):
        cc.assert_not_chance_level_per_head("arm", dead_binary, cc.HEADS)

    healthy = {"color": 60.0, "shape": 70.0, "material": 80.0, "size": 85.0}
    cc.assert_not_chance_level_per_head("arm", healthy, cc.HEADS)

    # 50% on an 8-way head is far above its 12.5% chance floor and must pass.
    assert not cc.is_chance_level(50.0, n_classes=8)
    assert cc.is_chance_level(50.0, n_classes=2)


def test_harness_reuses_phase15_statistics_rather_than_reimplementing_them():
    """Forked training loops caused both Phase-1 audits. The fork here is
    deliberate, but the STATISTICS must not fork -- there must be exactly one
    Welch implementation in the repo."""
    from qnlp.image_tower.classification.quantum import phase15_common as pc

    assert cc.compare is pc.compare
    assert cc.summarise is pc.summarise
    assert cc.mde_unpaired is pc.mde_unpaired
    assert cc.is_chance_level is pc.is_chance_level
    # The one intentional protocol deviation, pinned so it stays visible.
    assert cc.CLEVR_PROTOCOL["test_samples"] == 512
    for key in ("train_samples", "epochs", "batch_size", "lr"):
        assert cc.CLEVR_PROTOCOL[key] == pc.PROTOCOL[key], f"{key} must match Phase 1"


@needs_objects
def test_multihead_training_runs_and_moves_every_head():
    """End-to-end smoke test: a tiny run must produce one curve per head, and
    training must not leave a head pinned at its initial value."""
    curves, model = cc.train_run_multihead(
        lambda: CoherentQTTNClassifier(
            readout="top_layer_qubits", encoding="multi_axis", ansatz="iqp", n_classes=cc.HEADS
        ),
        seed=0,
        epochs=2,
        train_samples=64,
        test_samples=64,
    )
    assert set(curves) == set(cc.HEADS)
    for head, curve in curves.items():
        assert len(curve) == 2, head
        assert all(0.0 <= v <= 100.0 for v in curve), head
    grads = [n for n, p in model.named_parameters() if p.grad is None]
    assert not grads or all("head" not in n for n in grads)
