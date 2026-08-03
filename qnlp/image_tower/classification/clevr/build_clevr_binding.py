"""Task C6: build the shape-binding composite dataset, and render it for inspection.

  --montage     Render labelled composites and STOP. Run this FIRST and look at
                it before building anything, for two reasons:

                  1. `shape` is an integer in the cache and this code does not
                     know which integer is a cube. The montage is how the
                     mapping gets established -- record it in research_log.md
                     rather than assuming it.
                  2. It is the only real check that the label matches the
                     picture. A sign or ordering error would still produce a
                     perfectly balanced, plausible-looking dataset and a
                     plausible ~50% result. C0's relation montage is the
                     precedent: it is what caught the rotated-axis error.

  (default)     Build and cache train/val composites, then report balance.

Run:
  conda run -n qnlp python -m qnlp.image_tower.classification.clevr.build_clevr_binding --montage
  conda run -n qnlp python -m qnlp.image_tower.classification.clevr.build_clevr_binding
"""

import argparse
import json
import os

import numpy as np

from qnlp.utils.data import clevr_binding as cb
from qnlp.utils.data.clevr_objects import ATTRIBUTES


def montage(attribute, attr_pair, n_show=8, out=None, **geom):
    """Render composites with their labels, plus the source crops of each shape.

    The bottom two rows show single-object crops of each shape class on its own,
    which is what makes the integer->name mapping readable at a glance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or f"figures/clevr_binding_{attribute}_examples.png"
    imgs, labels, meta = cb.build_split("train", 2 * n_show, attribute=attribute, attr_pair=attr_pair, **geom)

    from qnlp.utils.data.clevr_objects import cache_path

    with np.load(cache_path("objects", geom.get("source_res", cb.SOURCE_RES), "train")) as z:
        src, src_attr = z["images"], z[attribute]

    fig, axes = plt.subplots(4, n_show, figsize=(1.7 * n_show, 7.6))
    for cls in (0, 1):
        picks = np.flatnonzero(labels == cls)[:n_show]
        for c, i in enumerate(picks):
            ax = axes[cls, c]
            ax.imshow(imgs[i])
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"class {cls}\n{cb.BINDING_CLASSES[cls]}", fontsize=8)
    for r, val in enumerate(attr_pair):
        picks = np.flatnonzero(src_attr == val)[:n_show]
        for c, i in enumerate(picks):
            ax = axes[2 + r, c]
            ax.imshow(src[i])
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"source\n{attribute}={val}", fontsize=8)

    fig.suptitle(
        f"C6 binding composites: {attribute} {attr_pair[0]} vs {attr_pair[1]}.\n"
        f"Row 0 = {attribute}={attr_pair[0]} on the LEFT; row 1 = {attribute}={attr_pair[1]} on the LEFT. "
        f"Rows 2-3 show each class alone -- use them to name the classes.",
        fontsize=10,
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    print("LOOK AT IT before building. Check three things:")
    print("  1. every image has exactly two objects, side by side, not overlapping;")
    print(f"  2. row 0 really does have {attribute}={attr_pair[0]} on the left, and row 1 the other way;")
    print(f"  3. which integer of `{attribute}` is which -- record it in research_log.md.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", action="store_true", help="Render examples and stop. DO THIS FIRST.")
    ap.add_argument(
        "--attribute",
        default=cb.DEFAULT_ATTRIBUTE,
        choices=sorted(ATTRIBUTES),
        help="Which attribute to bind. DEFAULT `size` BECAUSE EVERY ARM PERCEIVES IT (84.8-99.1%% on "
        "single objects in C3), so a failure here is a BINDING failure. Do not use `shape`: "
        "classical_bare (36.5) and mlp_param_matched (36.9) sit at its 35.4 floor, so their scores "
        "would measure perception. `material` is the fallback if size saturates.",
    )
    ap.add_argument(
        "--classes",
        type=int,
        nargs=2,
        default=list(cb.DEFAULT_ATTR_PAIR),
        help="The two DISTINCT attribute values to bind. Integers, not names -- see --montage.",
    )
    ap.add_argument("--train-samples", type=int, default=4096)
    ap.add_argument("--val-samples", type=int, default=2048)
    ap.add_argument("--source-res", type=int, default=cb.SOURCE_RES)
    ap.add_argument("--canvas", type=int, default=cb.CANVAS)
    ap.add_argument("--cell", type=int, default=cb.CELL)
    ap.add_argument("--jitter-x", type=int, default=cb.JITTER_X)
    ap.add_argument("--jitter-y", type=int, default=cb.JITTER_Y)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    geom = dict(
        canvas=args.canvas,
        cell=args.cell,
        jitter_x=args.jitter_x,
        jitter_y=args.jitter_y,
        source_res=args.source_res,
    )
    pair, attribute = tuple(args.classes), args.attribute

    if args.montage:
        montage(attribute, pair, **geom)
        return

    print(
        f"=== C6 binding composites | {attribute} {pair} | canvas {args.canvas} | cell {args.cell} ===",
        flush=True,
    )
    splits = []
    for split, n, seed in (("train", args.train_samples, args.seed), ("val", args.val_samples, args.seed + 1)):
        imgs, labels, meta = cb.build_split(split, n, attribute=attribute, attr_pair=pair, seed=seed, **geom)
        path = cb.save_split(split, imgs, labels, canvas=args.canvas, attribute=attribute)
        counts = np.bincount(labels, minlength=2)
        rate = counts.max() / counts.sum()
        print(f"  {split}: {len(labels)} composites  counts={counts.tolist()}  majority={rate:.3f}  -> {path}")
        # Exactly balanced by construction (the classes are alternated, not
        # sampled), so anything other than 0.500 is a bug, not sampling noise.
        assert abs(rate - 0.5) < 1e-9, f"{split} is not exactly balanced ({rate:.4f}) -- this cannot happen; fix it"
        splits.append({"split": split, "n": int(len(labels)), "counts": counts.tolist(), **meta})

    manifest = {
        "task": "binding",
        "attribute": attribute,
        "attr_pair": list(pair),
        "class_names": None,  # Fill in from the montage. Do NOT guess.
        "perception_rationale": (
            "size is bound because every arm perceives it on single objects (C3: 84.8-99.1%), so a "
            "failure here is a BINDING failure. The first C6 build used `shape`, where "
            "classical_bare (36.5) and mlp_param_matched (36.9) sit at the 35.4 floor -- that run is "
            "confounded by perception and is kept only as a footnote."
        ),
        "canvas": args.canvas,
        "cell": args.cell,
        "jitter_x": args.jitter_x,
        "jitter_y": args.jitter_y,
        "source_res": args.source_res,
        "source": "clevr_objects_<source_res>_<split>.npz (train/val from different CLEVR shards)",
        "splits": splits,
    }
    mpath = cb.binding_manifest_path(attribute)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {mpath}")
    print("\n⚠️  `class_names` in the manifest is null. Fill it in from the montage before writing anything up.")


if __name__ == "__main__":
    main()
