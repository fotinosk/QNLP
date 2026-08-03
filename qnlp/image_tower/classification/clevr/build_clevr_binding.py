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
from qnlp.utils.data.clevr_objects import CACHE_DIR


def montage(shape_pair, n_show=8, out="figures/clevr_binding_examples.png", **geom):
    """Render composites with their labels, plus the source crops of each shape.

    The bottom two rows show single-object crops of each shape class on its own,
    which is what makes the integer->name mapping readable at a glance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs, labels, meta = cb.build_split("train", 2 * n_show, shape_pair=shape_pair, **geom)

    from qnlp.utils.data.clevr_objects import cache_path

    with np.load(cache_path("objects", geom.get("source_res", cb.SOURCE_RES), "train")) as z:
        src, src_shape = z["images"], z["shape"]

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
    for r, sh in enumerate(shape_pair):
        picks = np.flatnonzero(src_shape == sh)[:n_show]
        for c, i in enumerate(picks):
            ax = axes[2 + r, c]
            ax.imshow(src[i])
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"source\nshape={sh}", fontsize=8)

    fig.suptitle(
        f"C6 binding composites: shape {shape_pair[0]} vs {shape_pair[1]}.\n"
        f"Row 0 = shape {shape_pair[0]} on the LEFT; row 1 = shape {shape_pair[1]} on the LEFT. "
        f"Rows 2-3 show each shape alone -- use them to name the classes.",
        fontsize=10,
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    print("LOOK AT IT before building. Check three things:")
    print("  1. every image has exactly two objects, side by side, not overlapping;")
    print("  2. row 0 really does have shape", shape_pair[0], "on the left, and row 1 the other way;")
    print("  3. which integer is the cube -- record it in research_log.md.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", action="store_true", help="Render examples and stop. DO THIS FIRST.")
    ap.add_argument(
        "--shapes",
        type=int,
        nargs=2,
        default=list(cb.DEFAULT_SHAPE_PAIR),
        help="The two DISTINCT shape classes to bind. Integers, not names -- see --montage.",
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
    pair = tuple(args.shapes)

    if args.montage:
        montage(pair, **geom)
        return

    print(f"=== C6 binding composites | shapes {pair} | canvas {args.canvas} | cell {args.cell} ===", flush=True)
    splits = []
    for split, n, seed in (("train", args.train_samples, args.seed), ("val", args.val_samples, args.seed + 1)):
        imgs, labels, meta = cb.build_split(split, n, shape_pair=pair, seed=seed, **geom)
        path = cb.save_split(split, imgs, labels, canvas=args.canvas)
        counts = np.bincount(labels, minlength=2)
        rate = counts.max() / counts.sum()
        print(f"  {split}: {len(labels)} composites  counts={counts.tolist()}  majority={rate:.3f}  -> {path}")
        # Exactly balanced by construction (the classes are alternated, not
        # sampled), so anything other than 0.500 is a bug, not sampling noise.
        assert abs(rate - 0.5) < 1e-9, f"{split} is not exactly balanced ({rate:.4f}) -- this cannot happen; fix it"
        splits.append({"split": split, "n": int(len(labels)), "counts": counts.tolist(), **meta})

    manifest = {
        "task": "binding",
        "shape_pair": list(pair),
        "shape_names": None,  # Fill in from the montage. Do NOT guess.
        "canvas": args.canvas,
        "cell": args.cell,
        "jitter_x": args.jitter_x,
        "jitter_y": args.jitter_y,
        "source_res": args.source_res,
        "source": "clevr_objects_<source_res>_<split>.npz (train/val from different CLEVR shards)",
        "splits": splits,
    }
    mpath = os.path.join(CACHE_DIR, "clevr_binding_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {mpath}")
    print("\n⚠️  `shape_names` in the manifest is null. Fill it in from the montage before writing anything up.")


if __name__ == "__main__":
    main()
