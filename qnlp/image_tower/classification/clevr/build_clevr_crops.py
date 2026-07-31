"""Task C0: build the CLEVR object-crop and relation-crop datasets.

Two modes:

  --calibrate   Sweep CROP_K, render a montage, and print the measured fill
                fractions. Run this FIRST, look at the picture, then freeze
                CROP_K in clevr_objects.py. Do not re-tune it later -- a moving
                data definition is how `classical_bare` acquired three different
                "measurements" in Phase 1.

  (default)     Build and cache train/val splits at 16, 32 and 64 px for both
                tasks, and report class balance per attribute.

Train and val come from DIFFERENT CLEVR shards, so no scene appears in both.

Run:
  conda run -n qnlp python -m qnlp.image_tower.classification.clevr.build_clevr_crops --calibrate
  conda run -n qnlp python -m qnlp.image_tower.classification.clevr.build_clevr_crops
"""

import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from qnlp.utils.data import clevr_objects as co


def shard_path(shard):
    """Download once into the HF cache; subsequent runs are local."""
    print(f"Fetching {shard} (~0.4-0.5 GB, cached after the first run) ...", flush=True)
    return hf_hub_download(repo_id=co.CLEVR_REPO, filename=shard, repo_type="dataset")


def iter_rows(path, max_scenes):
    """Stream row groups so a 0.5 GB shard never lands in memory whole."""
    pf = pq.ParquetFile(path)
    seen = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        for row in table.to_pylist():
            yield row
            seen += 1
            if seen >= max_scenes:
                return


def build_split(task, shard, split, max_scenes, resolutions, seed):
    path = shard_path(shard)
    images_by_res = {s: [] for s in resolutions}
    heads = list(co.ATTRIBUTES) if task == "objects" else ["relation"]
    labels = {h: [] for h in heads}
    rng = np.random.default_rng(seed)

    n_scenes = 0
    for row in iter_rows(path, max_scenes):
        n_scenes += 1
        it = (
            co.iter_object_crops(row, resolutions)
            if task == "objects"
            else co.iter_relation_crops(row, resolutions, rng=rng)
        )
        for crops, label in it:
            for s in resolutions:
                images_by_res[s].append(crops[s])
            for h in heads:
                labels[h].append(label[h])
        if n_scenes % 500 == 0:
            print(f"  {n_scenes} scenes -> {len(labels[heads[0]])} crops", flush=True)

    n = len(labels[heads[0]])
    if n == 0:
        raise RuntimeError(f"{task}/{split}: produced zero crops. Check the occlusion and margin filters.")

    keep = co.balance_classes(labels, task, seed=seed)
    if keep is not None:
        print(f"  balancing {task}: {n} -> {len(keep)} crops (equal per relation)")
        images_by_res = {s: [imgs[i] for i in keep] for s, imgs in images_by_res.items()}
        labels = {h: [labels[h][i] for i in keep] for h in heads}

    balance = co.class_balance(labels, task)
    co.save_split(task, split, images_by_res, labels)
    print(f"  {task}/{split}: {n_scenes} scenes -> {len(labels[heads[0]])} crops")
    for name, info in balance.items():
        print(f"    {name:<9} counts={info['counts']}  majority={info['majority_rate']:.3f}")
    return {"split": split, "n_scenes": n_scenes, "n_crops": len(labels[heads[0]]), "balance": balance}


def calibrate(max_scenes=40, ks=(1100.0, 1470.0, 1900.0), out="figures/clevr_crop_calibration.png"):
    """Render crops at several CROP_K values so the choice is made by looking.

    Also prints the fill fraction for small and large objects. The property to
    check is that it depends only on `size` and NOT on depth -- that invariance
    is what makes the `size` head learnable at all.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = shard_path(co.TRAIN_SHARD)
    rows = list(iter_rows(path, max_scenes))
    original_k = co.CROP_K

    n_show = 8
    fig, axes = plt.subplots(len(ks), n_show, figsize=(2 * n_show, 2.3 * len(ks)))
    for r, k in enumerate(ks):
        co.CROP_K = k
        shown, fills = 0, {0: [], 1: []}
        for row in rows:
            objs = row["objects"]
            # Fill fractions come straight from the geometry (3d_coords[2] is the
            # object's radius), so they need no image decoding.
            pixel_coords = [tuple(map(float, p)) for p in objs["pixel_coords"]]
            for i in range(len(objs["color"])):
                if co.is_occluded(i, pixel_coords):
                    continue
                fills[int(objs["size"][i])].append(co.fill_fraction(objs["3d_coords"][i][2]))
            for crops, label in co.iter_object_crops(row, resolutions=(64,)):
                if shown >= n_show:
                    break
                ax = axes[r][shown]
                ax.imshow(crops[64])
                ax.set_title(f"sz={label['size']} sh={label['shape']}", fontsize=7)
                ax.axis("off")
                shown += 1
            if shown >= n_show:
                break
        lo = np.mean(fills[0]) if fills[0] else float("nan")
        hi = np.mean(fills[1]) if fills[1] else float("nan")
        axes[r][0].set_ylabel(f"K={k:.0f}")
        print(f"CROP_K={k:>6.0f}   small fills {lo:.3f} of the box, large fills {hi:.3f}")
    co.CROP_K = original_k

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.suptitle("CLEVR crop calibration: pick the K where the object fills the frame without clipping")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nSaved {out} -- LOOK AT IT, then freeze CROP_K in clevr_objects.py.")


def relation_montage(max_scenes=200, per_class=4, out="figures/clevr_relation_examples.png"):
    """Render labelled relation crops, grouped by relation.

    This is the ONLY real check that CLEVR's direction vectors are applied
    correctly. A sign error or a swapped axis would still produce a balanced,
    plausible-looking dataset and a plausible-looking ~25% result -- there is no
    automated test for "the labels mean what they say". Look at the picture: in
    every tile the reference object is DEAD CENTRE, and the other object should
    sit where the row label says it does.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = shard_path(co.TRAIN_SHARD)
    rng = np.random.default_rng(0)
    buckets = {r: [] for r in co.RELATIONS}
    for row in iter_rows(path, max_scenes):
        for crops, label in co.iter_relation_crops(row, resolutions=(64,), rng=rng):
            rel = co.RELATIONS[label["relation"]]
            if len(buckets[rel]) < per_class:
                buckets[rel].append(crops[64])
        if all(len(v) >= per_class for v in buckets.values()):
            break

    fig, axes = plt.subplots(len(co.RELATIONS), per_class, figsize=(2.2 * per_class, 2.4 * len(co.RELATIONS)))
    for r, rel in enumerate(co.RELATIONS):
        for c in range(per_class):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(buckets[rel]):
                ax.imshow(buckets[rel][c])
            if c == 0:
                ax.set_title(f"other object is {rel.upper()} of the centre object", fontsize=9, loc="left")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.suptitle("CLEVR relations: reference object is centred; check the label matches the picture")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Saved {out} -- LOOK AT IT before trusting any C4 number.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="Render the CROP_K montage instead of building.")
    ap.add_argument("--relation-montage", action="store_true", help="Render labelled relation examples.")
    ap.add_argument("--tasks", nargs="+", default=["objects", "relations"], choices=["objects", "relations"])
    ap.add_argument("--train-scenes", type=int, default=3000)
    ap.add_argument("--val-scenes", type=int, default=1500)
    ap.add_argument("--resolutions", type=int, nargs="+", default=list(co.RESOLUTIONS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
        return
    if args.relation_montage:
        relation_montage()
        return

    for task in args.tasks:
        print(f"\n=== {task} (CROP_K={co.CROP_K}) ===", flush=True)
        manifest = {
            "task": task,
            "crop_k": co.CROP_K,
            "occlusion_frac": co.OCCLUSION_FRAC,
            "relation_margin": co.RELATION_MARGIN,
            "max_out_of_frame": co.MAX_OUT_OF_FRAME,
            "max_relation_side": co.MAX_RELATION_SIDE,
            "max_out_of_frame_relation": co.MAX_OUT_OF_FRAME_RELATION,
            "resolutions": args.resolutions,
            "source": {"repo": co.CLEVR_REPO, "train_shard": co.TRAIN_SHARD, "test_shard": co.TEST_SHARD},
            "splits": [
                build_split(task, co.TRAIN_SHARD, "train", args.train_scenes, args.resolutions, args.seed),
                build_split(task, co.TEST_SHARD, "val", args.val_scenes, args.resolutions, args.seed + 1),
            ],
        }
        os.makedirs(co.CACHE_DIR, exist_ok=True)
        mpath = os.path.join(co.CACHE_DIR, f"clevr_{task}_manifest.json")
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved {mpath}")


if __name__ == "__main__":
    main()
