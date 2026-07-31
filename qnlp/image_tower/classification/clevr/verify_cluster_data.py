"""Verify a cluster (or fresh local) checkout is ready to run C1-C4.

Every failure mode this checks for is one that otherwise surfaces *hours* into an
array job, after the scheduler has already handed out the slots:

  * missing .npz caches  -> every task dies instantly in get_clevr_loaders
  * a cache built with a different CROP_K -> results silently incomparable to the
    ones already in research_log.md, which is worse than a crash
  * too few crops for the protocol -> ValueError only once training starts
  * a degenerate or unbalanced head -> looks like a result, is not
  * no lightning.qubit -> runs ~4x slower than the cost model assumes

Exits non-zero on anything that would invalidate a run, so it can gate a submit
script. Run: python -m qnlp.image_tower.classification.clevr.verify_cluster_data
"""

import os
import sys

import numpy as np

from qnlp.utils.data import clevr_objects as co

# What the protocol actually consumes, so "enough data" is checked against the
# real requirement rather than a guess.
NEEDED = {"train": 1024, "val": 512}


def _check(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


def main():
    print("=" * 70)
    print("CLEVR cluster readiness check")
    print("=" * 70)
    ok = True

    print("\nEnvironment:")
    try:
        import torch

        ok &= _check("torch", True, torch.__version__)
    except Exception as e:
        ok &= _check("torch", False, str(e))
    try:
        import pennylane as qml

        ok &= _check("pennylane", True, qml.__version__)
        try:
            qml.device("lightning.qubit", wires=4)
            _check("lightning.qubit device", True, "quantum arms will run at the costed speed")
        except Exception as e:
            # Not fatal: default.qubit still produces correct results, just slower.
            _check("lightning.qubit device", False, f"{e} -- falls back to default.qubit, ~4x slower")
    except Exception as e:
        ok &= _check("pennylane", False, str(e))

    print("\nData caches:")
    for task, heads in (("objects", list(co.ATTRIBUTES)), ("relations", ["relation"])):
        manifest_path = os.path.join(co.CACHE_DIR, f"clevr_{task}_manifest.json")
        if not os.path.exists(manifest_path):
            ok &= _check(f"{task} manifest", False, f"{manifest_path} missing -- run build_clevr_crops")
            continue
        manifest = co.load_manifest(task)

        # A cache built at a different CROP_K is the dangerous case: it loads and
        # trains fine, and produces numbers that cannot be compared with anything
        # already logged.
        same_k = abs(manifest["crop_k"] - co.CROP_K) < 1e-9
        ok &= _check(
            f"{task} CROP_K matches code",
            same_k,
            f"cache {manifest['crop_k']} vs code {co.CROP_K} -- REBUILD, results would be incomparable"
            if not same_k
            else f"{co.CROP_K}",
        )

        for res in co.RESOLUTIONS:
            for split in ("train", "val"):
                path = co.cache_path(task, res, split)
                if not os.path.exists(path):
                    ok &= _check(f"{task} {res}px {split}", False, f"{path} missing")
                    continue
                with np.load(path) as z:
                    n, imgs = z["images"].shape[0], z["images"]
                    shape_ok = imgs.shape[1:] == (res, res, 3)
                    label_ok = all(h in z for h in heads)
                    enough = n >= NEEDED[split]
                good = shape_ok and label_ok and enough
                ok &= _check(
                    f"{task} {res}px {split}",
                    good,
                    f"n={n} (need {NEEDED[split]}), shape={imgs.shape[1:]}"
                    + ("" if label_ok else f", MISSING label(s) {[h for h in heads if h not in z]}"),
                )

        # Balance. A head at its majority class looks like a result and is not.
        for split in manifest["splits"]:
            for head, info in split["balance"].items():
                rate = info["majority_rate"]
                limit = 0.65 if head in ("material", "size") else (0.30 if head == "relation" else 0.99)
                ok &= _check(
                    f"{task}/{split['split']} {head} balance",
                    rate <= limit,
                    f"majority {rate:.3f} (limit {limit})",
                )

    print("\n" + "=" * 70)
    print("READY" if ok else "NOT READY -- fix the FAILs above before submitting array jobs")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
