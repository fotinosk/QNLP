"""Task C7: is frozen CLIP's embedding geometry blind to compositional binding?

WHY COSINE SIMILARITY AND NOT A TRAINED PROBE
---------------------------------------------
Cosine similarity IS the operation CLIP performs at inference -- retrieval,
zero-shot classification and the contrastive objective all reduce to cosine
distance in embedding space. So "swapped images sit at near-identical cosine
similarity" is a statement about what the model actually does, not about what
could in principle be dug back out of it.

That is also the limitation, and it must be stated: this measures SALIENCE IN THE
GEOMETRY, not RECOVERABILITY. A linear probe might still extract the binding from
a low-variance direction at high cosine similarity. For a contrastive retrieval
model the geometric claim is the relevant one, but they are different claims.

WHAT MAKES THE MEASUREMENT INTERPRETABLE
----------------------------------------
Two things, both easy to get wrong:

1. **The pairs are MATCHED.** The swap reuses the SAME TWO CROPS, so the only
   difference between the two images is which side each object is on. Comparing
   two independently sampled composites would confound content with arrangement
   and measure their sum.

2. **Cosine similarity has no scale**, so every item carries its own ceiling and
   floor. CLIP embeddings are strongly anisotropic and unrelated images sit at
   0.5-0.8 routinely; a bare "0.97" means nothing. Hence:

       jitter   same objects, same sides, re-placed   -> CEILING
       swap     same objects, sides exchanged         -> the effect
       content  different objects, same arrangement   -> FLOOR

       swap_invariance = (cos_swap - cos_content) / (cos_jitter - cos_content)

   near 1 -> a compositional swap is treated almost like the same image;
   near 0 -> the swap is as separable as a change of content.

PRE-REGISTERED READINGS -- FIXED BEFORE THE RUN
-----------------------------------------------
  * swap_invariance near 1, AND cos_content clearly below cos_jitter
        -> CLIP's geometry is nearly blind to binding while remaining sensitive
           to content. This is the mechanism C6 could not supply.
  * cos_swap ~ cos_content
        -> CLIP does separate the swap; the premise does not hold here.
  * cos_jitter ~ cos_content
        -> VOID, NOT A FINDING. CLIP is not distinguishing anything in these
           images at all, so the run measures resolution or out-of-distribution
           inputs rather than composition. This guard is the C1 lesson applied
           before the compute rather than after.

Run: conda run -n qnlp python -m qnlp.image_tower.classification.clevr.run_c7_clip_geometry
"""

import argparse

import numpy as np
import torch

from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.utils.data import clevr_binding as cb
from qnlp.utils.data.clevr_objects import cache_path as objects_cache_path

MODEL_ID = "openai/clip-vit-base-patch32"


def embed(images, batch_size=64, model_id=MODEL_ID):
    """Frozen CLIP image embeddings, L2-normalised (so a dot product IS cosine)."""
    from transformers import CLIPImageProcessor, CLIPModel

    proc = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = [images[j] for j in range(i, min(i + batch_size, len(images)))]
            px = proc(images=batch, return_tensors="pt")["pixel_values"]
            e = model.get_image_features(pixel_values=px)
            outs.append(torch.nn.functional.normalize(e, dim=-1))
            print(f"  encoded {min(i + batch_size, len(images))}/{len(images)}", end="\r", flush=True)
    print()
    return torch.cat(outs)


def summarise(name, vals):
    """Mean with a 95% CI, using the project's own t critical value."""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    sd = float(v.std(ddof=1))
    half = pc.t_crit(n - 1) * sd / np.sqrt(n)
    return {"name": name, "mean": float(v.mean()), "std": sd, "n": n, "ci95": float(half)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quads",
        type=int,
        default=500,
        help="Matched quads; 500 x 4 = 2000 CLIP forward passes. NOT named --n: `conda run` parses it "
        "as an ambiguous prefix of its own --name.",
    )
    ap.add_argument("--canvas", type=int, default=224, help="CLIP's native resolution. Do NOT hand it a thumbnail.")
    ap.add_argument("--cell", type=int, default=96)
    ap.add_argument("--jitter-x", type=int, default=14, help="Scaled from the 32px build's 2px.")
    ap.add_argument("--jitter-y", type=int, default=42, help="Scaled from the 32px build's 6px.")
    ap.add_argument("--source-res", type=int, default=64, help="Largest cached object crop, for the least upscaling.")
    ap.add_argument("--attribute", default=cb.DEFAULT_ATTRIBUTE)
    ap.add_argument("--classes", type=int, nargs=2, default=list(cb.DEFAULT_ATTR_PAIR))
    ap.add_argument(
        "--background",
        default="floor",
        choices=["floor", "black"],
        help="Canvas padding. `floor` fills with the CLEVR floor grey so the composite reads as one "
        "scene; `black` was the first run and leaves ~60%% of the image black, which compresses "
        "every cosine similarity upward. Run both -- the ratio should be robust to it.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="c7_clip_geometry_results.json")
    args = ap.parse_args()

    path = objects_cache_path("objects", args.source_res, "val")
    with np.load(path) as z:
        src, attrs = z["images"], z[args.attribute]

    print(
        f"C7 CLIP geometry | {MODEL_ID} | {args.canvas}x{args.canvas} canvas, {args.cell}px cells\n"
        f"attribute={args.attribute} classes={tuple(args.classes)} | background={args.background}\n"
        f"{args.quads} matched quads from {path}",
        flush=True,
    )
    quads, meta = cb.build_matched_quads(
        src,
        attrs,
        args.quads,
        attr_pair=tuple(args.classes),
        seed=args.seed,
        canvas=args.canvas,
        cell=args.cell,
        jitter_x=args.jitter_x,
        jitter_y=args.jitter_y,
        background=args.background,
    )

    flat = [quads[i, k] for i in range(args.quads) for k in range(4)]
    e = embed(flat).reshape(args.quads, 4, -1)
    base, swap, jit, cont = e[:, 0], e[:, 1], e[:, 2], e[:, 3]

    cos = {
        "jitter": (base * jit).sum(-1).numpy(),
        "swap": (base * swap).sum(-1).numpy(),
        "content": (base * cont).sum(-1).numpy(),
    }
    stats = {k: summarise(k, v) for k, v in cos.items()}

    print("\n" + "=" * 78)
    print(f"C7: frozen CLIP cosine similarity, {args.quads} matched quads")
    print("=" * 78)
    print(f"{'comparison':<12}{'cos sim':>12}{'95% CI':>12}   role")
    for k, role in (
        ("jitter", "CEILING -- same objects, re-placed"),
        ("swap", "THE EFFECT -- same objects, sides exchanged"),
        ("content", "FLOOR -- different objects, same arrangement"),
    ):
        s = stats[k]
        print(f"{k:<12}{s['mean']:>12.4f}{s['ci95']:>12.4f}   {role}")

    span = stats["jitter"]["mean"] - stats["content"]["mean"]
    inv = (stats["swap"]["mean"] - stats["content"]["mean"]) / span if span > 1e-9 else float("nan")
    # Per-item, so the statistic carries a CI rather than being a ratio of means.
    per_item = (cos["swap"] - cos["content"]) / np.maximum(cos["jitter"] - cos["content"], 1e-9)
    inv_stats = summarise("swap_invariance", np.clip(per_item, -1.0, 2.0))

    print(f"\nceiling - floor span: {span:.4f}")
    print(f"SWAP INVARIANCE: {inv:.3f}  (per-item {inv_stats['mean']:.3f} +/- {inv_stats['ci95']:.3f})")
    print("  1.0 = a compositional swap is treated exactly like the same image")
    print("  0.0 = the swap is as separable as a change of content")

    # The guard, applied before any interpretation.
    void = span < 0.02
    if void:
        verdict = (
            f"VOID, NOT A FINDING. The ceiling and floor are only {span:.4f} apart, so CLIP is not "
            f"meaningfully distinguishing ANY of these images -- this measures resolution or "
            f"out-of-distribution inputs, not composition. Raise the canvas/cell resolution or use "
            f"less upscaled source crops, and re-run before reading anything into the swap."
        )
    elif inv > 0.8:
        verdict = (
            f"CLIP'S GEOMETRY IS NEARLY BLIND TO BINDING. A compositional swap moves the embedding "
            f"only {1 - inv:.0%} of the way toward a genuine content change ({inv:.3f} of the "
            f"ceiling-floor span), while content changes move it fully. Since cosine similarity is "
            f"the operation CLIP performs at retrieval time, this is the mechanism behind its "
            f"bag-of-words behaviour. NOTE: this is salience in the geometry, NOT recoverability -- "
            f"a probe might still extract the binding from a low-variance direction."
        )
    elif inv < 0.4:
        verdict = (
            f"CLIP DOES SEPARATE THE SWAP ({inv:.3f} of the ceiling-floor span), so its embedding is "
            f"NOT blind to binding on this task. The premise does not hold here and must be "
            f"restated for the vision case."
        )
    else:
        verdict = (
            f"INTERMEDIATE ({inv:.3f}). CLIP is partially sensitive to the swap. Report the number "
            f"and the span; do not round it to either story."
        )
    print(f"\nVERDICT: {verdict}")

    pc.save(
        {
            "task": "clip_geometry",
            "model": MODEL_ID,
            "n_quads": args.quads,
            "geometry": {**meta, "background": args.background},
            "source": path,
            "cosine": stats,
            "ceiling_minus_floor": float(span),
            "swap_invariance": float(inv),
            "swap_invariance_per_item": inv_stats,
            "void": bool(void),
            "verdict": verdict,
            "limitation": (
                "Measures salience in the embedding geometry (what cosine retrieval sees), not "
                "recoverability by a trained probe. Frozen image tower only; the full image-text "
                "contrastive setting that ARO/SugarCrepe probe is not tested here."
            ),
        },
        args.out,
    )


if __name__ == "__main__":
    main()
