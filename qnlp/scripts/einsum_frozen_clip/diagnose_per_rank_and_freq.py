"""Deep diagnostic on an einsum_frozen_clip COCO checkpoint.

Produces three insights about overfitting:

  1. PER-RANK MOVEMENT: rank-1 nouns vs rank-2 vs rank-3 verbs — which
     learned and which didn't?

  2. PER-FREQUENCY MOVEMENT: for each typed symbol, compute its training
     frequency (from train parquet) and its parameter movement from init.
     Then bin into deciles by frequency. Hypothesis: rare symbols overfit
     (move a lot per gradient step, since they only see 1-2 batches) while
     common symbols move little but generalise.

  3. PER-FREQUENCY VAL ACCURACY (if feasible): does the model do better
     on val captions where every symbol is high-frequency in train?

Run on klo (where the checkpoint is) or vanilla.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import polars as pl
import torch

from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.einsum_frozen_clip.config import ExperimentConfig
from qnlp.scripts.einsum_frozen_clip.image_model import FrozenClipVisionModel
from qnlp.utils.seeding import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-parquet", type=Path, default=Path("data/datasets/coco_contrastive_train.parquet"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/diagnose_einsum_frozen_clip.json"))
    args = parser.parse_args()

    cfg = ExperimentConfig()

    # Build CPU model
    text_model = EinsumModel()
    image_model = FrozenClipVisionModel(cfg.clip_model_name, cfg.embedding_dim)
    trained = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    trained.load_state_dict(sd, strict=True)
    print(f"loaded checkpoint: {args.checkpoint}")
    print(f"  n_symbols: {len(text_model.symbols)}")
    print(f"  n_params (text): {sum(p.numel() for p in text_model.parameters()):,}")

    # Rebuild a fresh-init reference with same seed
    set_seed()
    text_init = EinsumModel(list(text_model.symbols), list(text_model.sizes))

    # ── Count symbol occurrences in train ────────────────────────────────────
    print(f"\ncounting symbol occurrences in {args.train_parquet}...")
    freq: Counter = Counter()
    df = pl.read_parquet(args.train_parquet)
    for col in ("true_symbols", "false_symbols"):
        for raw in df[col].to_list():
            if raw is None:
                continue
            entries = orjson.loads(raw) if isinstance(raw, str) else raw
            for e in entries:
                if isinstance(e, (list, tuple)) and e and isinstance(e[0], dict):
                    d = e[0]
                    key = (d["name"], d["directed_dom"], d["directed_cod"])
                    freq[key] += 1
    print(f"  total occurrences: {sum(freq.values()):,}")
    print(f"  unique symbols in train: {len(freq):,}")

    # ── Per-symbol movement vs frequency ─────────────────────────────────────
    sym_records = []
    for sym, p_final, p_init in zip(text_model.symbols, text_model.weights, text_init.weights):
        key = (sym.name, sym.directed_dom, sym.directed_cod)
        f = freq.get(key, 0)
        i_norm = p_init.detach().norm().item()
        f_norm = p_final.detach().norm().item()
        d_norm = (p_final.detach() - p_init.detach()).norm().item()
        sym_records.append({
            "name": sym.name,
            "rank": len(p_final.shape),
            "size": list(p_final.shape),
            "freq": f,
            "init_norm": i_norm,
            "final_norm": f_norm,
            "delta_norm": d_norm,
            "rel_movement": d_norm / (i_norm + 1e-12),
        })

    # ── Aggregate by rank ────────────────────────────────────────────────────
    by_rank: dict[int, dict] = defaultdict(lambda: {"records": []})
    for r in sym_records:
        by_rank[r["rank"]]["records"].append(r)

    print("\n" + "=" * 78)
    print("PER-RANK MOVEMENT")
    print("=" * 78)
    print(f"{'Rank':<5} {'n':>7} {'<freq>':>10} {'<init>':>10} {'<final>':>10} "
          f"{'<Δ>':>10} {'<Δ>/<init>':>12} {'frac_dead':>10}")
    for r in sorted(by_rank):
        recs = by_rank[r]["records"]
        n = len(recs)
        avg_freq = sum(x["freq"] for x in recs) / n
        avg_init = sum(x["init_norm"] for x in recs) / n
        avg_fin = sum(x["final_norm"] for x in recs) / n
        avg_d = sum(x["delta_norm"] for x in recs) / n
        avg_rel = sum(x["rel_movement"] for x in recs) / n
        dead = sum(1 for x in recs if x["delta_norm"] < 0.01 * x["init_norm"])
        print(f"{r:<5} {n:>7,} {avg_freq:>10.1f} {avg_init:>10.4f} {avg_fin:>10.4f} "
              f"{avg_d:>10.4f} {avg_rel:>12.4f} {dead/n:>10.2%}")

    # ── Aggregate by frequency bucket ────────────────────────────────────────
    sym_records_sorted = sorted(sym_records, key=lambda x: x["freq"])
    print("\n" + "=" * 78)
    print("PER-FREQUENCY-BUCKET MOVEMENT  (sorted by train frequency, decile bins)")
    print("=" * 78)
    n_total = len(sym_records_sorted)
    print(f"{'Decile':<8} {'freq range':<20} {'n':>7} {'<Δ>/<init>':>12} {'<final norm>':>14}")
    for i in range(10):
        lo = (i * n_total) // 10
        hi = ((i + 1) * n_total) // 10
        bucket = sym_records_sorted[lo:hi]
        if not bucket: continue
        fmin = bucket[0]["freq"]; fmax = bucket[-1]["freq"]
        n = len(bucket)
        rel = sum(x["rel_movement"] for x in bucket) / n
        fn = sum(x["final_norm"] for x in bucket) / n
        print(f"{i+1:<8} [{fmin:>5}, {fmax:>5}]      {n:>7,} {rel:>12.4f} {fn:>14.4f}")

    # ── How many never-seen-in-train symbols are in the model? ───────────────
    never_seen = sum(1 for r in sym_records if r["freq"] == 0)
    once_only = sum(1 for r in sym_records if r["freq"] == 1)
    twice_only = sum(1 for r in sym_records if r["freq"] == 2)
    print(f"\nNever-trained symbols (freq=0):  {never_seen:,} ({100*never_seen/n_total:.1f}%)")
    print(f"Trained once (freq=1):           {once_only:,} ({100*once_only/n_total:.1f}%)")
    print(f"Trained twice (freq=2):          {twice_only:,} ({100*twice_only/n_total:.1f}%)")

    # ── Save raw records for further analysis ────────────────────────────────
    with open(args.out, "w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "n_total": n_total,
            "n_records": len(sym_records),
            "freq_buckets": [
                {"bucket": i+1, "freq_range": [sym_records_sorted[(i*n_total)//10]["freq"],
                                                sym_records_sorted[((i+1)*n_total)//10-1]["freq"]],
                 "n": ((i+1)*n_total)//10 - (i*n_total)//10,
                 "avg_rel_movement": sum(x["rel_movement"] for x in sym_records_sorted[(i*n_total)//10:((i+1)*n_total)//10])/(((i+1)*n_total)//10-(i*n_total)//10)
                } for i in range(10)
            ],
            "records": sym_records,
        }, f)
    print(f"\nFull records written to: {args.out}")


if __name__ == "__main__":
    main()
