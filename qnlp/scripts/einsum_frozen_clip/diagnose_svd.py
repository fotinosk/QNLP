"""For each rank-3 typed tensor (shape e.g. 10×512×10), compute its
'effective rank' via the SVD spectrum: unfold to 512×100 matrix, take SVD,
report the participation ratio of singular values.

Hypothesis: if rank-3 tensors are using their full capacity, the SVD
should be relatively flat (many large singular values). If they're
collapsed to a near-rank-1 manifold, the spectrum will be dominated by
the top singular value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from collections import defaultdict

from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.einsum_frozen_clip.config import ExperimentConfig
from qnlp.scripts.einsum_frozen_clip.image_model import FrozenClipVisionModel
from qnlp.utils.seeding import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    text_model = EinsumModel()
    image_model = FrozenClipVisionModel(cfg.clip_model_name, cfg.embedding_dim)
    trained = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trained.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)

    set_seed()
    text_init = EinsumModel(list(text_model.symbols), list(text_model.sizes))

    # SVD per typed tensor → effective rank
    # Effective rank (Roy & Vetterli) = exp(entropy of normalized singular values)
    # Higher = more capacity used; lower = collapsed to a few directions.
    by_size = defaultdict(lambda: {"eff_ranks_trained": [], "eff_ranks_init": [], "explained_top1_trained": [], "explained_top1_init": []})

    n_processed = 0
    for sym, p_final, p_init in zip(text_model.symbols, text_model.weights, text_init.weights):
        size = tuple(p_final.shape)
        if len(size) < 2:
            continue  # rank-1 has no SVD spectrum
        # Unfold higher-rank to a 2D matrix: first axis × prod(rest)
        with torch.no_grad():
            t = p_final.detach()
            mat = t.reshape(t.shape[0], -1)
            s_final = torch.linalg.svdvals(mat.float())
            t_i = p_init.detach()
            mat_i = t_i.reshape(t_i.shape[0], -1)
            s_init = torch.linalg.svdvals(mat_i.float())

        # Normalize sv distribution
        def eff_rank(s):
            s = s.cpu().numpy().astype(np.float64)
            s_sq = s ** 2
            total = s_sq.sum()
            if total <= 0:
                return 0.0
            p = s_sq / total
            entropy = -np.sum(p * np.log(p + 1e-30))
            return float(np.exp(entropy))

        def explained_top1(s):
            s = s.cpu().numpy().astype(np.float64) ** 2
            total = s.sum()
            return float(s[0] / total) if total > 0 else 0.0

        by_size[size]["eff_ranks_trained"].append(eff_rank(s_final))
        by_size[size]["eff_ranks_init"].append(eff_rank(s_init))
        by_size[size]["explained_top1_trained"].append(explained_top1(s_final))
        by_size[size]["explained_top1_init"].append(explained_top1(s_init))

        n_processed += 1
        if n_processed % 5000 == 0:
            print(f"  processed {n_processed}")

    print("\n" + "=" * 78)
    print("EFFECTIVE RANK per typed-tensor shape (mean ± std)")
    print("=" * 78)
    print(f"{'shape':<20} {'n':>7} {'max poss.':>10} {'eff_rank init':>16} {'eff_rank trained':>18} {'top-1 var init':>16} {'top-1 var trained':>20}")
    summary = []
    for size in sorted(by_size):
        b = by_size[size]
        n = len(b["eff_ranks_trained"])
        max_rank = min(size[0], int(np.prod(size[1:])))
        er_t = np.mean(b["eff_ranks_trained"])
        er_t_std = np.std(b["eff_ranks_trained"])
        er_i = np.mean(b["eff_ranks_init"])
        t1_t = np.mean(b["explained_top1_trained"])
        t1_i = np.mean(b["explained_top1_init"])
        print(f"{str(size):<20} {n:>7,} {max_rank:>10} {er_i:>16.3f} "
              f"{er_t:>10.3f} ± {er_t_std:<5.3f}     {t1_i:>10.4f}      {t1_t:>10.4f}")
        summary.append({"shape": list(size), "n": n, "max_rank": max_rank,
                        "eff_rank_init": er_i, "eff_rank_trained": er_t,
                        "top1_var_init": t1_i, "top1_var_trained": t1_t})

    print("\nInterpretation:")
    print("  - max poss = matrix rank ceiling for the unfold.")
    print("  - eff_rank trained << max poss → tensor has collapsed to lower-rank manifold during training.")
    print("  - top-1 var rising → first singular direction now explains much more of the tensor's energy.")

    with open("/tmp/diagnose_svd.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
