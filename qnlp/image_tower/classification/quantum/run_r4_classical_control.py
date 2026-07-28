"""Task R4 (roadmap Section 7): classical CP-TTN control -- also closes Question A.3.

The whole quantum investigation to date is quantum-vs-quantum: there is not one
run of a matched classical tensor-network model on the same data at the same
size. Without that, no result here can support a claim about quantum models
*relative to their classical analogue*, which is the thesis's actual claim
territory. This is the largest structural gap identified by the 2026-07-27 audit.

It is simultaneously Question A.3 -- "does the unitary constraint (U^dag U = I)
limit representation capacity versus unconstrained classical CP factor weights?"
-- which is Section 6 backlog item 3 and the only substantive un-run item in
Question A. A quantum node applies a unitary and traces out; a classical CP node
applies unconstrained factor matrices and contracts. Matching structure and
parameter count isolates that constraint.

Three arms, identical data/seeds/protocol, same 16 patches -> 4 nodes -> 1 root
quad-tree:

  quantum        -- the architecture of record (phase15_common.ARCH).
  classical_bare -- faithful analogue: CP quad-node, NO residual, NO dropout,
                    matching the quantum model's constraints. This is the arm
                    that answers A.3.
  classical_full -- the repo's actual classical node (`cp_node.CPQuadRankLayer`)
                    with residual + dropout, i.e. what classical practice does.
                    Included because the existing classical tower uses both and
                    got good results, while this investigation has issued rules
                    against both for the quantum tower -- that divergence needs
                    an explicit measurement, not an assumption.

Rank is chosen to match the quantum model's parameter count as closely as
possible, so the comparison is capacity-matched rather than size-confounded.

Run: conda run -n qnlp python -m qnlp.image_tower.classification.quantum.run_r4_classical_control
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from qnlp.discoviz.models.cp_node import CPQuadRankLayer
from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.image_tower.classification.quantum.qttn_core import HierarchicalQTTNClassifier


class ClassicalTTNClassifier(nn.Module):
    """Classical CP quad-tree with the same topology as the quantum QTTN.

    16 patches -> 4 level-1 nodes (4 children each) -> 1 root -> linear head.
    `bond_dim` is the classical analogue of the quantum model's surviving-qubit
    width: the quantum tree passes 1 qubit (a 2-dimensional bond) between levels,
    so bond_dim=2 is the structurally faithful choice. Question A.2 established
    this equivalence (chi=2 per internal bond).
    """

    def __init__(
        self, rank=8, bond_dim=2, enc_dim=3, img_size=16, patch_size=4, n_classes=4, use_residual=False, dropout_p=0.0
    ):
        super().__init__()
        assert img_size // patch_size == 4, "level-1 grouping assumes a 4x4 patch grid"
        # Same patch encoder as the quantum tower's I/O boundary.
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, enc_dim)
        self.level1 = CPQuadRankLayer(
            num_nodes=4, in_dim=enc_dim, out_dim=bond_dim, rank=rank, use_residual=use_residual, dropout_p=dropout_p
        )
        self.level2 = CPQuadRankLayer(
            num_nodes=1, in_dim=bond_dim, out_dim=bond_dim, rank=rank, use_residual=use_residual, dropout_p=dropout_p
        )
        self.head = nn.Linear(bond_dim, n_classes)

    @staticmethod
    def _group_2x2(grid):
        b, h, w, d = grid.shape
        grid = grid.view(b, h // 2, 2, w // 2, 2, d)
        grid = grid.permute(0, 1, 3, 2, 4, 5).contiguous()
        return grid.view(b, (h // 2) * (w // 2), 4, d)

    def forward(self, x, p_noise=0.0):  # p_noise ignored: no quantum channel here
        b = x.shape[0]
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, 16, 48)
        feats = torch.tanh(self.patch_embed(x))
        feats = feats.view(b, 4, 4, -1)
        x1 = self._group_2x2(feats)  # [B, 4, 4, enc_dim]
        n1 = self.level1(x1)  # [B, 4, bond_dim]
        n2 = self.level2(n1.unsqueeze(1))  # [B, 1, bond_dim]
        return self.head(n2.squeeze(1))


class MLPReference(nn.Module):
    """Task-difficulty reference. NOT a tensor network and NOT an architecture
    proposal -- it exists only to calibrate how hard 16x16 overlapping shapes
    actually is for a classical model of comparable size.

    Without it, a low classical-CP score is uninterpretable: it could mean the
    unitarity constraint helps (a real result), or simply that CPQuadRankLayer's
    degree-4 multiplicative merge is a poor fit for this task (an artifact).
    If this reference scores high while classical_bare scores low, the CP node
    is the problem, not classical computation, and no quantum-advantage claim
    can be made from R4.
    """

    def __init__(self, hidden=16, enc_dim=3, patch_size=4, n_classes=4):
        super().__init__()
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, enc_dim)
        self.net = nn.Sequential(nn.Linear(16 * enc_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x, p_noise=0.0):
        b = x.shape[0]
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, 16, 48)
        return self.net(torch.tanh(self.patch_embed(x)).flatten(1))


def tune_classical(
    use_residual, dropout_p, q_params, seeds=(0, 1, 2), lrs=(0.003, 0.01, 0.03, 0.1), bond_dims=(2, 4, 8)
):
    """Give the classical arm a fair hyperparameter search before comparing.

    Necessary for the comparison to mean anything. The first R4 attempt used the
    quantum model's lr (0.03) and the structurally-faithful bond_dim=2 for the
    classical arm and got 25.8% -- random chance for 4 classes. A dead baseline
    would have produced a spectacular and entirely fake "quantum beats classical
    by 57.8 points" headline.

    Note on fairness: the quantum arm's settings were themselves tuned across
    R1/R2 (readout, encoding, protocol, lr), so sweeping the classical arm is
    equal treatment, not a handicap. bond_dim is included because it is the
    classical analogue of the quantum readout width -- the exact axis that
    turned out to be binding for the quantum model in R1/R1b, so it would be
    inconsistent to fix it at its narrowest value here.
    """
    best = None
    for lr in lrs:
        for bd in bond_dims:
            rank = match_rank_to(q_params, bond_dim=bd, use_residual=use_residual, dropout_p=dropout_p)
            scores = []
            for s in seeds:
                c, _ = pc.train_run(
                    lambda: ClassicalTTNClassifier(
                        rank=rank, bond_dim=bd, use_residual=use_residual, dropout_p=dropout_p
                    ),
                    seed=s,
                    lr=lr,
                )
                scores.append(sum(c[-pc.SCORE_LAST_K :]) / pc.SCORE_LAST_K)
            mean = float(np.mean(scores))
            print(f"    lr={lr:<6} bond_dim={bd:<2} rank={rank:<3} -> {mean:.1f}%", flush=True)
            if best is None or mean > best["score"]:
                best = {"lr": lr, "bond_dim": bd, "rank": rank, "score": mean}
    print(
        f"  best: lr={best['lr']} bond_dim={best['bond_dim']} rank={best['rank']} " f"({best['score']:.1f}%)",
        flush=True,
    )
    return best


def match_rank_to(target_params, **kw):
    """Smallest rank whose parameter count is >= the quantum model's, so the
    comparison is capacity-matched rather than size-confounded."""
    best = 1
    for r in range(1, 65):
        n = sum(p.numel() for p in ClassicalTTNClassifier(rank=r, **kw).parameters())
        best = r
        if n >= target_params:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds-per-arm",
        type=int,
        default=21,
        help="21 resolves ~5-pt effects; quantum-vs-classical gaps are expected "
        "to be much larger than the 2-3 pt intra-quantum effects R3 chases.",
    )
    args = ap.parse_args()
    seeds = list(range(args.seeds_per_arm))

    q_params = sum(p.numel() for p in HierarchicalQTTNClassifier(**pc.ARCH).parameters())
    print(f"Quantum model: {q_params} params (lr={pc.PROTOCOL['lr']}, tuned across R1/R2).")

    print("\nTuning classical_bare (fair-shot hyperparameter search):", flush=True)
    tuned_bare = tune_classical(False, 0.0, q_params)
    print("\nTuning classical_full:", flush=True)
    tuned_full = tune_classical(True, 0.1, q_params)

    specs = {
        "quantum": (lambda: HierarchicalQTTNClassifier(**pc.ARCH), pc.PROTOCOL["lr"], None),
        "classical_bare": (
            lambda: ClassicalTTNClassifier(
                rank=tuned_bare["rank"], bond_dim=tuned_bare["bond_dim"], use_residual=False, dropout_p=0.0
            ),
            tuned_bare["lr"],
            tuned_bare,
        ),
        "classical_full": (
            lambda: ClassicalTTNClassifier(
                rank=tuned_full["rank"], bond_dim=tuned_full["bond_dim"], use_residual=True, dropout_p=0.1
            ),
            tuned_full["lr"],
            tuned_full,
        ),
        # Task-difficulty references; see MLPReference docstring. Two sizes because
        # hidden=16 is NOT parameter-matched to the quantum arm (999 vs 211), so a
        # matched-size variant is needed before any "beats the quantum model"
        # statement can be made without a size confound.
        "mlp_reference": (lambda: MLPReference(hidden=16), 0.01, None),
        "mlp_param_matched": (lambda: MLPReference(hidden=2), 0.01, None),
    }

    arms = []
    for name, (factory, lr, tuned) in specs.items():
        print(f"\n--- {name} ({args.seeds_per_arm} seeds, lr={lr}) ---", flush=True)
        curves = []
        for s in seeds:
            c, _ = pc.train_run(factory, seed=s, lr=lr)
            curves.append(c)
            print(f"  seed {s:>2}: score={sum(c[-pc.SCORE_LAST_K:])/pc.SCORE_LAST_K:.1f}%", flush=True)
        arms.append(
            pc.summarise(name, curves, lr=lr, tuned=tuned, num_params=sum(p.numel() for p in factory().parameters()))
        )

    by = {a["config"]: a for a in arms}
    cmps = [
        pc.compare(by["quantum"], by["classical_bare"]),
        pc.compare(by["quantum"], by["classical_full"]),
        pc.compare(by["classical_bare"], by["classical_full"]),
    ]

    pc.print_arms(
        f"R4: quantum vs classical CP-TTN, 16x16, 1024/30, "
        f"{args.seeds_per_arm} seeds (classical arms hyperparameter-tuned)",
        arms,
    )
    print(f"\n{'arm':<20}{'params':>10}")
    for a in arms:
        print(f"{a['config']:<20}{a['num_params']:>10}")
    pc.print_comparisons(cmps)

    # Interpretability gate: is the CP node the problem, or classical computation?
    mlp = by["mlp_reference"]
    cp_vs_mlp = pc.compare(by["classical_bare"], mlp)
    q_vs_mlp = pc.compare(by["quantum"], mlp)
    q_vs_mlp_matched = pc.compare(by["quantum"], by["mlp_param_matched"])
    interpretable = not (cp_vs_mlp["resolved"] and cp_vs_mlp["difference"] > 0)

    a3 = cmps[0]
    if not a3["resolved"]:
        a3_verdict = (
            f"Question A.3: NO RESOLVED DIFFERENCE between the unitary-constrained "
            f"quantum node and the unconstrained classical CP node "
            f"({a3['difference']:+.1f} pts vs a {a3['min_detectable_effect']:.1f}-pt limit). "
            f"At matched topology and parameter count, the unitarity constraint does not "
            f"measurably cost representation capacity on this task -- a genuinely useful "
            f"result for the thesis, and the first quantum-vs-classical statement the "
            f"investigation can actually support."
        )
    elif a3["difference"] > 0:
        a3_verdict = (
            f"Question A.3: the unconstrained classical CP node BEATS the unitary quantum "
            f"node by {a3['difference']:.1f} pts (limit {a3['min_detectable_effect']:.1f}). "
            f"The unitarity constraint costs measurable capacity at matched size. State "
            f"this plainly in the thesis -- it bounds what the quantum tower can claim."
        )
    else:
        a3_verdict = (
            f"Question A.3: the quantum node BEATS the unconstrained classical CP node by "
            f"{-a3['difference']:.1f} pts (limit {a3['min_detectable_effect']:.1f}) at "
            f"matched parameter count. This is the strongest possible form of the thesis "
            f"claim -- verify it is not an optimisation artifact (try tuning the classical "
            f"arm's lr/rank) before relying on it."
        )
    if not interpretable:
        a3_verdict = (
            "Question A.3: NOT ANSWERABLE from this run. The MLP reference "
            f"({mlp['score_mean']:.1f}%) beats the classical CP arm "
            f"({by['classical_bare']['score_mean']:.1f}%) by {cp_vs_mlp['difference']:.1f} pts, "
            f"above the {cp_vs_mlp['min_detectable_effect']:.1f}-pt limit -- so the CP quad-node, "
            "not classical computation, is what is underperforming. Any 'quantum beats classical' "
            "reading of the quantum-vs-classical_bare gap would be an artifact of a mis-specified "
            "baseline. A.3 needs a classical TTN baseline that is at least competitive with this "
            "reference before the unitarity question can be asked. DO NOT cite the raw gap."
        )
    print(f"\n{a3_verdict}")
    print(
        f"\nInterpretability check: classical_bare -> mlp_reference "
        f"{cp_vs_mlp['difference']:+.1f} pts ({cp_vs_mlp['verdict']}); "
        f"A.3 {'answerable' if interpretable else 'NOT answerable'} from this run."
    )

    res_cmp = cmps[2]
    print(
        f"\nClassical residual+dropout vs bare: {res_cmp['difference']:+.1f} pts "
        f"({res_cmp['verdict']}). Context: the quantum tower has binding rules against both "
        f"mechanisms; this measures whether they even help the classical analogue at this scale."
    )

    print(
        f"\nQuantum vs MLP reference (999 params, NOT size-matched): "
        f"{q_vs_mlp['difference']:+.1f} pts ({q_vs_mlp['verdict']})"
    )
    print(
        f"Quantum vs MLP param-matched ({by['mlp_param_matched']['num_params']} params): "
        f"{q_vs_mlp_matched['difference']:+.1f} pts ({q_vs_mlp_matched['verdict']})"
    )
    if q_vs_mlp_matched["resolved"] and q_vs_mlp_matched["difference"] > 0:
        print(
            "  ^ A size-matched classical MLP beats the quantum architecture of record on this\n"
            "    task. This does not invalidate the quantum work, but it bounds what the thesis\n"
            "    can claim and MUST be stated explicitly rather than omitted -- it is the first\n"
            "    quantum-vs-classical measurement the investigation has ever had."
        )

    pc.save(
        {
            "architecture": pc.ARCH,
            "protocol": pc.PROTOCOL,
            "tuned_classical_bare": tuned_bare,
            "tuned_classical_full": tuned_full,
            "quantum_params": q_params,
            "seeds_per_arm": args.seeds_per_arm,
            "arms": arms,
            "comparisons": cmps,
            "question_a3_verdict": a3_verdict,
            "cp_vs_mlp_reference": cp_vs_mlp,
            "a3_answerable": interpretable,
            "quantum_vs_mlp": q_vs_mlp,
            "quantum_vs_mlp_param_matched": q_vs_mlp_matched,
        },
        "r4_classical_control_results.json",
    )


if __name__ == "__main__":
    main()
