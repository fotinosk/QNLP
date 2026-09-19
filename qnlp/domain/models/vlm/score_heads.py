"""
Structured scoring heads for ContrastiveVLM — Routes B and A from
TTN_CIFAR_EXPERIMENTS.md's "Implementation spec: Routes N, B, A".

Both replace the default pooled-vector cosine comparison with a comparison
that keeps image regions (from TTNImageModel.forward_regions, P1) separate,
so the discriminative burden isn't collapsed onto a single scalar before
scoring. Selected via ContrastiveVLM(score_head=...); the cosine default
(score_head=None) is untouched and remains bit-for-bit unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreHead(nn.Module):
    """Interface every structured score head implements.

    needs_regions: if True, ContrastiveVLM calls image_model.forward_regions
        instead of the pooled head — image_repr is [B, n_regions, region_dim].
    needs_roles: if True, ContrastiveVLM calls text_model.forward_roles
        instead of the pooled caption vector — text_repr is the dict
        {"subj", "obj", "verb_left", "verb_mid", "verb_right"} from
        EinsumModel.forward_roles.
    """

    needs_regions: bool = False
    needs_roles: bool = False
    region_level: int = 1

    def score_matrix(self, text_repr, image_repr) -> torch.Tensor:
        """Full [Bt, Bi] cross-score matrix, for InfoNCE-style in-batch negatives."""
        raise NotImplementedError

    def score_pairs(self, text_repr, image_repr) -> torch.Tensor:
        """Row-aligned [B] scores (Bt == Bi, matched pairs) — used for the
        explicit hard-negative triplet term."""
        raise NotImplementedError


class TrilinearScoreHead(ScoreHead):
    """Route B — CP-factorised trilinear score, no sentence-structure assumption.

    score(t, R) = sum_j sum_r (u_r . t) (v_r . R_j) (w_r)_j

    Multilinear in both the pooled caption vector t and the region tensor R.
    Deliberately NOT a plain weighted sum of per-region dot products — that
    collapses algebraically to pooling regions first, i.e. the existing
    cosine head. The `w` factor (indexed by region) is what makes the score
    depend on *which* region matches.
    """

    needs_regions = True
    needs_roles = False

    def __init__(
        self,
        text_dim: int,
        region_dim: int,
        n_regions: int,
        score_dim: int = 128,
        rank: int = 32,
        region_level: int = 1,
        tie_uv: bool = False,
        normalize_terms: bool = False,
    ):
        super().__init__()
        self.region_level = region_level
        self.text_proj = nn.Linear(text_dim, score_dim)
        self.region_proj = nn.Linear(region_dim, score_dim)
        self.u = nn.Parameter(torch.randn(rank, score_dim) * score_dim**-0.5)
        # V4 (capacity reduction): tie_uv shares one factor between the text
        # and region projections instead of learning them independently,
        # halving that part of the head's parameter count.
        self.v = self.u if tie_uv else nn.Parameter(torch.randn(rank, score_dim) * score_dim**-0.5)
        self.w = nn.Parameter(torch.randn(rank, n_regions) * n_regions**-0.5)
        # V4: L2-normalise t and each R_j before scoring, so every term in
        # the sum is a bounded cosine-like quantity instead of an
        # unconstrained dot product — a direct capacity/stability control,
        # not a correctness fix.
        self.normalize_terms = normalize_terms
        # Attention-style scaling: unlike cosine (bounded in [-1,1]), this
        # score's raw magnitude grows with rank and is otherwise
        # uncontrolled, which can destabilise the fixed InfoNCE temperature
        # at init. 1/sqrt(rank) keeps init scores O(1), matching the
        # convention scaled dot-product attention uses for the same reason.
        self._scale = rank**-0.5

    def _factors(self, text_repr: torch.Tensor, image_repr: torch.Tensor):
        t = self.text_proj(text_repr)  # [Bt, D]
        R = self.region_proj(image_repr)  # [Bi, n, D]
        if self.normalize_terms:
            t = F.normalize(t, dim=-1)
            R = F.normalize(R, dim=-1)
        ut = t @ self.u.t()  # [Bt, rank]
        vR = torch.einsum("ind,rd->inr", R, self.v)  # [Bi, n, rank]
        wvR = torch.einsum("inr,rn->ir", vR, self.w)  # [Bi, rank]
        return ut, wvR

    def score_matrix(self, text_repr, image_repr):
        ut, wvR = self._factors(text_repr, image_repr)
        return (ut @ wvR.t()) * self._scale  # [Bt, Bi]

    def score_pairs(self, text_repr, image_repr):
        ut, wvR = self._factors(text_repr, image_repr)  # row-aligned, Bt == Bi
        return (ut * wvR).sum(-1) * self._scale


class RoleGroundedScoreHead(ScoreHead):
    """Route A — role-grounded cross-modal contraction.

    g_subj[j] = <P_s . s, R_j>        affinity of the subject to region j
    g_obj [k] = <P_o . o, R_k>        affinity of the object  to region k
    score = sum_{j,k} g_subj[j] * M(v)[j,k] * g_obj[k]

    M(v) is built from the verb's own local MPS chain (EinsumModel.
    forward_roles / get_verb_chain). Phase 0 found the verb tensor is
    neither of the spec's anticipated cases (not rank-1, and not a single
    dense rank-3 tensor) but a bond_dim-factored chain whose two outer legs
    are full embedding_dim (e.g. 512), making a literal dense rank-3
    materialisation (512^3 elements) infeasible per-row at batch scale.
    Fix: project the chain's outer legs into n_regions-space *before* the
    internal bond contraction, and reduce the (also embedding_dim-sized)
    middle "sentence" leg to a scalar via a learned weight vector — exactly
    equivalent to projecting the fully materialised tensor afterward, since
    tensor contraction is multilinear/associative, but touches only
    O(bond_dim * n_regions) intermediate values instead of O(embedding_dim^3).

    v1 scope: only the mainline 3-piece transitive-verb chain (see
    EinsumModel.forward_roles) is supported.
    """

    needs_regions = True
    needs_roles = True

    def __init__(
        self,
        noun_dim: int,
        verb_leg_dim: int,
        region_dim: int,
        n_regions: int,
        score_dim: int = 128,
        region_level: int = 1,
    ):
        super().__init__()
        self.region_level = region_level
        self.n_regions = n_regions
        self.region_proj = nn.Linear(region_dim, score_dim)
        self.subj_proj = nn.Linear(noun_dim, score_dim)
        self.obj_proj = nn.Linear(noun_dim, score_dim)
        self.verb_left_proj = nn.Linear(verb_leg_dim, n_regions)
        self.verb_right_proj = nn.Linear(verb_leg_dim, n_regions)
        self.verb_mid_weight = nn.Parameter(torch.empty(verb_leg_dim))
        nn.init.normal_(self.verb_mid_weight, std=verb_leg_dim**-0.5)
        # Attention-style scaling — see TrilinearScoreHead's `_scale` comment;
        # here the score is a sum over n_regions^2 terms, so 1/n_regions
        # keeps init scores O(1) instead of growing with n_regions^2.
        self._scale = 1.0 / n_regions

    def _verb_matrix(self, verb_left: torch.Tensor, verb_mid: torch.Tensor, verb_right: torch.Tensor) -> torch.Tensor:
        # verb_left:  [B, n_r_dim, bond]      (semantic leg, then trailing bond)
        # verb_mid:   [B, bond, s_dim, bond]  (leading bond, semantic leg, trailing bond)
        # verb_right: [B, bond, n_l_dim]      (leading bond, then semantic leg)
        left_p = self.verb_left_proj(verb_left.transpose(1, 2)).transpose(1, 2)  # [B, n_regions, bond]
        right_p = self.verb_right_proj(verb_right)  # [B, bond, n_regions]
        mid_r = torch.einsum("basc,s->bac", verb_mid, self.verb_mid_weight)  # [B, bond, bond]
        m = torch.bmm(left_p, mid_r)  # [B, n_regions, bond]
        m = torch.bmm(m, right_p)  # [B, n_regions, n_regions]
        return m

    def _project(self, text_repr: dict, image_repr: torch.Tensor):
        s = self.subj_proj(text_repr["subj"])  # [Bt, D]
        o = self.obj_proj(text_repr["obj"])  # [Bt, D]
        M = self._verb_matrix(text_repr["verb_left"], text_repr["verb_mid"], text_repr["verb_right"])  # [Bt,n,n]
        R = self.region_proj(image_repr)  # [Bi, n, D]
        return s, o, M, R

    def score_matrix(self, text_repr, image_repr):
        s, o, M, R = self._project(text_repr, image_repr)
        g_subj = torch.einsum("td,ijd->tij", s, R)  # [Bt, Bi, n]
        g_obj = torch.einsum("td,ikd->tik", o, R)  # [Bt, Bi, n]
        return torch.einsum("tij,tjk,tik->ti", g_subj, M, g_obj) * self._scale

    def score_pairs(self, text_repr, image_repr):
        s, o, M, R = self._project(text_repr, image_repr)  # row-aligned, Bt == Bi
        g_subj = torch.einsum("td,tjd->tj", s, R)
        g_obj = torch.einsum("td,tkd->tk", o, R)
        return torch.einsum("tj,tjk,tk->t", g_subj, M, g_obj) * self._scale


class GatedResidualTrilinearScoreHead(TrilinearScoreHead):
    """Route B variation V1 — gated residual trilinear.

    score = cosine(t, pooled(R)) + gate * trilinear(t, R)     gate init 0.0

    B1 (plain TrilinearScoreHead) *replaced* the cosine comparison outright,
    discarding the 0.5323 baseline behaviour instead of building on it — a
    real result (SVO-Probes 0.5168) diagnosed as memorising an absolute
    region index (see TTN_CIFAR_EXPERIMENTS.md's "Route B variations").
    Every change that has worked in this project has instead been a strict
    generalisation of the working model, gate-initialised to recover it
    exactly (A1's isometric init, B1's opt-in flag, Route N's gate-init-0).
    Here "the working model" is a cosine comparison of the pooled caption
    vector against the mean-pooled region tensor (both in this head's own
    score_dim projection) — at gate=0 the additive trilinear term vanishes
    and only that cosine term survives.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = nn.Parameter(torch.zeros(1))

    def _cosine_terms(self, text_repr: torch.Tensor, image_repr: torch.Tensor):
        t = F.normalize(self.text_proj(text_repr), dim=-1)  # [Bt, D]
        r_pooled = F.normalize(self.region_proj(image_repr).mean(dim=1), dim=-1)  # [Bi, D]
        return t, r_pooled

    def score_matrix(self, text_repr, image_repr):
        t, r_pooled = self._cosine_terms(text_repr, image_repr)
        cosine = t @ r_pooled.t()  # [Bt, Bi]
        trilinear = super().score_matrix(text_repr, image_repr)
        return cosine + self.gate * trilinear

    def score_pairs(self, text_repr, image_repr):
        t, r_pooled = self._cosine_terms(text_repr, image_repr)
        cosine = (t * r_pooled).sum(-1)
        trilinear = super().score_pairs(text_repr, image_repr)
        return cosine + self.gate * trilinear


class BornRuleScoreHead(ScoreHead):
    """Route B variation V2 — Born-rule region pooling (best-motivated).

    score(t, R) = sum_j |<t, R_j>|^2 = ||R t||^2

    Permutation-invariant over regions (no absolute-position memorisation
    surface, unlike V1/B1's `w_r` factor), and quadratic — not linear — in
    the regions, so it does not collapse to the degenerate pooling-first
    case the Route B spec warns against. The natural generalisation of
    cosine similarity from a single image vector to a *set* of regional
    states, and the same Born-rule principle Route N validated one level
    down in the tree.
    """

    needs_regions = True
    needs_roles = False

    def __init__(self, text_dim: int, region_dim: int, n_regions: int, score_dim: int = 128, region_level: int = 1):
        super().__init__()
        self.region_level = region_level
        self.n_regions = n_regions
        self.text_proj = nn.Linear(text_dim, score_dim)
        self.region_proj = nn.Linear(region_dim, score_dim)

    def _terms(self, text_repr: torch.Tensor, image_repr: torch.Tensor):
        t = F.normalize(self.text_proj(text_repr), dim=-1)  # [Bt, D]
        R = F.normalize(self.region_proj(image_repr), dim=-1)  # [Bi, n, D]
        return t, R

    def score_matrix(self, text_repr, image_repr):
        t, R = self._terms(text_repr, image_repr)
        dots = torch.einsum("td,ind->tin", t, R)  # [Bt, Bi, n]
        return (dots**2).sum(-1) / self.n_regions

    def score_pairs(self, text_repr, image_repr):
        t, R = self._terms(text_repr, image_repr)  # row-aligned, Bt == Bi
        dots = torch.einsum("td,tnd->tn", t, R)  # [B, n]
        return (dots**2).sum(-1) / self.n_regions


class AggregationScoreHead(ScoreHead):
    """Route B variation V3 — position-agnostic aggregation via max / LSE.

    score(t, R) = sum_r (u_r . t) * agg_j (v_r . R_j)

    Drops the region-indexed `w_r` factor entirely (the diagnosed cause of
    B1's memorisation) in favour of an aggregation that is invariant to
    *which* region matches, only *whether* one does — the bias SVO-Probes
    actually needs ("does this entity appear anywhere in the image").
    `agg="lse"` (log-sum-exp) is a smooth relaxation of max, for cases
    where max's sparse gradients stall training.
    """

    needs_regions = True
    needs_roles = False

    def __init__(
        self,
        text_dim: int,
        region_dim: int,
        n_regions: int,
        score_dim: int = 128,
        rank: int = 32,
        region_level: int = 1,
        agg: str = "max",
    ):
        super().__init__()
        if agg not in ("max", "lse"):
            raise ValueError(f"agg must be 'max' or 'lse', got {agg!r}")
        self.region_level = region_level
        self.agg = agg
        self.text_proj = nn.Linear(text_dim, score_dim)
        self.region_proj = nn.Linear(region_dim, score_dim)
        self.u = nn.Parameter(torch.randn(rank, score_dim) * score_dim**-0.5)
        self.v = nn.Parameter(torch.randn(rank, score_dim) * score_dim**-0.5)
        self._scale = rank**-0.5

    def _aggregate(self, vR: torch.Tensor) -> torch.Tensor:
        # vR: [Bi, n, rank] -> [Bi, rank], aggregating over the region axis.
        if self.agg == "max":
            return vR.max(dim=1).values
        return torch.logsumexp(vR, dim=1)

    def _factors(self, text_repr: torch.Tensor, image_repr: torch.Tensor):
        t = self.text_proj(text_repr)  # [Bt, D]
        R = self.region_proj(image_repr)  # [Bi, n, D]
        ut = t @ self.u.t()  # [Bt, rank]
        vR = torch.einsum("ind,rd->inr", R, self.v)  # [Bi, n, rank]
        agg = self._aggregate(vR)  # [Bi, rank]
        return ut, agg

    def score_matrix(self, text_repr, image_repr):
        ut, agg = self._factors(text_repr, image_repr)
        return (ut @ agg.t()) * self._scale  # [Bt, Bi]

    def score_pairs(self, text_repr, image_repr):
        ut, agg = self._factors(text_repr, image_repr)  # row-aligned, Bt == Bi
        return (ut * agg).sum(-1) * self._scale
