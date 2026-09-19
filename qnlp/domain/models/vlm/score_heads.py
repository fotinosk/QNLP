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
    ):
        super().__init__()
        self.region_level = region_level
        self.text_proj = nn.Linear(text_dim, score_dim)
        self.region_proj = nn.Linear(region_dim, score_dim)
        self.u = nn.Parameter(torch.randn(rank, score_dim) * score_dim**-0.5)
        self.v = nn.Parameter(torch.randn(rank, score_dim) * score_dim**-0.5)
        self.w = nn.Parameter(torch.randn(rank, n_regions) * n_regions**-0.5)
        # Attention-style scaling: unlike cosine (bounded in [-1,1]), this
        # score's raw magnitude grows with rank and is otherwise
        # uncontrolled, which can destabilise the fixed InfoNCE temperature
        # at init. 1/sqrt(rank) keeps init scores O(1), matching the
        # convention scaled dot-product attention uses for the same reason.
        self._scale = rank**-0.5

    def _factors(self, text_repr: torch.Tensor, image_repr: torch.Tensor):
        t = self.text_proj(text_repr)  # [Bt, D]
        R = self.region_proj(image_repr)  # [Bi, n, D]
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
