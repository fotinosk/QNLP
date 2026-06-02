"""Tree-walking neural composition over CCG trees.

Each WORD leaf maps to a learned vector (nn.Embedding).
Each INTERNAL node (CCG rule) maps to a per-rule MLP f: R^d × R^d → R^d
that takes left and right child vectors and produces the parent vector.

No tensors, no bonds. Pure functional composition guided by the
parser's tree structure.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Map lambeq's verbose rule names to compact ones we share by
_RULE_ALIAS = {
    "FORWARD_APPLICATION": "FA",
    "BACKWARD_APPLICATION": "BA",
    "FORWARD_COMPOSITION": "FC",
    "BACKWARD_COMPOSITION": "BC",
    "FORWARD_CROSSED_COMPOSITION": "FX",
    "BACKWARD_CROSSED_COMPOSITION": "BX",
    "GENERALIZED_FORWARD_COMPOSITION": "GFC",
    "GENERALIZED_BACKWARD_COMPOSITION": "GBC",
    "GENERALIZED_FORWARD_CROSSED_COMPOSITION": "GFX",
    "GENERALIZED_BACKWARD_CROSSED_COMPOSITION": "GBX",
    "FORWARD_TYPE_RAISING": "FTR",
    "BACKWARD_TYPE_RAISING": "BTR",
    "CONJUNCTION": "CONJ",
    "REMOVE_PUNCTUATION_LEFT": "RPL",
    "REMOVE_PUNCTUATION_RIGHT": "RPR",
    "LEXICAL": "LEX",
    "UNARY": "UNARY",
}


def _rule_alias(name: str) -> str:
    return _RULE_ALIAS.get(name, name)


class _RuleMLP(nn.Module):
    """One small MLP per rule name. Takes (left, right) ∈ R^d × R^d → R^d."""

    def __init__(self, d: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, d),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1))


class TreeNeuralComposer(nn.Module):
    """Compose CCG trees through per-rule neural functions.

    Args:
      vocab: list of unique lemmas (the words appearing in any tree)
      rule_names: list of unique CCG rule names (in alias form: FA, BA, ...)
      d: word-vector dimension (also model output dim)
      hidden: MLP hidden width
      unary_as_identity: if True, unary rules (1 child) just pass through
    """

    def __init__(
        self,
        vocab: List[str],
        rule_names: List[str],
        d: int = 512,
        hidden: int = 512,
        unary_as_identity: bool = True,
    ):
        super().__init__()
        self.d = d
        self.vocab = list(vocab)
        self.lemma_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.unk_idx = len(self.vocab)  # extra UNK row
        self.word_emb = nn.Embedding(len(self.vocab) + 1, d)
        nn.init.normal_(self.word_emb.weight, std=0.02)

        self.rule_names = list(rule_names)
        self.rule_mlps = nn.ModuleDict({
            r: _RuleMLP(d, hidden) for r in self.rule_names
        })
        # Fallback for rules we didn't see at init time
        self.unary_as_identity = unary_as_identity
        self.fallback_mlp = _RuleMLP(d, hidden)

    def _encode(self, node: dict) -> torch.Tensor:
        """Recursively walk a serialized CCG tree dict."""
        if "leaf" in node:
            lemma = node.get("lemma") or node["leaf"]
            idx = self.lemma_to_idx.get(lemma.lower(), self.unk_idx)
            return self.word_emb(torch.tensor(idx, device=self.word_emb.weight.device))

        children = node.get("children", [])
        rule_name = _rule_alias(node.get("rule", "?"))
        if len(children) == 1:
            inner = self._encode(children[0])
            if self.unary_as_identity:
                return inner
            else:
                # treat unary as a binary with zero right-child
                zero = torch.zeros_like(inner)
                mlp = self.rule_mlps[rule_name] if rule_name in self.rule_mlps else self.fallback_mlp
                return mlp(inner, zero)
        if len(children) == 2:
            l = self._encode(children[0])
            r = self._encode(children[1])
            mlp = self.rule_mlps[rule_name] if rule_name in self.rule_mlps else self.fallback_mlp
            return mlp(l, r)
        # 3+ children: fold left-associatively
        out = self._encode(children[0])
        mlp = self.rule_mlps[rule_name] if rule_name in self.rule_mlps else self.fallback_mlp
        for c in children[1:]:
            out = mlp(out, self._encode(c))
        return out

    def forward(self, trees: List[dict]) -> torch.Tensor:
        """Encode a batch of trees → [B, d] tensor."""
        encs = []
        for t in trees:
            if t is None:
                encs.append(torch.zeros(self.d, device=self.word_emb.weight.device))
                continue
            v = self._encode(t)
            encs.append(F.normalize(v, dim=-1))
        return torch.stack(encs)
