from functools import lru_cache
from typing import TypeAlias

import opt_einsum

# Each inner tuple is one tensor's shape, e.g. ((512, 10), (10, 512), ...)
TensorShapes: TypeAlias = tuple[tuple[int, ...], ...]

ContractionPath: TypeAlias = list[tuple[int, int]]

# Largest intermediate tensor (in elements) we allow a contraction to produce.
# Diagrams whose optimal path exceeds this are excluded at dataset-creation time.
MAX_INTERMEDIATE_ELEMENTS = 50_000_000

# opt_einsum optimizer used to plan contraction paths. "dp" (exact dynamic
# programming) finds the true-optimal order; the "branch-2" heuristic fails on
# tree-reader (NO_TYPE) topologies (huge intermediates / timeouts).
PATH_OPTIMIZER = "dp"


def get_contraction_path_and_cost(einsum_str: str, shapes: TensorShapes) -> tuple[ContractionPath, int]:
    """Compute the pairwise contraction path and the size (in elements) of the
    largest intermediate tensor it produces.

    Uses opt_einsum's ``shapes=True`` interface so no tensors are allocated —
    only the shape tuples are needed to plan the contraction.

    Uses the exact ``dp`` (dynamic-programming optimal) optimizer rather than the
    ``branch-2`` heuristic: for the tree-reader (NO_TYPE) topologies branch-2 fails
    to find the cheap order (largest intermediate blows past the feasibility limit,
    or it times out), whereas dp finds the true optimum — typically a ~512-element
    intermediate — quickly, because tree tensor networks have low contraction width.
    """
    path, info = opt_einsum.contract_path(einsum_str, *shapes, shapes=True, optimize=PATH_OPTIMIZER)
    return path, int(info.largest_intermediate)


@lru_cache(maxsize=1000)
def get_contraction_path(einsum_str: str, shapes: TensorShapes) -> ContractionPath:
    """Cached path-only lookup, used as a runtime fallback when a pre-computed
    path is not supplied by the dataset."""
    path, _ = get_contraction_path_and_cost(einsum_str, shapes)
    return path


def get_right_to_left_path(n_operands: int) -> ContractionPath:
    """Fold-right contraction: always contract the two rightmost remaining operands.

    Produces t0 ⊗ (t1 ⊗ (t2 ⊗ t3)) working from the end of the sentence inward.
    Used as a controlled suboptimal baseline to test path sensitivity.

    In opt_einsum path notation, after contracting (i,j) both operands are removed
    and the result is appended — so indices must account for shrinking list size.
    """
    if n_operands <= 1:
        return []
    return [(i, i + 1) for i in range(n_operands - 2, -1, -1)]


def get_random_path(einsum_str: str, seed: int | None = None) -> ContractionPath:
    """Random contraction order, restricted to pairs that share a bond index.

    At each step, builds the set of connected pairs (those with at least one
    shared index in the current network) and picks uniformly at random from them.
    After each contraction the result's index set is tracked: an index is summed
    away if it appears in both contracted tensors but nowhere else in the remaining
    network; all other indices survive into the result.

    Falls back to a random unconnected pair only if no connected pair exists
    (e.g. disconnected sub-networks), which should not occur in valid CCG diagrams.

    Memory safety is NOT guaranteed here — callers should validate with opt_einsum
    and discard if intermediates exceed MAX_INTERMEDIATE_ELEMENTS.
    """
    import random

    rng = random.Random(seed)
    input_part = einsum_str.split("->")[0]
    index_sets: list[set[str]] = [set(op) for op in input_part.split(",")]
    n = len(index_sets)

    if n <= 1:
        return []

    path: ContractionPath = []
    for _ in range(n - 1):
        current_n = len(index_sets)
        connected = [(i, j) for i in range(current_n) for j in range(i + 1, current_n) if index_sets[i] & index_sets[j]]
        i, j = rng.choice(connected) if connected else tuple(sorted(rng.sample(range(current_n), 2)))

        # Indices that survive into the result: union minus those summed away.
        # An index is summed if it appears only in tensors i and j (not elsewhere).
        other = set().union(*(s for k, s in enumerate(index_sets) if k != i and k != j))
        summed = (index_sets[i] & index_sets[j]) - other
        result_indices = (index_sets[i] | index_sets[j]) - summed

        # opt_einsum convention: remove both operands, append result at end.
        index_sets = [s for k, s in enumerate(index_sets) if k != i and k != j]
        index_sets.append(result_indices)
        path.append((i, j))

    return path


def get_left_to_right_path(n_operands: int) -> ContractionPath:
    """Fold-left contraction: always contract the running result with the next operand.

    Produces ((w0 ⊗ w1) ⊗ w2) ⊗ w3 ... in sentence order. With NLC this ensures
    every gate application corresponds to a linguistically valid partial composition
    rather than a computationally convenient but semantically arbitrary pairing.

    O(n) to generate — no opt_einsum needed.
    """
    if n_operands <= 1:
        return []
    return [(0, 1)] + [(0, n_operands - k) for k in range(2, n_operands)]


def _print_path(einsum_str: str) -> None:
    parts = einsum_str.split("->")
    reprs = parts[0].split(",")
    shapes = tuple(tuple([4] * len(r)) for r in reprs)

    _, info = opt_einsum.contract_path(einsum_str, *shapes, shapes=True, optimize="branch-2")

    for step, contraction in enumerate(info.contraction_list):
        _, _, einsum_str_step, _, _ = contraction
        print(f"  step {step + 1}: {einsum_str_step}")


if __name__ == "__main__":
    for diagram in [
        "ab,bd,d,ai,igl,ln,n,aq,qst,tv,w,wy,yvB,BD,DF,FH,H->s",
        "a,bc,ce,eg,gij,jl,ln,np,p,is,sbv,vay,yAB,BD,E,EG,GI,IK,KD,IP,PD->A",
    ]:
        print(f"\nDiagram: {diagram}")
        _print_path(diagram)
