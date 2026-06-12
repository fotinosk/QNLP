from functools import lru_cache
from typing import TypeAlias

import opt_einsum

# Each inner tuple is one tensor's shape, e.g. ((512, 10), (10, 512), ...)
TensorShapes: TypeAlias = tuple[tuple[int, ...], ...]

ContractionPath: TypeAlias = list[tuple[int, int]]

# Largest intermediate tensor (in elements) we allow a contraction to produce.
# Diagrams whose optimal path exceeds this are excluded at dataset-creation time.
MAX_INTERMEDIATE_ELEMENTS = 50_000_000


def get_contraction_path_and_cost(einsum_str: str, shapes: TensorShapes) -> tuple[ContractionPath, int]:
    """Compute the pairwise contraction path and the size (in elements) of the
    largest intermediate tensor it produces.

    Uses opt_einsum's ``shapes=True`` interface so no tensors are allocated —
    only the shape tuples are needed to plan the contraction.
    """
    path, info = opt_einsum.contract_path(einsum_str, *shapes, shapes=True, optimize="branch-2")
    return path, int(info.largest_intermediate)


@lru_cache(maxsize=1000)
def get_contraction_path(einsum_str: str, shapes: TensorShapes) -> ContractionPath:
    """Cached path-only lookup, used as a runtime fallback when a pre-computed
    path is not supplied by the dataset."""
    path, _ = get_contraction_path_and_cost(einsum_str, shapes)
    return path


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
