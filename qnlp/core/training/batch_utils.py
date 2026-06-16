import torch
from torch import Tensor


def drop_nonfinite_rows(outputs: dict[str, Tensor], keys: list[str]) -> tuple[dict[str, Tensor], int]:
    """Drop batch rows where any of the given embedding tensors is non-finite.

    A diagram that was skipped during the non-linear contraction (too-large
    intermediate) yields a NaN embedding; this removes those samples from every
    listed tensor together, keeping image/caption rows aligned before the loss.

    Returns (filtered_outputs, n_dropped). `keys` are filtered by the shared mask;
    other entries in `outputs` are passed through unchanged.
    """
    mask: Tensor | None = None
    for k in keys:
        finite = torch.isfinite(outputs[k]).all(dim=-1)
        mask = finite if mask is None else (mask & finite)

    if mask is None:
        return outputs, 0

    n_dropped = int((~mask).sum().item())
    if n_dropped == 0:
        return outputs, 0

    filtered = {k: (v[mask] if k in keys else v) for k, v in outputs.items()}
    return filtered, n_dropped
