import torch


def recall_at_k(sim: torch.Tensor, k: int) -> float:
    """Recall@k where diagonal entries are the true pairs."""
    N = sim.shape[0]
    topk = sim.topk(min(k, N), dim=1).indices
    correct = (topk == torch.arange(N, device=sim.device).unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def retrieval_metrics(
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """
    Compute image→text and text→image Recall@k.

    img_embs / txt_embs: [N, D], L2-normalised. Row i of img_embs is paired
    with row i of txt_embs (the diagonal of the similarity matrix is the
    true-pair score).
    """
    sim = img_embs.cpu() @ txt_embs.cpu().T
    return {
        f"{direction}_R{k}": recall_at_k(matrix, k) for direction, matrix in (("i2t", sim), ("t2i", sim.T)) for k in ks
    }
