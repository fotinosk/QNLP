import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TripletMarginWithDistanceLoss


class InfoNCE(nn.Module):
    """
    InfoNCE Loss for contrastive learning as described in:
    "A Simple Framework for Contrastive Learning of Visual Representations" (Chen et al., 2020)
    """

    def __init__(self, temperature=0.07, reduction="mean"):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for scaling logits
            reduction: Reduction method for loss ('mean', 'sum', 'none')
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, text_emb, pos_emb, neg_emb=None):
        """
        Calculate InfoNCE loss between text embeddings and image embeddings.

        Args:
            text_emb: Text embeddings [batch_size, embedding_dim]
            pos_emb: Positive image embeddings [batch_size, embedding_dim]
            neg_emb: Optional negative image embeddings [batch_size, num_negatives, embedding_dim]

        Returns:
            Tuple of (loss, accuracy) if compute_accuracy=True, otherwise just loss
        """
        # Normalize embeddings to unit length
        text_emb = F.normalize(text_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)

        batch_size = len(text_emb)
        device = text_emb.device

        if neg_emb is not None:
            # Use explicit negatives
            neg_emb = F.normalize(neg_emb, dim=-1)
            pos_logits = torch.sum(text_emb * pos_emb, dim=-1, keepdim=True) / self.temperature
            neg_logits = torch.bmm(text_emb.unsqueeze(1), neg_emb.transpose(-1, -2)).squeeze(1) / self.temperature
            logits = torch.cat([pos_logits, neg_logits], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            # Use in-batch negatives (standard InfoNCE)
            logits = torch.matmul(text_emb, pos_emb.t()) / self.temperature
            labels = torch.arange(batch_size, device=device)

        loss = self.cross_entropy(logits, labels)

        # Calculate accuracy
        with torch.no_grad():
            if neg_emb is not None:
                # Check if positive similarity > all negative similarities
                correct = (logits.argmax(dim=1) == labels).float().mean()
            else:
                # For in-batch negatives, check if the diagonal elements are the largest in their rows
                correct = (torch.argmax(logits, dim=1) == labels).float().mean()

        return loss, correct


def create_loss_functions(
    temperature: float, hard_neg_distance_function: str, margin: float, swap: bool
) -> tuple[InfoNCE, TripletMarginWithDistanceLoss]:
    contrastive_loss = InfoNCE(temperature=temperature)
    if hard_neg_distance_function == "cosine":
        distance_function = lambda x, y: 1 - nn.CosineSimilarity(dim=-1)(x, y)
    elif hard_neg_distance_function == "euclidean":
        distance_function = nn.PairwiseDistance(p=2)
    else:
        raise ValueError(f"Unknown distance function: {hard_neg_distance_function}")
    hard_neg_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=distance_function,
        margin=margin,
        swap=swap,
    )
    return contrastive_loss, hard_neg_loss
