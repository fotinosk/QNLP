import torch
from torch import nn
from typing import Sequence, Union


class LookupEmbedding(nn.Module):
    def __init__(self, labels: Sequence[str], embedding_dim: int):
        super().__init__()
        self.labels = list(labels)
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.embeddings = nn.Embedding(len(self.labels), embedding_dim)

    @classmethod
    def from_embeddings(cls,
                       labels: Sequence[str],
                       embedding_list: Sequence[torch.Tensor]):
        emb_dim = embedding_list[0].size(-1)
        model = cls(labels, emb_dim)
        stacked = torch.stack(embedding_list)  # [n_labels, emb_dim]
        with torch.no_grad():
            model.embeddings.weight.copy_(stacked)
        return model

    def forward(self, labels: Union[str, Sequence[str]]) -> torch.Tensor:
        if isinstance(labels, str):
            labels = [labels]
        try:
            idxs = [self.label_to_idx[str(lbl)] for lbl in labels]
        except KeyError as e:
            raise ValueError(f"Unknown label: {e.args[0]}") from None
        idxs_tensor = torch.tensor(idxs,
                                 dtype=torch.long,
                                 device=self.embeddings.weight.device)
        return self.embeddings(idxs_tensor)

    def save_to_checkpoint(self, path: str):
        """
        Save a checkpoint dict to `path`, containing:
          - labels (so we can rebuild the mapping)
          - embedding_dim
          - state_dict
        """
        ckpt = {
            'labels': self.labels,
            'embedding_dim': self.embeddings.embedding_dim,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    @classmethod
    def load_from_checkpoint(cls,
             path: str,
             map_location: Union[str, torch.device] = 'cpu'):
        """
        Load from a checkpoint saved via `save()`.
        Returns an ImageEmbedding on the requested device.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ckpt['labels'], ckpt['embedding_dim'])
        model.load_state_dict(ckpt['state_dict'])
        return model