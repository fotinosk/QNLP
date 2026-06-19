import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from qnlp.core.training.batch_utils import drop_nonfinite_rows
from qnlp.core.training.losses.single_caption import SingleCaptionLoss


class HardNegativeMiner:
    """
    Maintains a pool of hard negative text embeddings sampled from the full
    training set. After build(), pool contains the text embeddings that are
    most confusable (highest image-text similarity) across the dataset,
    excluding true-positive pairs.

    These are used to augment the i2t similarity matrix during training —
    replacing easy random in-batch negatives with globally hard ones.
    """

    def __init__(self, pool_size: int = 4096):
        self.pool_size = pool_size
        self._pool: Tensor | None = None

    @property
    def ready(self) -> bool:
        return self._pool is not None

    @torch.no_grad()
    def build(self, model: nn.Module, loader: DataLoader, device: torch.device) -> None:
        model.eval()
        all_img, all_txt = [], []
        for batch in loader:
            images = batch["local_image_path"].to(device)
            outputs = model(images, batch["caption"])
            img_e = outputs["image_embeddings"]
            txt_e = outputs["true_caption_embeddings"]
            finite = torch.isfinite(img_e).all(-1) & torch.isfinite(txt_e).all(-1)
            all_img.append(img_e[finite].cpu())
            all_txt.append(txt_e[finite].cpu())
        model.train()

        img = torch.cat(all_img)  # [N, D]
        txt = torch.cat(all_txt)  # [N, D]
        N = img.shape[0]

        # Similarity matrix; mask diagonal so true positives are excluded
        sim = img @ txt.T  # [N, N]
        sim.fill_diagonal_(-1e9)

        # For each image, collect top-16 hardest text negatives
        k = min(16, N - 1)
        topk_idx = sim.topk(k, dim=1).indices.flatten().unique()  # unique hard neg indices

        if len(topk_idx) > self.pool_size:
            perm = torch.randperm(len(topk_idx))[: self.pool_size]
            topk_idx = topk_idx[perm]

        self._pool = txt[topk_idx]  # [P, D]

    def sample(self, k: int, device: torch.device) -> Tensor | None:
        if self._pool is None:
            return None
        n = min(k, len(self._pool))
        idx = torch.randperm(len(self._pool))[:n]
        return self._pool[idx].to(device)


class MultiCaptionHardNegStep:
    """
    Training step for multi-caption COCO with hard negative mining.

    Phase 1 — warmup (epoch ≤ hard_neg_warmup_epochs):
        Plain symmetric InfoNCE with in-batch negatives only.

    Phase 2 — mining active (epoch > hard_neg_warmup_epochs):
        After warmup, and then every hard_neg_refresh_epochs epochs, a full
        pass over the train loader builds a pool of hard negative text
        embeddings (globally most confusable pairs). Each batch's i2t logits
        are augmented with hard_neg_sample_k samples from this pool, exposing
        the image encoder to much harder negatives than in-batch random.

    Multi-caption behaviour:
        No change is needed in the step — the dataset already provides all 5
        captions as separate rows (SingleCaptionStrategy is a pass-through).
        With shuffle=True, each epoch sees each image paired with a randomly
        ordered caption. Deduplication for retrieval eval is handled in run.py.
    """

    def __init__(
        self,
        loss_fn: SingleCaptionLoss,
        train_loader: DataLoader,
        device: torch.device,
        hard_neg_warmup_epochs: int = 5,
        hard_neg_refresh_epochs: int = 3,
        hard_neg_pool_size: int = 4096,
        hard_neg_sample_k: int = 256,
    ):
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.device = device
        self.hard_neg_warmup_epochs = hard_neg_warmup_epochs
        self.hard_neg_refresh_epochs = hard_neg_refresh_epochs
        self.hard_neg_sample_k = hard_neg_sample_k
        self.miner = HardNegativeMiner(pool_size=hard_neg_pool_size)
        self._current_epoch = 0

    def on_epoch_start(self, epoch: int, model: nn.Module | None = None) -> None:
        self._current_epoch = epoch
        if model is None:
            return
        past_warmup = epoch > self.hard_neg_warmup_epochs
        first_mining_epoch = epoch == self.hard_neg_warmup_epochs + 1
        refresh_due = past_warmup and ((epoch - self.hard_neg_warmup_epochs) % self.hard_neg_refresh_epochs == 1)
        if first_mining_epoch or refresh_due:
            self.miner.build(model, self.train_loader, self.device)

    def __call__(
        self,
        model: nn.Module,
        batch: dict,
        train: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        images = batch["local_image_path"].to(self.device)
        captions = batch["caption"]

        outputs = model(images, captions)

        loss_inputs = {
            "image_embeddings": outputs["image_embeddings"],
            "caption_embeddings": outputs["true_caption_embeddings"],
        }
        loss_inputs, n_dropped = drop_nonfinite_rows(loss_inputs, list(loss_inputs))

        if loss_inputs["image_embeddings"].shape[0] == 0:
            return torch.zeros((), device=self.device, requires_grad=True), {
                "n_skipped": images.new_tensor(float(n_dropped))
            }

        img_emb = loss_inputs["image_embeddings"]
        txt_emb = loss_inputs["caption_embeddings"]

        if train and self.miner.ready:
            loss, metrics = self._hard_neg_loss(img_emb, txt_emb)
        else:
            loss, metrics = self.loss_fn({"image_embeddings": img_emb, "caption_embeddings": txt_emb})

        if n_dropped:
            metrics["n_skipped"] = images.new_tensor(float(n_dropped))

        gate = getattr(model.text_model, "nonlinear_gate", None)
        if gate is not None:
            metrics["nonlinear_gate"] = gate.detach()

        return loss, metrics

    def _hard_neg_loss(self, img_emb: Tensor, txt_emb: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """InfoNCE with hard-negative-augmented i2t logits plus standard metrics."""
        img_n = F.normalize(img_emb, dim=-1)
        txt_n = F.normalize(txt_emb, dim=-1)

        logit_scale = self.loss_fn._loss.logit_scale.exp().clamp(min=1.0, max=100.0)
        B = img_n.shape[0]
        labels = torch.arange(B, device=img_n.device)

        sim = logit_scale * (img_n @ txt_n.T)  # [B, B]

        hard_pool = self.miner.sample(self.hard_neg_sample_k, self.device)
        if hard_pool is not None:
            hard_n = F.normalize(hard_pool, dim=-1)
            sim_hard = logit_scale * (img_n @ hard_n.T)  # [B, K]
            sim_i2t = torch.cat([sim, sim_hard], dim=1)  # [B, B+K]
        else:
            sim_i2t = sim

        i2t_loss = F.cross_entropy(sim_i2t, labels)
        t2i_loss = F.cross_entropy(sim.T, labels)
        loss = (i2t_loss + t2i_loss) / 2

        with torch.no_grad():
            pos_sim = sim.diagonal()
            eye = torch.eye(B, dtype=torch.bool, device=sim.device)
            mean_pos = pos_sim.mean()
            mean_neg = sim[~eye].mean()
            hardest_neg = sim.masked_fill(eye, -torch.inf).max(dim=1).values
            hard_neg_accuracy = (pos_sim > hardest_neg).float().mean()

            temperature = 1.0 / logit_scale.clamp(max=100.0)
            accuracy = ((sim.argmax(1) == labels).float() + (sim.T.argmax(1) == labels).float()) / 2

        metrics: dict[str, Tensor] = {
            "loss": loss,
            "infonce_loss": loss,
            "accuracy": accuracy.mean(),
            "cosine_similarity": mean_pos,
            "alignment_gap": mean_pos - mean_neg,
            "hard_neg_accuracy": hard_neg_accuracy,
            "sim_ratio": mean_pos / (mean_neg.abs() + 1e-6),
            "modality_gap": F.cosine_similarity(img_n.mean(0, keepdim=True), txt_n.mean(0, keepdim=True)).squeeze(),
            "temperature": temperature,
        }
        return loss, metrics
