import random
from collections import defaultdict
from typing import Iterator, Sequence

from torch.utils.data import DataLoader, Sampler

from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="topology_bucket_sampler")


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """Call once per epoch on any DataLoader. If it was built with an
    epoch-aware batch_sampler (e.g. TopologyBucketSampler), forwards the epoch
    so batch order/composition varies run-to-run instead of repeating the same
    grouping every epoch. Safe no-op for a plain (default) batch_sampler."""
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)


class TopologyBucketSampler(Sampler[list[int]]):
    """
    Batches dataset indices by shared CCG diagram topology, so a future batched
    contraction can run one call per homogeneous-diagram batch instead of one
    contraction per sample (EinsumModel.forward is currently a per-sample loop).

    Diagrams occurring >= min_bucket_size times get "pure" batches: every row in
    the batch shares the exact same diagram string (the fast-path candidates).
    Diagrams below that threshold are pooled together and split into ordinary
    mixed batches — same shape as today's random batching, handled by the
    existing per-sample loop. Batch order is shuffled across the whole epoch so
    pure and mixed batches are interleaved, not run head-then-tail.

    No truncation or subsampling happens today: tail_fraction=1.0 means every
    tail row is included every epoch, so this is a drop-in, loss-free change of
    batch *composition* only. tail_fraction and set_epoch exist so a later
    revision can shrink the tail to a rotating subset (see _select_tail_indices)
    — guaranteeing full coverage over several epochs instead of every epoch —
    without changing how this class is constructed or used.

    Usage: pass as `batch_sampler` to a DataLoader (not `sampler`/`batch_size`,
    which DataLoader disallows in combination with `batch_sampler`):

        sampler = TopologyBucketSampler(diagrams, batch_size=256)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=...)
        for epoch in range(n_epochs):
            sampler.set_epoch(epoch)
            for batch in loader:
                ...
    """

    def __init__(
        self,
        diagrams: Sequence[str],
        batch_size: int,
        min_bucket_size: int = 64,
        tail_fraction: float = 1.0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if not 0.0 <= tail_fraction <= 1.0:
            raise ValueError(f"tail_fraction must be in [0, 1], got {tail_fraction}")
        if min_bucket_size < 2:
            # A head bucket's total size can never be smaller than min_bucket_size.
            # Below 2, a bucket whose entire size is 1 has no previous chunk to
            # merge its degenerate singleton batch into (see _chunk) — that batch
            # would reach an InfoNCE-style loss and produce NaN (S[~eye] is empty
            # for B=1). No caller in this codebase sets this below the default 64.
            raise ValueError(f"min_bucket_size must be >= 2, got {min_bucket_size}")

        self.batch_size = batch_size
        self.min_bucket_size = min_bucket_size
        self.tail_fraction = tail_fraction
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

        buckets: dict[str, list[int]] = defaultdict(list)
        for idx, diagram in enumerate(diagrams):
            buckets[diagram].append(idx)

        self._head_buckets: dict[str, list[int]] = {
            d: idxs for d, idxs in buckets.items() if len(idxs) >= min_bucket_size
        }
        self._tail_indices: list[int] = [
            idx for idxs in buckets.values() if len(idxs) < min_bucket_size for idx in idxs
        ]

        n_head_rows = sum(len(v) for v in self._head_buckets.values())
        n_total = len(diagrams)
        logger.info(
            f"TopologyBucketSampler: {len(self._head_buckets)}/{len(buckets)} diagrams are "
            f"'head' (>= {min_bucket_size} rows), covering {n_head_rows}/{n_total} rows "
            f"({100 * n_head_rows / n_total:.1f}%). Tail: {len(self._tail_indices)} rows across "
            f"{len(buckets) - len(self._head_buckets)} diagrams (tail_fraction={tail_fraction})."
        )

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch (mirrors DistributedSampler) so batch order — and,
        once tail_fraction < 1.0, which tail rows get sampled — varies
        deterministically across epochs while staying reproducible for a given seed.
        """
        self._epoch = epoch

    def _select_tail_indices(self, rng: random.Random) -> list[int]:
        """Rows from below-threshold diagrams to include this epoch.

        Today this always returns every tail row (no truncation). A later
        revision can sample `tail_fraction` of self._tail_indices here, rotating
        the subset using self._epoch so full coverage is guaranteed over
        multiple epochs — the rest of the class does not need to change.
        """
        if self.tail_fraction >= 1.0:
            return list(self._tail_indices)
        k = round(len(self._tail_indices) * self.tail_fraction)
        return rng.sample(self._tail_indices, k)

    def _n_chunks(self, n: int) -> int:
        if n == 0:
            return 0
        if self.drop_last:
            return n // self.batch_size
        q, r = divmod(n, self.batch_size)
        if r == 0:
            return q
        if r == 1 and q >= 1:
            return q  # remainder of 1 gets merged into the last full chunk, not its own
        return q + 1

    def _chunk(self, indices: list[int]) -> list[list[int]]:
        chunks = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.drop_last:
            if chunks and len(chunks[-1]) < self.batch_size:
                chunks.pop()
            return chunks
        # Avoid a degenerate final batch of size 1: InfoNCE-style losses compute
        # an off-diagonal mean (S[~eye]) which is EMPTY for a batch of 1, and
        # .mean() of an empty tensor is NaN — poisoning that epoch's accumulated
        # metrics. Per-bucket chunking (unlike plain random batching, which only
        # ever produces one undersized batch per epoch) makes this common: every
        # head bucket gets its own remainder. Merge a size-1 remainder into the
        # previous chunk instead of yielding it standalone.
        if len(chunks) > 1 and len(chunks[-1]) < 2:
            chunks[-2].extend(chunks.pop())
        return chunks

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        batches: list[list[int]] = []

        # Fast-path: one or more pure (single-diagram) batches per head bucket.
        for indices in self._head_buckets.values():
            pool = list(indices)
            if self.shuffle:
                rng.shuffle(pool)
            batches.extend(self._chunk(pool))

        # Slow-path: mixed batches pooled from tail rows (all included today).
        tail = self._select_tail_indices(rng)
        if self.shuffle:
            rng.shuffle(tail)
        batches.extend(self._chunk(tail))

        # Interleave pure and mixed batches so the epoch doesn't run all
        # single-topology batches back-to-back followed by all tail batches.
        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        n_head_batches = sum(self._n_chunks(len(idxs)) for idxs in self._head_buckets.values())
        n_tail = round(len(self._tail_indices) * self.tail_fraction)
        return n_head_batches + self._n_chunks(n_tail)
