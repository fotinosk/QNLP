import numpy as np
import polars as pl

from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="contrastive_pair_strategy")

_OUTPUT_COLUMNS = [
    "sample_id",
    "local_image_path",
    "true_diagram",
    "true_symbols",
    "false_diagram",
    "false_symbols",
]

_OUTPUT_SCHEMA = {
    "sample_id": pl.String,
    "local_image_path": pl.String,
    "true_diagram": pl.String,
    "true_symbols": pl.String,
    "false_diagram": pl.String,
    "false_symbols": pl.String,
}


class ContrastivePairStrategy:
    """
    Composes atoms into contrastive pairs: one positive and one negative caption per row.

    Two modes, applied independently and then concatenated:

    Labeled atoms (have a boolean 'label' column — e.g. ARO):
        Groups by sample_id and joins the True atom with the False atom directly.
        The negative is pre-defined by the source dataset.

    Unlabeled atoms (no 'label' column, or label is null — e.g. COCO):
        Every atom is a positive. A random derangement over sample_id groups assigns
        each group a distinct negative group. One atom from the negative group is
        randomly selected as the synthetic negative for all atoms in the positive group.
        Composition is fully vectorised via Polars joins.

    Output schema:
        sample_id, local_image_path, true_diagram, true_symbols, false_diagram, false_symbols
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def compose(self, atoms: pl.DataFrame) -> pl.DataFrame:
        has_labels = "label" in atoms.columns and atoms["label"].drop_nulls().len() > 0

        if has_labels:
            labeled = atoms.filter(pl.col("label").is_not_null())
            unlabeled = atoms.filter(pl.col("label").is_null())
        else:
            labeled = pl.DataFrame(schema=atoms.schema)
            unlabeled = atoms

        parts = []
        if not labeled.is_empty():
            parts.append(self._reconstruct_labeled(labeled))
        if not unlabeled.is_empty():
            synthesized = self._synthesize_negatives(unlabeled)
            if not synthesized.is_empty():
                parts.append(synthesized)

        if not parts:
            return pl.DataFrame(schema=_OUTPUT_SCHEMA)

        return pl.concat(parts, how="diagonal_relaxed").select(_OUTPUT_COLUMNS)

    def _reconstruct_labeled(self, atoms: pl.DataFrame) -> pl.DataFrame:
        """Pair pre-labeled positive and negative atoms by sample_id."""
        positives = (
            atoms.filter(pl.col("label"))
            .select(["sample_id", "local_image_path", "diagram", "symbols"])
            .rename({"diagram": "true_diagram", "symbols": "true_symbols"})
        )
        negatives = (
            atoms.filter(~pl.col("label"))
            .select(["sample_id", "diagram", "symbols"])
            .rename({"diagram": "false_diagram", "symbols": "false_symbols"})
        )
        return positives.join(negatives, on="sample_id", how="inner")

    def _synthesize_negatives(self, atoms: pl.DataFrame) -> pl.DataFrame:
        """
        Assign each sample_id group a random negative group via derangement,
        then join to produce contrastive rows without any per-row Python loops.
        """
        rng = np.random.default_rng(self.seed)
        unique_ids = atoms["sample_id"].unique().to_list()

        if len(unique_ids) < 2:
            logger.warning("Cannot synthesize negatives: fewer than 2 sample_id groups in split.")
            return pl.DataFrame(schema=_OUTPUT_SCHEMA)

        # Random derangement: each sample_id maps to a distinct different sample_id.
        # Rejection sampling converges in ~e ≈ 2.7 iterations on average.
        neg_ids = unique_ids.copy()
        while any(a == b for a, b in zip(unique_ids, neg_ids)):
            rng.shuffle(neg_ids)

        mapping = pl.DataFrame(
            {
                "sample_id": unique_ids,
                "neg_sample_id": neg_ids,
            }
        )

        # Randomly select one atom per group to serve as the negative representative.
        # Shuffle first so that group_by().first() gives a random atom per group.
        neg_pool = (
            atoms.sample(fraction=1.0, shuffle=True, seed=int(rng.integers(0, 2**31)))
            .group_by("sample_id")
            .first()
            .select(["sample_id", "diagram", "symbols"])
            .rename(
                {
                    "sample_id": "neg_sample_id",
                    "diagram": "false_diagram",
                    "symbols": "false_symbols",
                }
            )
        )

        return (
            atoms.select(["sample_id", "local_image_path", "diagram", "symbols"])
            .rename({"diagram": "true_diagram", "symbols": "true_symbols"})
            .join(mapping, on="sample_id", how="left")
            .join(neg_pool, on="neg_sample_id", how="left")
            .drop("neg_sample_id")
        )
