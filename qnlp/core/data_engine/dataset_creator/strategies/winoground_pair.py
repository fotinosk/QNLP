import polars as pl

from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="winoground_pair_strategy")

_OUTPUT_COLUMNS = [
    "pair_id",
    "local_image_0_path",
    "local_image_1_path",
    "cap0_diagram",
    "cap0_symbols",
    "cap1_diagram",
    "cap1_symbols",
]


class WinogroundPairStrategy:
    """
    Reconstructs Winoground 2x2 pairs from flattened pipeline atoms.

    The pipeline flattens each Winoground item into two atoms with sample_ids
    suffixed __0 and __1, where:
        __0 -> local_image_0_path, caption_0 compiled diagram/symbols
        __1 -> local_image_1_path, caption_1 compiled diagram/symbols

    This strategy strips the suffix to recover pair_id, splits atoms by
    caption index, and inner-joins them back into one row per pair.

    Output schema:
        pair_id, local_image_0_path, local_image_1_path,
        cap0_diagram, cap0_symbols, cap1_diagram, cap1_symbols
    """

    def compose(self, atoms: pl.DataFrame) -> pl.DataFrame:
        atoms = atoms.with_columns(
            [
                pl.col("sample_id").str.split("__").list.first().alias("pair_id"),
                pl.col("sample_id").str.split("__").list.last().alias("caption_index"),
            ]
        )

        cap0 = (
            atoms.filter(pl.col("caption_index") == "0")
            .select(["pair_id", "local_image_path", "diagram", "symbols"])
            .rename(
                {
                    "local_image_path": "local_image_0_path",
                    "diagram": "cap0_diagram",
                    "symbols": "cap0_symbols",
                }
            )
        )

        cap1 = (
            atoms.filter(pl.col("caption_index") == "1")
            .select(["pair_id", "local_image_path", "diagram", "symbols"])
            .rename(
                {
                    "local_image_path": "local_image_1_path",
                    "diagram": "cap1_diagram",
                    "symbols": "cap1_symbols",
                }
            )
        )

        paired = cap0.join(cap1, on="pair_id", how="inner")

        dropped = len(cap0) - len(paired)
        if dropped:
            logger.warning(
                f"Dropped {dropped} pairs where one atom was missing " "(CCG compilation failure or split mismatch)."
            )

        return paired.select(_OUTPUT_COLUMNS)
