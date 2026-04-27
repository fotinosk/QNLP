import polars as pl

from qnlp.core.data_engine.processing.pipeline import PipelineStep
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="conform_rank_step")


class UnifyEinsumRankStep(PipelineStep):
    """
    Ensures that all compiled CCG tensor networks output a Rank-1 tensor (a vector).
    If the CCG parser resolves a sentence to a complex type (multiple open wires),
    this step modifies the einsum string to trace out (sum over) the extra open indices,
    leaving exactly one output index.
    """

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Starting UnifyEinsumRankStep for chunk of size {len(df)}")
        if "compiled_bytes" not in df.columns:
            return df

        # Use Polars native Rust-based regex engine to find and truncate the diagram output
        # directly on the JSON string. This avoids the massive Python overhead of JSON
        # deserialising -> mutating -> serialising every single row.

        # Regex explanation:
        # Group 1: Matches exactly the diagram key up to the arrow ('"diagram": "...->')
        # Group 2: Matches the FIRST character of the output indices ('a')
        # '[a-zA-Z]+': Matches the REST of the output indices (only if > 1 exist)
        # Group 3: Matches the closing quote ('"')
        # Replacement: Keeps only Group 1, Group 2, and Group 3, effectively deleting the rest.

        return df.with_columns(
            pl.col("compiled_bytes")
            .cast(pl.String)
            .str.replace(r'("diagram"\s*:\s*"[^"]*->)([a-zA-Z])[a-zA-Z]+(")', "${1}${2}${3}")
            .cast(pl.Binary)
            .alias("compiled_bytes")
        )
