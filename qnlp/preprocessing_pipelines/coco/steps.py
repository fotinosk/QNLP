import polars as pl

from qnlp.core.data_engine.processing.common_steps import RemoveTrailingDotsStep, SchemaMappingStep

__all__ = ["COCOFlattenStep", "SchemaMappingStep", "RemoveTrailingDotsStep"]


class COCOFlattenStep:
    """
    Explodes a list column (multiple captions per image) so each caption
    gets its own row, duplicating the image path.
    """

    def __init__(self, list_column: str = "sentences"):
        self.list_column = list_column

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.explode(self.list_column)
