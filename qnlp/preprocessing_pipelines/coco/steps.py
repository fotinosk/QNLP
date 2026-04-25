"""
Coco pipeline:
    flatten dataset > confrom schema > cleanup to ensure vectors not tensors > einsum step and db registration
"""

import polars as pl


class COCOFlattenStep:
    """
    Explodes/flattens a list of strings (e.g., multiple captions per image)
    so that each caption gets its own distinct row, duplicating the image path.
    """

    def __init__(self, list_column: str = "sentences"):
        self.list_column = list_column

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.list_column in df.columns:
            # explode() duplicates all other columns for each item in the list
            return df.explode(self.list_column)
        return df
