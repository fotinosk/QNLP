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
        return df.explode(self.list_column)


class SchemaMappingStep:
    def __init__(self, column_mapping: dict[str, str]):
        """
        Maps arbitrary raw columns to standard target columns.
        e.g., {"raw_caption_column": "processed_text"}
        """
        self.column_mapping = column_mapping

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        rename_dict = {k: v for k, v in self.column_mapping.items() if k in df.columns}
        if rename_dict:
            df = df.rename(rename_dict)
        return df


class RemoveTrailingDotsStep:
    """
    Removes trailing dots from sentences to avoid CCG parsing errors
    like 'Bobcat failed to parse'.
    """

    def __init__(self, text_column: str = "processed_text"):
        self.text_column = text_column

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.text_column not in df.columns:
            return df

        return df.with_columns(pl.col(self.text_column).str.replace(r"\.+$", "").alias(self.text_column))
