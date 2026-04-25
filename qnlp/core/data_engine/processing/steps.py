from typing import Dict, Protocol

import polars as pl


class PipelineStep(Protocol):
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Process a dataframe chunk and return the modified dataframe."""
        ...


class SchemaMappingStep:
    def __init__(self, column_mapping: Dict[str, str]):
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


class CleanTextStep:
    def __init__(self, target_column: str = "processed_text"):
        self.target_column = target_column

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.target_column in df.columns:
            df = df.with_columns(pl.col(self.target_column).str.strip_chars().str.to_lowercase())
        return df
